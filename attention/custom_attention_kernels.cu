#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

//#define TIME_FLOPS

template <typename scalar_t>
__global__ void custom_transpose_forward_kernel(const scalar_t* __restrict__ input,
                                                scalar_t* __restrict__ output,
                                                size_t batch_size,
                                                size_t seq_size,
                                                size_t embed_size) {
    
    // Calculate global thread indices
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int z = blockIdx.z * blockDim.z + threadIdx.z; // batch index

    size_t batch_elem_size = seq_size * embed_size;

    if (x < seq_size && y < embed_size && z < batch_size) {
        output[z * batch_elem_size + y * seq_size + x] = input[z * batch_elem_size + x * embed_size + y];
    }
}

torch::Tensor custom_transpose_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "inputs must be fp32");

    const int64_t batch_size = input.size(0);
    const int64_t seq_size = input.size(1);
    const int64_t embed_size = input.size(2);

    auto opts = input.options();
    torch::Tensor output = torch::empty({batch_size, embed_size, seq_size}, opts);

    dim3 block_size(32, 32, 1);
    dim3 grid_size((seq_size - 1) / block_size.x + 1, 
                   (embed_size - 1) / block_size.y + 1,
                   batch_size);

    custom_transpose_forward_kernel<float><<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_size,
        embed_size
    );
    
    return output;
}


template <typename scalar_t>
__global__ void custom_matmul_forward_kernel(const scalar_t* __restrict__ A,
                                             const scalar_t* __restrict__ B,
                                             scalar_t* __restrict__ C,
                                             size_t b_size,
                                             size_t m_size,
                                             size_t n_size, 
                                             size_t k_size) {
    
    // Calculate global thread index within the batch dimension
    int batch_idx = blockIdx.z;

    // Calculate global thread index within the output matrix C for the current batch
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread is within the bounds of the output matrix C
    if (row < m_size && col < n_size && batch_idx < b_size) {
        float sum = 0.0f;
        // Calculate the starting offset for the current batch's matrices
        int A_offset = batch_idx * m_size * k_size;
        int B_offset = batch_idx * k_size * n_size;
        int C_offset = batch_idx * m_size * n_size;

        // Perform the dot product for the current element C[row][col]
        for (int k = 0; k < k_size; ++k) {
            sum += A[A_offset + row * k_size + k] * B[B_offset + k * n_size + col];
        }
        C[C_offset + row * n_size + col] = sum;
    }
}


torch::Tensor custom_matmul_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "inputs must be fp32");
    TORCH_CHECK(b.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "inputs must be fp32");

    const int64_t batch_size = a.size(0);
    const int64_t m_size = a.size(1);
    const int64_t k_size = a.size(2);
    const int64_t n_size = b.size(2);

    auto opts = a.options();
    torch::Tensor c = torch::empty({batch_size, m_size, n_size}, opts);

    dim3 block_size(32, 32, 1);
    dim3 grid_size((n_size - 1) / block_size.x + 1, 
                   (m_size - 1) / block_size.y + 1,
                   batch_size);

    #ifdef TIME_FLOPS
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.stream());
    #endif

    custom_matmul_forward_kernel<float><<<grid_size, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        batch_size,
        m_size,
        n_size,
        k_size
    );

    #ifdef TIME_FLOPS
    cudaEventRecord(stop, stream.stream());
    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    double sec_per_call = elapsed_time_ms/1e3;
    double flops_per_call = 2.0 * static_cast<double>(batch_size) * static_cast<double>(m_size) * static_cast<double>(k_size) * static_cast<double>(n_size);
    std::cout << "NAIVE matmul sizes: [" << m_size << "x" << k_size << "] * [" << k_size << "x" << n_size << "]" << std::endl;
    std::cout << "NAIVE matmul TFLOP/s: " << flops_per_call / (sec_per_call*1e12) << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
    
    return c;
}

template <typename scalar_t>
__global__ void custom_softmax_forward_kernel(const scalar_t* __restrict__ input,
                                              scalar_t* __restrict__ output,
                                              size_t b_size,
                                              size_t h_size,
                                              size_t s_size) {
    
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;

    if(b_idx < b_size && h_idx < h_size) {
        scalar_t max_val = 0.0;
        int row_offset = s_size * (b_idx * h_size + h_idx);
        for(int i = 0; i < s_size; i++) {
            max_val = input[row_offset + i] > max_val ? input[row_offset + i] : max_val;
        }

        scalar_t sum = 0.0;
        for(int i = 0; i < s_size; i++) {
            scalar_t e = __expf(input[row_offset + i] - max_val);
            output[row_offset + i] = e;
            sum += e;
        }

        for(int i = 0; i < s_size; i++) {
            output[row_offset + i] /= sum;
        }
    }
}


torch::Tensor custom_softmax_forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "inputs must be fp32");

    const int64_t batch_size = input.size(0);
    const int64_t h_size = input.size(1);
    const int64_t s_size = input.size(2);

    auto opts = input.options();
    torch::Tensor output = torch::empty({batch_size, h_size, s_size}, opts);

    dim3 block_size(1, 1);
    dim3 grid_size(batch_size, h_size);

    custom_softmax_forward_kernel<float><<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        h_size,
        s_size
    );
    
    return output;
}


template <typename scalar_t>
__global__ void custom_eltwise_div_kernel(const scalar_t* __restrict__ input,
                                          scalar_t* __restrict__ output,
                                          scalar_t val,
                                          size_t num_elem) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elem) {
        output[idx] = input[idx] / val;
    }
}


torch::Tensor custom_eltwise_div_forward(torch::Tensor input, float val) {
    TORCH_CHECK(input.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "inputs must be fp32");

    const int64_t num_elements = input.numel();

    auto opts = input.options();
    torch::Tensor output = torch::empty({input.size(0), input.size(1), input.size(2)}, opts);

    dim3 block_size(1024);
    dim3 grid_size((num_elements - 1)/block_size.x + 1);

    custom_eltwise_div_kernel<float><<<grid_size, block_size>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        val,
        num_elements
    );
    
    return output;
}

// ------------------------------------ optimize --------------------------------------------------

torch::Tensor qkt_cublas_forward(torch::Tensor Q, torch::Tensor K, float scale) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda(), "CUDA tensors required");
    TORCH_CHECK(Q.dtype()==torch::kFloat32 && K.dtype()==torch::kFloat32, "fp32 only here");

    const int64_t B = Q.size(0), S = Q.size(1), D = Q.size(2);

    const int m = S, n = S, k = D;
    const int lda = k;
    const int ldb = k;
    const int ldc = n;
    const long long strideA = static_cast<long long>(S) * static_cast<long long>(D);
    const long long strideB = static_cast<long long>(S) * static_cast<long long>(D);
    const long long strideC = static_cast<long long>(S) * static_cast<long long>(S);

    auto opts = Q.options();
    torch::Tensor scores = torch::empty({B, S, S}, opts);

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    const float alpha = scale;
    const float beta  = 0.f;

    // Enable Tensor Cores for fp32 via TF32 (Ampere+):
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    cudaDataType_t T = CUDA_R_32F;

    #ifdef TIME_FLOPS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    #endif

    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N, // we do a small trick here. transpose Q instead of T (since row-col major), and K is already transposed
        n, m, k,                      // (S, S, D)
        &alpha,
        K.data_ptr<float>(), T, lda, strideB,
        Q.data_ptr<float>(), T, lda, strideA,
        &beta,
        scores.data_ptr<float>(), T, ldc, strideC,
        static_cast<int>(B),
        compute,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    #ifdef TIME_FLOPS
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    double sec_per_call = elapsed_time_ms/1e3;
    double flops_per_call = 2.0 * static_cast<double>(B) * static_cast<double>(S) * static_cast<double>(S) * static_cast<double>(D);
    std::cout << "qkt sizes: [" << S << "x" << D << "] * [" << D << "x" << S << "]" << std::endl;
    std::cout << "qkt TFLOP/s: " << flops_per_call / (sec_per_call*1e12) << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif

    return scores;
}

torch::Tensor pv_cublas_forward(torch::Tensor Probs, torch::Tensor V) {
    TORCH_CHECK(Probs.is_cuda() && V.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(Probs.dtype() == torch::kFloat32 && V.dtype() == torch::kFloat32, "fp32 only (v0)");
    TORCH_CHECK(Probs.dim() == 3 && V.dim() == 3, "Probs,V must be [B,S,S] and [B,S,D]");
    TORCH_CHECK(Probs.is_contiguous() && V.is_contiguous(), "inputs must be contiguous");
    TORCH_CHECK(Probs.size(0) == V.size(0) && Probs.size(1) == V.size(1),
                "batch and S must match: Probs=[B,S,S], V=[B,S,D]");

    const int64_t B = Probs.size(0);
    const int64_t S = Probs.size(1);
    const int64_t D = V.size(2);

    // Output: [B, S, D] (row-major)
    auto opts = V.options();
    torch::Tensor Out = torch::empty({B, S, D}, opts);

    // Make sure we execute on the right device/stream
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    auto stream = at::cuda::getCurrentCUDAStream();
    cublasSetStream(handle, stream);

    // Row-major trick:
    // Compute C_col(D×S) = A_col(D×S) * B_col(S×S) with:
    //   A = V,  opA = N  -> A_col = V_row^T (D×S)
    //   B = P,  opB = T  -> B_col = (Probs_row^T)^T? (we need P_row^T), opB=T gives P_row^T (S×S)
    // Result C_col(D×S) maps to Out_row(S×D) in memory.
    const int m = static_cast<int>(D);  // rows of C_col
    const int n = static_cast<int>(S);  // cols of C_col
    const int k = static_cast<int>(S);

    const int lda = static_cast<int>(D);  // rows of A_col (V_row^T)
    const int ldb = static_cast<int>(S);  // rows of B_col (P_col)
    const int ldc = static_cast<int>(D);  // rows of C_col

    const long long strideA = static_cast<long long>(S) * static_cast<long long>(D); // V
    const long long strideB = static_cast<long long>(S) * static_cast<long long>(S); // Probs
    const long long strideC = static_cast<long long>(S) * static_cast<long long>(D); // Out

    const float alpha = 1.f;
    const float beta  = 0.f;

    // Fast TF32 (Ampere+) or switch to CUBLAS_COMPUTE_32F for strict fp32
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F_FAST_TF32;
    const cudaDataType T = CUDA_R_32F;

    #ifdef TIME_FLOPS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    #endif

    cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,           // A=V (N), B=Probs (T)
        m, n, k,                            // (D, S, S)
        &alpha,
        V.data_ptr<float>(),     T, lda, strideA,   // A
        Probs.data_ptr<float>(), T, ldb, strideB,   // B
        &beta,
        Out.data_ptr<float>(),   T, ldc, strideC,   // C (row-major [S,D] as col-major [D,S])
        static_cast<int>(B),
        compute,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    #ifdef TIME_FLOPS
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    double sec_per_call = elapsed_time_ms/1e3;
    double flops_per_call = 2.0 * static_cast<double>(B) * static_cast<double>(D) * static_cast<double>(S) * static_cast<double>(S);
    std::cout << "   pv sizes: [" << D << "x" << S << "] * [" << S << "x" << S << "]" << std::endl;
    std::cout << "   pv TFLOP/s: " << flops_per_call / (sec_per_call*1e12) << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif

    return Out;
}


#define FULL_MASK 0xffffffff
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024

__device__ __forceinline__ float warp_reduce_max(float val) {
    // Iterate over log2(warpSize) steps
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val; // The maximum value will be in thread 0 of the warp
}

__device__ float warp_reduce_sum(float val) {
    // Perform a tree reduction using __shfl_down_sync
    // Threads exchange values with threads at a decreasing offset
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void opt_softmax_forward_kernel(scalar_t* __restrict__ data,
                                           size_t b_size,
                                           size_t h_size,
                                           size_t s_size) {
    
    int b_idx = blockIdx.x;
    int h_idx = blockIdx.y;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    __shared__ scalar_t reduce_max_ws[MAX_BLOCK_SIZE / WARP_SIZE];
    __shared__ scalar_t reduce_sum_ws[MAX_BLOCK_SIZE / WARP_SIZE];

    reduce_max_ws[lane] = std::numeric_limits<scalar_t>::lowest();
    reduce_sum_ws[lane] = 0.0;
    __syncthreads();

    if(b_idx < b_size && h_idx < h_size) {
        scalar_t max_val = std::numeric_limits<scalar_t>::lowest();
        int row_offset = s_size * (b_idx * h_size + h_idx);
        for(int i = tid; i < s_size; i += block_size) {
            max_val = data[row_offset + i] > max_val ? data[row_offset + i] : max_val;
        }
        scalar_t max_in_warp = warp_reduce_max(max_val);
        if(lane == 0) { // each warp writes its max
            reduce_max_ws[warp_id] = max_in_warp;
        }
        __syncthreads();

        // to obtain global max inside each warp
        max_val = warp_reduce_max(reduce_max_ws[lane]);

        // broadcast global max to all threads inside warp
        max_val = __shfl_sync(FULL_MASK, max_val, 0);

        scalar_t sum = 0.0;
        for(int i = tid; i < s_size; i += block_size) {
            scalar_t e = __expf(data[row_offset + i] - max_val);
            data[row_offset + i] = e;
            sum += e;
        }
        scalar_t sum_in_warp = warp_reduce_sum(sum);
        if(lane == 0) {
            reduce_sum_ws[warp_id] = sum_in_warp;
        }
        __syncthreads();

        sum = warp_reduce_sum(reduce_sum_ws[lane]);

        sum = __shfl_sync(FULL_MASK, sum, 0);

        for(int i = tid; i < s_size; i += block_size) {
            data[row_offset + i] /= sum;
        }
    }
}


void opt_softmax_forward(torch::Tensor data) {
    TORCH_CHECK(data.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(data.dtype() == torch::kFloat32, "inputs must be fp32");

    const int64_t batch_size = data.size(0);
    const int64_t h_size = data.size(1);
    const int64_t s_size = data.size(2);

    dim3 block_size(std::min(MAX_BLOCK_SIZE, static_cast<int>(s_size)), 1);
    dim3 grid_size(batch_size, h_size);

    opt_softmax_forward_kernel<float><<<grid_size, block_size>>>(
        data.data_ptr<float>(),
        batch_size,
        h_size,
        s_size
    );
}

////////////////////////////////////////////// opt matmul /////////////////////////////////

#define TILE_M 128
#define TILE_K 16
#define TILE_N 96

#define BX 16
#define BY 16

#define MICRO_M (TILE_M/BX) // 
#define MICRO_N (TILE_N/BY) // changing 8 -> 6 gave +1.5 tflop

static_assert(BX == TILE_N / MICRO_N, "BX must me = TILE_N / MICRO_N");
static_assert(BY == TILE_M / MICRO_M, "BY must me = TILE_M / MICRO_M");

template <typename scalar_t>
__global__ __launch_bounds__(BX*BY, 2)
__global__ void opt_matmul_forward_kernel(const scalar_t* __restrict__ A,
                                          const scalar_t* __restrict__ B,
                                          scalar_t* __restrict__ C,
                                          const int B_size,
                                          const int M_size,
                                          const int N_size, 
                                          const int K_size) {
    
    // Calculate global thread index within the batch dimension
    int lda = K_size;
    int ldb = N_size;
    int ldc = N_size;

    int batch_idx = blockIdx.z;
    int A_offset = batch_idx * M_size * K_size;
    int B_offset = batch_idx * K_size * N_size;
    int C_offset = batch_idx * M_size * N_size;

    __shared__ float A_shared[TILE_M][TILE_K + 1];
    __shared__ float B_shared[TILE_K][TILE_N + 1];

    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    int block_size = blockDim.x * blockDim.y;

    float C_reg[MICRO_M][MICRO_N];
    #pragma unroll
    for(int i = 0; i < MICRO_M; i++) 
        #pragma unroll
        for(int j = 0; j < MICRO_N; j++)
            C_reg[i][j] = 0.0;
    
    for(int k_start = 0; k_start < K_size; k_start += TILE_K) {
        // load A tile
        #pragma unroll
        for(int i = tid; i < TILE_M * TILE_K; i += block_size) {
            int local_row = i / TILE_K;
            int local_col = i % TILE_K;
            int global_col = k_start + local_col;
            int global_row = TILE_M * blockIdx.y + local_row;
            int a_idx = global_col + global_row * lda + A_offset;
            float val = 0;
            if(global_col < K_size && global_row < M_size)
                val = A[a_idx];
            A_shared[local_row][local_col] = val;
        }

        // load B tile
        #pragma unroll
        for(int i = tid; i < TILE_N * TILE_K; i += block_size) {
            int local_row = i / TILE_N;
            int local_col = i % TILE_N;
            int global_col = TILE_N * blockIdx.x + local_col;
            int global_row = k_start + local_row;
            int b_idx = global_col + global_row * ldb + B_offset;
            float val = 0;
            if(global_col < N_size && global_row < K_size)
                val = B[b_idx];
            B_shared[local_row][local_col] = val;
        }

        __syncthreads();

        #pragma unroll
        for(int k = 0; k < TILE_K; k++) {
            float a_reg[MICRO_M];
            float b_reg[MICRO_N];

            #pragma unroll
            for(int i = 0; i < MICRO_M; i++) { // this is first part of column of A (for first tile)
                a_reg[i] = A_shared[i + MICRO_M * threadIdx.y][k];
            }

            #pragma unroll
            for(int i = 0; i < MICRO_N; i++) { // this is first part of row of B (for first tile)
                b_reg[i] = B_shared[k][i + MICRO_N * threadIdx.x];
            }

            #pragma unroll
            for(int i = 0; i < MICRO_M; i++) {
                for(int j = 0; j < MICRO_N; j++) {
                    C_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int i = 0; i < MICRO_M; i++) {
        #pragma unroll
        for(int j = 0; j < MICRO_N; j++) {
            int c_row = MICRO_M * threadIdx.y + blockIdx.y * TILE_M + i;
            int c_col = MICRO_N * threadIdx.x + blockIdx.x * TILE_N + j;
            int c_idx = c_col + ldc * c_row + C_offset;
            if(c_col < N_size && c_row < M_size)
                C[c_idx] = C_reg[i][j];
        }
    }
}


torch::Tensor opt_matmul_forward(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "inputs must be fp32");
    TORCH_CHECK(b.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "inputs must be fp32");

    const int64_t batch_size = a.size(0);
    const int64_t m_size = a.size(1);
    const int64_t k_size = a.size(2);
    const int64_t n_size = b.size(2);

    auto opts = a.options();
    torch::Tensor c = torch::empty({batch_size, m_size, n_size}, opts);

    dim3 block_size(BX, BY, 1);
    dim3 grid_size((n_size - 1) / (BX * MICRO_N) + 1, 
                   (m_size - 1) / (BY * MICRO_M) + 1,
                   batch_size);

    #ifdef TIME_FLOPS
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.stream());
    #endif

    opt_matmul_forward_kernel<float><<<grid_size, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        batch_size,
        m_size,
        n_size,
        k_size
    );

    #ifdef TIME_FLOPS
    cudaEventRecord(stop, stream.stream());
    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    double sec_per_call = elapsed_time_ms/1e3;
    double flops_per_call = 2.0 * static_cast<double>(batch_size) * static_cast<double>(m_size) * static_cast<double>(k_size) * static_cast<double>(n_size);
    std::cout << "OPT matmul sizes: [" << m_size << "x" << k_size << "] * [" << k_size << "x" << n_size << "]" << std::endl;
    std::cout << "OPT matmul TFLOP/s: " << flops_per_call / (sec_per_call*1e12) << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
    
    return c;
}

////////////////////////////////////////////// fused with online softmax and transpose /////////////////////////////////

#define TILE_M 128
#define TILE_K 16
#define TILE_N 128

#define BX 16
#define BY 16

#define MICRO_M 8 // 128/16
#define MICRO_N 8 // 128/16

static_assert(BX == TILE_N / MICRO_N, "BX must me = TILE_N / MICRO_N");
static_assert(BY == TILE_M / MICRO_M, "BY must me = TILE_M / MICRO_M");

#define TILE_DV (BX * MICRO_N)

template <typename scalar_t>
__global__ void flash_attention_kernel(const scalar_t* __restrict__ Q,
                                       const scalar_t* __restrict__ K,
                                       const scalar_t* __restrict__ V,
                                       scalar_t* __restrict__ O,
                                       size_t B_size,
                                       size_t M_size,
                                       size_t N_size, 
                                       size_t K_size,
                                       size_t DV_size,
                                       scalar_t scale) {
    
    // Calculate global thread index within the batch dimension
    int ldq = K_size; // Q [MxK]
    int ldk = K_size; // K [NxK] -> [KxN]
    int ldv = DV_size; // V size [N x dv]
    int ldo = DV_size; // O size [M x dv]

    int batch_idx = blockIdx.z;
    int Q_offset = batch_idx * M_size * K_size;
    int K_offset = batch_idx * K_size * N_size;
    int V_offset = batch_idx * N_size * DV_size;
    int O_offset = batch_idx * M_size * DV_size;

    __shared__ float Q_shared[TILE_M][TILE_K + 1];
    __shared__ float K_shared[TILE_K][TILE_N + 1];

    int tid = threadIdx.x + blockDim.x * threadIdx.y;
    int block_size = blockDim.x * blockDim.y;

    // Online softmax per-row state for the 128 rows owned by this CTA
    __shared__ float row_m[TILE_M];                  // running max
    __shared__ float row_l[TILE_M];                  // running sum of exp(score - m)

    // Reduction scratch across THREADS_X (=BX) for 128 rows
    __shared__ float tile_row_max[TILE_M][BX];       // store per-thread partial maxima
    __shared__ float tile_row_sum[TILE_M][BX];       // store per-thread partial sums

    // 8-wide slices for E8·V8 micro-GEMMs (K=8)
    __shared__ float E8[TILE_M][MICRO_N];            // 128 x 8, exp(score - new_m)
    __shared__ float V8[MICRO_N][TILE_DV + 1];       // 8 x TILE_DV (+1 pad)

    if(threadIdx.x == 0) {
        for(int i = 0; i < TILE_M; i++) {
            row_m[i] = std::numeric_limits<float>::min();
            row_l[i] = 0;
        }
    }

    // This CTA writes O for rows [M tile] and Dv columns [dv_start : dv_start+TILE_DV)
    const int dv_start = blockIdx.x * TILE_DV;

    __syncthreads();

    float O_reg[MICRO_M][MICRO_N];
    for (int i = 0; i < MICRO_M; ++i)
        for (int j = 0; j < MICRO_N; ++j)
            O_reg[i][j] = 0.f;

    // loop to replace N-side parallelism over C
    for(int n_start = 0; n_start < N_size; n_start += TILE_N) {
        // each block on this loop should be accumulated separatly and written in different locations of C
        float C_reg[MICRO_M][MICRO_N];
        #pragma unroll
        for(int i = 0; i < MICRO_M; i++) 
            #pragma unroll
            for(int j = 0; j < MICRO_N; j++)
                C_reg[i][j] = 0.0;

        for(int k_start = 0; k_start < K_size; k_start += TILE_K) {
            // load Q tile
            #pragma unroll
            for(int i = tid; i < TILE_M * TILE_K; i += block_size) {
                int local_row = i / TILE_K;
                int local_col = i % TILE_K;
                int global_col = k_start + local_col;
                int global_row = TILE_M * blockIdx.y + local_row;
                int q_idx = global_col + global_row * ldq + Q_offset;
                float val = 0;
                if(global_col < K_size && global_row < M_size)
                    val = Q[q_idx];
                Q_shared[local_row][local_col] = val;
            }

            // load Q tile
            #pragma unroll
            for(int i = tid; i < TILE_N * TILE_K; i += block_size) {
                int local_k = i / TILE_N;
                int local_n = i % TILE_N;
                int global_k = k_start + local_k;
                int global_n = n_start + local_n;
                int k_idx = global_k + global_n * ldk + K_offset; // can we just do global_row + global_col * ldb + B_offset here?
                float val = 0;
                if(global_n < N_size && global_k < K_size)
                    val = K[k_idx];
                K_shared[local_k][local_n] = val; // it is already transposed
            }

            __syncthreads();

            #pragma unroll
            for(int k = 0; k < TILE_K; k++) {
                float q_reg[MICRO_M];
                float k_reg[MICRO_N];

                #pragma unroll
                for(int i = 0; i < MICRO_M; i++) { // this is first part of column of A (for first tile)
                    q_reg[i] = Q_shared[i + MICRO_M * threadIdx.y][k];
                }

                #pragma unroll
                for(int i = 0; i < MICRO_N; i++) { // this is first part of row of B (for first tile)
                    k_reg[i] = K_shared[k][i + MICRO_N * threadIdx.x];
                }

                #pragma unroll
                for(int i = 0; i < MICRO_M; i++) {
                    for(int j = 0; j < MICRO_N; j++) {
                        C_reg[i][j] += q_reg[i] * k_reg[j];
                    }
                }
            }

            __syncthreads();
        }

        // Scale scores by 1/sqrt(d)
        #pragma unroll
        for(int i = 0; i < MICRO_M; i++) {
            #pragma unroll
            for(int j = 0; j < MICRO_N; j++) {
                C_reg[i][j] *= scale;
            }
        }

        // ---- Row-wise tile max → new_m
        float row_max_local[MICRO_M];
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) 
            row_max_local[i] = -INFINITY;

        // max reduction in each row
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i)
            #pragma unroll
            for (int j = 0; j < MICRO_N; ++j)
                row_max_local[i] = fmaxf(row_max_local[i], C_reg[i][j]);

        // prepare for reduction withing block
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) {
            int r = threadIdx.y * MICRO_M + i;
            tile_row_max[r][threadIdx.x] = row_max_local[i];
        }
        __syncthreads();

        // reduction of max in C inside block
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int g = 0; g < MICRO_M; ++g) {
                int r = threadIdx.y * MICRO_M + g;
                float mx = -INFINITY;
                #pragma unroll
                for (int c = 0; c < BX; ++c) 
                    mx = fmaxf(mx, tile_row_max[r][c]);
                tile_row_max[r][0] = fmaxf(row_m[r], mx); // new_m
            }
        }
        __syncthreads();

        float new_m_local[MICRO_M];
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) {
            int r = threadIdx.y * MICRO_M + i;
            new_m_local[i] = tile_row_max[r][0];
        }

        // ---- Rescale Ō by exp(old_m - new_m)
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) {
            int r = threadIdx.y * MICRO_M + i;
            float rescale = __expf(row_m[r] - new_m_local[i]);
            #pragma unroll
            for (int j = 0; j < MICRO_N; ++j)
                O_reg[i][j] *= rescale;
        }

        // zero partial row-sum slots
        #pragma unroll
        for (int i = 0; i < MICRO_M; ++i) {
            int r = threadIdx.y * MICRO_M + i;
            tile_row_sum[r][threadIdx.x] = 0.f;
        }
        __syncthreads();


        // ---- Stream the 128-key tile in 8-wide stripes and do E8·V8
        for (int owner = 0; owner < BX; ++owner) {
            // owner writes E8 = exp(score - new_m) for its 8 columns
            if (threadIdx.x == owner) {
                #pragma unroll
                for (int i = 0; i < MICRO_M; ++i) {
                    int r = threadIdx.y * MICRO_M + i;
                    float sum_part = 0.f;
                    #pragma unroll
                    for (int j = 0; j < MICRO_N; ++j) {
                        float e = __expf(C_reg[i][j] - new_m_local[i]);
                        E8[r][j] = e;
                        sum_part += e;
                    }
                    tile_row_sum[r][threadIdx.x] += sum_part; // accumulate per stripe
                }
            }
            __syncthreads();

            // load V8 for these 8 keys into shared: V8[8 x TILE_DV]
            #pragma unroll
            for (int ii = tid; ii < MICRO_N * TILE_DV; ii += block_size) {
                int j8 = ii / TILE_DV;            // 0..7
                int dv = ii % TILE_DV;            // 0..TILE_DV-1
                int gk = n_start + owner * MICRO_N + j8; // global key index
                int gdv = dv_start + dv;                 // global dv column
                float v = 0.f;
                if (gk < (int)N_size && gdv < (int)DV_size) {
                    size_t vidx = (size_t)gk * ldv + gdv + V_offset;
                    v = V[vidx];
                }
                V8[j8][dv] = v;
            }
            __syncthreads();

            // Ō += E8 · V8  (K=8 micro-GEMM)
            #pragma unroll
            for (int i = 0; i < MICRO_M; ++i) {
                int r = threadIdx.y * MICRO_M + i;
                #pragma unroll
                for (int j = 0; j < MICRO_N; ++j) {   // per-thread dv micro-columns
                    int dv_local = j;                 // 0..7
                    int dv_col   = threadIdx.x * MICRO_N + dv_local; // 0..TILE_DV-1
                    if (dv_start + dv_col < (int)DV_size) {
                        float acc = O_reg[i][dv_local];
                        #pragma unroll
                        for (int k8 = 0; k8 < MICRO_N; ++k8) {
                            acc = fmaf(E8[r][k8], V8[k8][dv_col], acc);
                        }
                        O_reg[i][dv_local] = acc;
                    }
                }
            }
            __syncthreads();
        } // end stripes

        // ---- Reduce row sums across BX and update (l, m)
        if (threadIdx.x == 0) {
            #pragma unroll
            for (int g = 0; g < MICRO_M; ++g) {
                int r = threadIdx.y * MICRO_M + g;
                float tsum = 0.f;
                #pragma unroll
                for (int c = 0; c < BX; ++c) tsum += tile_row_sum[r][c];
                float old_m = row_m[r];
                float new_m = new_m_local[g];
                row_l[r] = row_l[r] * __expf(old_m - new_m) + tsum;
                row_m[r] = new_m;
            }
        }
        __syncthreads();
    }

    // ---- finalize: divide by l and store O tile
    #pragma unroll
    for (int i = 0; i < MICRO_M; ++i) {
        int gm = blockIdx.y * TILE_M + threadIdx.y * MICRO_M + i;  // global row
        if (gm >= (int)M_size) continue;
        float denom = row_l[threadIdx.y * MICRO_M + i];
        #pragma unroll
        for (int j = 0; j < MICRO_N; ++j) {
            int dv_col = threadIdx.x * MICRO_N + j;
            int gdv    = dv_start + dv_col;
            if (gdv < (int)DV_size) {
                size_t oidx = (size_t)gm * ldo + gdv + O_offset;
                O[oidx] = (scalar_t)(O_reg[i][j] / denom);
            }
        }
    }
}

void print_shape(torch::Tensor ten) {
    for(int i = 0; i < 3; i++) 
        std::cout << ten.size(i) << " ";
    std::cout << std::endl;
}

torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V, float scale) {
    std::cout << "invoke flash attention\n";
    TORCH_CHECK(Q.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(Q.dtype() == torch::kFloat32, "inputs must be fp32");
    TORCH_CHECK(K.is_cuda(), "inputs must be on cuda");
    TORCH_CHECK(K.dtype() == torch::kFloat32, "inputs must be fp32");

    const int64_t batch_size = Q.size(0);
    const int64_t m_size = Q.size(1);
    const int64_t k_size = Q.size(2);
    const int64_t n_size = K.size(1); // transpose fusion change
    const int64_t dv_size = V.size(2);

    auto opts = Q.options();
    torch::Tensor out = torch::empty({batch_size, m_size, dv_size}, opts);

    dim3 block_size(BX, BY, 1);
    dim3 grid_size((dv_size - 1) / TILE_DV + 1, 
                   (m_size - 1) / TILE_M + 1, // ????????????????///
                   batch_size);

    #ifdef TIME_FLOPS
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.stream());
    #endif

    flash_attention_kernel<float><<<grid_size, block_size>>>(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        m_size,
        n_size,
        k_size, 
        dv_size,
        scale
    );

    #ifdef TIME_FLOPS
    cudaEventRecord(stop, stream.stream());
    cudaEventSynchronize(stop);
    float elapsed_time_ms;
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    double sec_per_call = elapsed_time_ms/1e3;
    double flops_qk = 2.0 * static_cast<double>(batch_size) * static_cast<double>(m_size) * static_cast<double>(k_size) * static_cast<double>(n_size);
    double flops_pv = 2.0 * static_cast<double>(batch_size) * static_cast<double>(m_size) * static_cast<double>(n_size) * static_cast<double>(dv_size);
    double flops_per_call = flops_qk + flops_pv;
    std::cout << "flash attention sizes: [" << m_size << "x" << k_size << "] * [" << k_size << "x" << n_size << "]" << std::endl;
    print_shape(Q);
    print_shape(K);
    print_shape(V);
    std::cout << "flash attention TFLOP/s: " << flops_per_call / (sec_per_call*1e12) << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
    
    return out;
}