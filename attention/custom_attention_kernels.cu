#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

#define TIME_FLOPS

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
    std::cout << "NAIVE matmul perf: " << flops_per_call / (sec_per_call*1e12) << " TFLOP/s" << std::endl;
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
    std::cout << "qkt perf: " << flops_per_call / (sec_per_call*1e12)<< " TFLOP/s" << std::endl;
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
    std::cout << "   pv perf: " << flops_per_call / (sec_per_call*1e12) << " TFLOP/s" << std::endl;
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
    std::cout << "OPT matmul perf: " << flops_per_call / (sec_per_call*1e12) << " TFLOP/s" << std::endl;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    #endif
    
    return c;
}

////////////////////////////////////////////// fused with online softmax and transpose /////////////////////////////////

// -------------------- TUNABLE LIMITS / TILES --------------------
// Br: # query rows per CTA (tile along M = q_seq)
// Bc: # key/value rows per tile (tile along N = kv_seq)
// D_MAX: compile-time upper bound for head dim (D)
// DV_CHUNK: process V/output in small column chunks (register control)

#define BR 32

#define BC 48

#define D_MAX 128

#define DV_CHUNK 32


// Shared memory footprint (floats):
//   Qs: BR*(D_MAX+1)
//   Ks: BC*(D_MAX+1)
//   Vs: BC*(DV_CHUNK+1)
// Example with BR=32, BC=48, D_MAX=128, DV_CHUNK=32:
//   32*129 + 48*129 + 48*33 = 11,904 floats ≈ 46.5 KB

// -------------------- DIMENSION SEMANTICS --------------------
//
// Q: [B, M, D]    (batch, q_seq_len,  head_dim_kq)
// K: [B, N, D]    (batch, kv_seq_len, head_dim_kq)
// V: [B, N, DV]   (batch, kv_seq_len, head_dim_v)
// O: [B, M, DV]   (batch, q_seq_len,  head_dim_v)
//
// Grid: (x=1, y=ceil(M/BR), z=B)  -> one CTA owns BR query rows
// Block: (x=BR, y=1, z=1)         -> 1 thread per query row inside CTA
//
// Online softmax per row i (m,l in registers):
//   s_j   = <q_i, k_j> * scale  for j in current K/V tile
//   m_new = max(m, max_j s_j)
//   l_new = l * exp(m - m_new) + sum_j exp(s_j - m_new)
//   o_new = o * (l * exp(m - m_new) / l_new)
//         + sum_j [exp(s_j - m_new) / l_new] * v_j
//
// No NxN attention is materialized; we stream over K/V tiles.

__global__ void flash_attention_kernel_fp32_static(
    const float* __restrict__ Q,  // [B, M, D]
    const float* __restrict__ K,  // [B, N, D]
    const float* __restrict__ V,  // [B, N, DV]
    float* __restrict__ O,        // [B, M, DV]
    int Bsz, int M, int N, int D, int DV_size,
    float scale)
{
    const int b  = blockIdx.z;  // batch/head idx
    const int tb = blockIdx.y;  // which BR-slab of M
    const int tx = threadIdx.x; // row-owner within the slab [0..Br)

    const int i  = tb * BR + tx;  // absolute query row

    if (tx >= BR || i >= M)
        return;

    // Bounds guard for static shared layout
    if (D > D_MAX) 
        return;  // (guard; also asserted in host wrapper)

    // Base offsets for this batch
    const int64_t Qb_offset = (int64_t)b * M * D;
    const int64_t Kb_offset = (int64_t)b * N * D;
    const int64_t Vb_offset = (int64_t)b * N * DV_size;
    const int64_t Ob_offset = (int64_t)b * M * DV_size;

    // -------------------- STATIC SHARED TILES --------------------
    // +1 padding on fastest dimension to reduce bank conflicts.
    __shared__ float Qs[BR][D_MAX + 1];         // BR x D
    __shared__ float Ks[BC][D_MAX + 1];         // BC x D
    __shared__ float Vs[BC][DV_CHUNK + 1];      // BC x DV_CHUNK

    // --------- Load group of Q FULL rows (Br x D) into shared ---------
    {
        // Each row-owner thread loads its row (D contiguous floats)
        const float* q_src = Q + Qb_offset + (int64_t)i * D;
        float* q_dst = &Qs[tx][0];
        // Only use the first D elements (D_MAX is the static bound)
        for (int d = 0; d < D; ++d)
            q_dst[d] = q_src[d];
    }
    __syncthreads();

    // Per-row running softmax stats in registers
    float m_row = -INFINITY;
    float l_row = 0.f;

    // Process output cols in small chunks to limit register pressure
    for (int dv_start = 0; dv_start < DV_size; dv_start += DV_CHUNK) {
        const int dv_lim = min(DV_CHUNK, DV_size - dv_start);

        float o_chunk[DV_CHUNK];
        #pragma unroll
        for (int t = 0; t < DV_CHUNK; ++t) 
            o_chunk[t] = 0.f;

        // --------- Loop over K/V tiles along N ---------
        for (int n0 = 0; n0 < N; n0 += BC) {
            const int n_lim = min(BC, N - n0);

            // Load K tile: [n_lim, D] -> Ks
            for (int idx = tx; idx < n_lim * D; idx += BR) {
                int r = idx / D;    // 0..n_lim-1
                int c = idx - r * D;// 0..D-1
                Ks[r][c] = K[Kb_offset + (int64_t)(n0 + r) * D + c];
            }

            // Load V tile slice: [n_lim, dv_lim] -> Vs
            for (int idx = tx; idx < n_lim * dv_lim; idx += BR) {
                int r = idx / dv_lim;     // 0..n_lim-1
                int c = idx - r * dv_lim; // 0..dv_lim-1
                Vs[r][c] = V[Vb_offset + (int64_t)(n0 + r) * DV_size + (dv_start + c)];
            }
            __syncthreads();

            // multiply Q x K^T, BR D-sized rows x BC D-sized cols
            // Compute scores s_j for this row i against this tile
            float s_local[BC];  // uses compile-time bound
            #pragma unroll
            for (int j = 0; j < n_lim; ++j) {
                const float* q_row = &Qs[tx][0];
                const float* k_row = &Ks[j][0];
                float dot = 0.f;
                // FMAs over D (optionally vectorize with float4 when D%4==0)
                for (int d = 0; d < D; ++d) 
                    dot = fmaf(q_row[d], k_row[d], dot);
                s_local[j] = dot * scale;
            }

            // Online running softmax update
            // for each BR row each thread find max in scores block (BC), traversing each row in a loop to find max
            float m_tile = -INFINITY;
            #pragma unroll
            for (int j = 0; j < n_lim; ++j) 
                m_tile = fmaxf(m_tile, s_local[j]);
            const float m_new = fmaxf(m_row, m_tile);

            float sum_exp = 0.f;
            const float carry = (l_row > 0.f) ? (l_row * expf(m_row - m_new)) : 0.f;

            float p_local[BC];
            #pragma unroll
            for (int j = 0; j < n_lim; ++j) {
                float p = expf(s_local[j] - m_new);
                p_local[j] = p;
                sum_exp += p;
            }

            const float l_new     = carry + sum_exp;
            const float inv_l_new = 1.f / l_new;
            const float alpha     = (l_row > 0.f) ? (carry * inv_l_new) : 0.f;

            // o = o*alpha + Σ_j (p_local[j]*inv_l_new) * v_j
            #pragma unroll
            for (int t = 0; t < dv_lim; ++t) 
                o_chunk[t] *= alpha;

            // multiply scores x V
            #pragma unroll
            for (int j = 0; j < n_lim; ++j) {
                const float beta = p_local[j] * inv_l_new;
                const float* v_row = &Vs[j][0];
                #pragma unroll
                for (int t = 0; t < dv_lim; ++t) {
                    o_chunk[t] = fmaf(beta, o_chunk[t], v_row[t]);
                }
            }

            // commit stats for next tile
            m_row = m_new;
            l_row = l_new;

            __syncthreads(); // protect Ks/Vs before next tile load
        } // tiles on N

        // Store output chunk for row i
        float* o_dst = O + Ob_offset + (int64_t)i * DV_size + dv_start;
        #pragma unroll
        for (int t = 0; t < dv_lim; ++t) 
            o_dst[t] = o_chunk[t];
    } // chunks over DV
}

// -------------------- TORCH WRAPPER --------------------

torch::Tensor flash_attention_forward(torch::Tensor Q,
                                      torch::Tensor K,
                                      torch::Tensor V,
                                      float scale) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Q,K,V must be CUDA");
    TORCH_CHECK(Q.dtype() == torch::kFloat32 &&
                K.dtype() == torch::kFloat32 &&
                V.dtype() == torch::kFloat32, "Only fp32 supported");
    TORCH_CHECK(Q.dim()==3 && K.dim()==3 && V.dim()==3,
                "Expected Q:[B,M,D], K:[B,N,D], V:[B,N,DV]");

    const int B  = (int)Q.size(0);
    const int M  = (int)Q.size(1);
    const int D  = (int)Q.size(2);
    const int N  = (int)K.size(1);
    TORCH_CHECK(K.size(0)==B && K.size(2)==D, "K shape mismatch with Q");
    TORCH_CHECK(V.size(0)==B && V.size(1)==N, "V shape mismatch with K");
    const int DV = (int)V.size(2);

    // Guard for static shared arrays
    TORCH_CHECK(D <= D_MAX,
        "D=", D, " exceeds D_MAX=", D_MAX, ". Recompile with larger D_MAX or lower D.");

    auto O = torch::empty({B, M, DV}, Q.options());

    dim3 block(BR, 1, 1);               // thread-per-row within CTA slab
    dim3 grid(1, (M + BR - 1) / BR, B); // y-tiles over M, z over B

#ifdef TIME_FLOPS
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t start, stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, stream.stream());
#endif

    flash_attention_kernel_fp32_static<<<grid, block>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        O.data_ptr<float>(), B, M, N, D, DV, scale
    );

#ifdef TIME_FLOPS
    cudaEventRecord(stop, stream.stream());
    cudaEventSynchronize(stop);
    float elapsed_ms = 0.f; cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    const double sec = elapsed_ms * 1e-3;
    const double flops_qk = 2.0 * (double)B * (double)M * (double)D * (double)N;
    const double flops_pv = 2.0 * (double)B * (double)M * (double)N * (double)DV;
    const double tflops   = (flops_qk + flops_pv) / (sec * 1e12);
    std::cout << "flash attention sizes: Q[" << B << "," << M << "," << D
              << "], K[" << B << "," << N << "," << D
              << "], V[" << B << "," << N << "," << DV << "]\n";
    std::cout << "flash attention perf: " << tflops << " TFLOP/s\n";
#endif

    return O;
}