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
    dim3 grid_size((m_size - 1) / block_size.x + 1, 
                   (n_size - 1) / block_size.y + 1,
                   batch_size);

    #ifdef TIME_FLOPS
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.stream());
    #endif

    custom_matmul_forward_kernel<float><<<grid_size, block_size, 0, stream.stream()>>>(
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

////////////////////////////////////////////// fused matmul /////////////////////////////////

template <typename scalar_t>
__global__ void opt_matmul_forward_kernel(const scalar_t* __restrict__ A,
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

    dim3 block_size(32, 32, 1);
    dim3 grid_size((m_size - 1) / block_size.x + 1, 
                   (n_size - 1) / block_size.y + 1,
                   batch_size);

    #ifdef TIME_FLOPS
    auto stream = at::cuda::getCurrentCUDAStream();
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream.stream());
    #endif

    opt_matmul_forward_kernel<float><<<grid_size, block_size, 0, stream.stream()>>>(
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
