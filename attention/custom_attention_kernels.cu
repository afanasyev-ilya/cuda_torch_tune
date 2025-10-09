#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>


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

    torch::Tensor output = torch::empty({batch_size, embed_size, seq_size}, torch::kFloat32).cuda();

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

    custom_matmul_forward_kernel<float><<<grid_size, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        batch_size,
        m_size,
        n_size,
        k_size
    );
    
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

// ------------------------------------ op -------------------------------------------------------

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
    return scores;
}
