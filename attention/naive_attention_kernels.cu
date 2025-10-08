#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

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
    const int64_t batch_size = input.size(0);
    const int64_t seq_size = input.size(1);
    const int64_t embed_size = input.size(2);

    torch::Tensor output = torch::empty({batch_size, embed_size, seq_size}, torch::kFloat32).cuda();

    dim3 block_size(32, 32, 1);
    dim3 grid_size((seq_size - 1) / block_size.x + 1, 
                   (embed_size - 1) / block_size.y + 1,
                   batch_size);

    if (input.scalar_type() == torch::kFloat32) {
        custom_transpose_forward_kernel<float><<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            seq_size,
            embed_size
        );
    } else {
        std::cout << "unsupported element type in custom transpose, should be float 32" << std::endl;
    }
    
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
    const int64_t batch_size = a.size(0);
    const int64_t m_size = a.size(1);
    const int64_t k_size = a.size(2);
    const int64_t n_size = b.size(2);

    torch::Tensor c = torch::empty({batch_size, m_size, n_size}, torch::kFloat32).cuda();

    dim3 block_size(32, 32, 1);
    dim3 grid_size((m_size - 1) / block_size.x + 1, 
                   (n_size - 1) / block_size.y + 1,
                   batch_size);

    if (a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32) {
        custom_matmul_forward_kernel<float><<<grid_size, block_size>>>(
            a.data_ptr<float>(),
            b.data_ptr<float>(),
            c.data_ptr<float>(),
            batch_size,
            m_size,
            n_size,
            k_size
        );
    } else {
        std::cout << "unsupported element type in custom matmul, should be float 32" << std::endl;
    }
    
    return c;
}