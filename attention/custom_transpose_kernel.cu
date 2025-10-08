#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
__global__ void custom_transpose_forward_kernel(
    const scalar_t* __restrict__ input,
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
                   (batch_size - 1) / block_size.z + 1);

    if (input.scalar_type() == torch::kFloat32) {
        custom_transpose_forward_kernel<float><<<grid_size, block_size>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            seq_size,
            embed_size
        );
    } else {
        std::cout << "unsupported type in cutom relu" << std::endl;
    }
    
    return output;
}