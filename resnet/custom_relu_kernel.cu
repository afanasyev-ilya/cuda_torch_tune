#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

template <typename scalar_t>
__global__ void custom_relu_forward_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t num_elements) {
    
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}

torch::Tensor custom_relu_forward(torch::Tensor input) {
    torch::Tensor output = torch::empty_like(input);
    
    const int64_t num_elements = input.numel();
    const int threads = 1024;
    const int blocks = (num_elements + threads - 1) / threads;

    if (input.scalar_type() == torch::kFloat32) {
        custom_relu_forward_kernel<float><<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            num_elements
        );
    } else if (input.scalar_type() == torch::kFloat64) {
        custom_relu_forward_kernel<double><<<blocks, threads>>>(
            input.data_ptr<double>(),
            output.data_ptr<double>(),
            num_elements
        );
    } else {
        std::cout << "unsupported type in cutom relu" << std::endl;
    }
    
    return output;
}