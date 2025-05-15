#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void custom_batchnorm_forward_kernel(const scalar_t* __restrict__ x,
                                                scalar_t* __restrict__ y,
                                                const scalar_t* __restrict__ gamma,
                                                const scalar_t* __restrict__ beta,
                                                const scalar_t* __restrict__ mean,
                                                const scalar_t* __restrict__ var,
                                                const float eps,
                                                const int n,
                                                const int c,
                                                const int h,
                                                const int w,
                                                const int num_elements) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) 
        return;

    // Calculate channel index (NCHW format)
    // tid / (h*w) = batch-channel index
    // %c - we extract channel index
    const int channel_idx = (idx / (h*w)) % c;

    // BatchNorm formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    y[idx] = gamma[channel_idx] * (x[idx] - mean[channel_idx]) * rsqrt(var[channel_idx] + eps) + beta[channel_idx];
}

torch::Tensor custom_batchnorm_forward(torch::Tensor input,
                                       torch::Tensor gamma,
                                       torch::Tensor beta,
                                       torch::Tensor mean,
                                       torch::Tensor var,
                                       double eps) {
    
    auto output = torch::empty_like(input);
    const int num_elements = input.numel();
    
    // Get tensor dimensions (NCHW format)
    const int n = input.size(0);
    const int c = input.size(1);
    const int h = input.size(2);
    const int w = input.size(3);

    // Kernel configuration
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "custom_batchnorm_forward", ([&] {
        custom_batchnorm_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            gamma.data_ptr<scalar_t>(),
            beta.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            static_cast<float>(eps),
            n,
            c,
            h,
            w,
            num_elements
        );
    }));

    return output;
}