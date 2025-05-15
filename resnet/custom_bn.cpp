#include <torch/extension.h>

torch::Tensor custom_batchnorm_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_batchnorm_forward", &custom_batchnorm_forward, "Custom BatchNorm forward");
}