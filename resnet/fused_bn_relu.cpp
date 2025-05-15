#include <torch/extension.h>

torch::Tensor fused_bn_relu_forward(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor mean,
    torch::Tensor var,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_bn_relu_forward", &fused_bn_relu_forward, "Fused BN+ReLU forward");
}
