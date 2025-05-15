#include <torch/extension.h>

torch::Tensor custom_relu_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_relu_forward", &custom_relu_forward, "Custom ReLU forward");
}
