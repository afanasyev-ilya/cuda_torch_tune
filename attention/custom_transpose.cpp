#include <torch/extension.h>

torch::Tensor custom_transpose_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_transpose_forward", &custom_transpose_forward, "Custom transpose forward");
}
