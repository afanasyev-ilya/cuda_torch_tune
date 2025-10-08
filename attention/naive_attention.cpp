#include <torch/extension.h>

torch::Tensor custom_transpose_forward(torch::Tensor input);
torch::Tensor custom_matmul_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor custom_softmax_forward(torch::Tensor a);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_transpose_forward", &custom_transpose_forward, "Custom transpose CUDA impl");
    m.def("custom_matmul_forward", &custom_matmul_forward, "Custom matmul CUDA impl");
    m.def("custom_softmax_forward", &custom_softmax_forward, "Custom softmax CUDA impl");
}
