#include <torch/extension.h>

torch::Tensor custom_transpose_forward(torch::Tensor input);
torch::Tensor custom_matmul_forward(torch::Tensor a, torch::Tensor b);
torch::Tensor custom_softmax_forward(torch::Tensor input);
torch::Tensor custom_eltwise_div_forward(torch::Tensor input, float val);
torch::Tensor qkt_cublas_forward(torch::Tensor Q, torch::Tensor K, float scale);
torch::Tensor pv_cublas_forward(torch::Tensor Probs, torch::Tensor V);
torch::Tensor opt_softmax_forward(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_transpose_forward", &custom_transpose_forward, "Custom transpose CUDA impl");
    m.def("custom_matmul_forward", &custom_matmul_forward, "Custom matmul CUDA impl");
    m.def("custom_softmax_forward", &custom_softmax_forward, "Custom softmax CUDA impl");
    m.def("custom_eltwise_div_forward", &custom_eltwise_div_forward, "Custom eltwise division CUDA impl");
    m.def("qkt_cublas_forward", &qkt_cublas_forward, "Fused cublas Q*K^T impl");
    m.def("pv_cublas_forward", &pv_cublas_forward, "PV cublas impl");
    m.def("opt_softmax_forward", &opt_softmax_forward, "Optimized softmax CUDA impl");
}
