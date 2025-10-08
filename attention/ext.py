# Custom CUDA extension module
import torch.utils.cpp_extension as ext


# Load custom transpose extension
_ext = ext.load(
    name="naive_attention_ext",  # Must be unique
    sources=["naive_attention.cpp", "naive_attention_kernels.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

def extensions():
    return _ext