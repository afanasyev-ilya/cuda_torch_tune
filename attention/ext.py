# Custom CUDA extension module
import torch.utils.cpp_extension as ext


# Load custom transpose extension
_ext = ext.load(
    name="custom_attention_ext",  # Must be unique
    sources=["custom_attention.cpp", "custom_attention_kernels.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

def extensions():
    return _ext