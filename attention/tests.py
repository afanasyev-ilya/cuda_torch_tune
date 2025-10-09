import torch
import pytest
from torch.nn import Module
from ext import extensions


@pytest.mark.parametrize("B,M,N", [(1, 5, 7), (2, 6, 8)])
def test_transpose_matches_pytorch(B, M, N):
    class CustomTranspose(Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input):
            return extensions().custom_transpose_forward(input)

    input = torch.randn(B, M, N).cuda()
    trans = CustomTranspose().cuda().eval()
    out = trans(input)

    ref = torch.transpose(input, -2, -1)
    assert torch.allclose(out, ref)



@pytest.mark.parametrize("B,M,N,K", [(1, 5, 7, 8), (2, 5, 5, 5), (1, 5, 10, 5)])
def test_matmul_matches_pytorch(B, M, N, K):
    class CustomMatmul(Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, a, b):
            return extensions().custom_matmul_forward(a, b)

    a = torch.randn(B, M, K).cuda()
    b = torch.randn(B, K, N).cuda()

    mm = CustomMatmul().cuda().eval()
    out = mm(a, b)
    ref = torch.matmul(a, b)
    assert torch.allclose(out, ref)


@pytest.mark.parametrize("B,M,N", [(1, 5, 15), (1, 6, 256), (1, 5, 1024), (1, 5, 32)])
def test_softmax_matches_pytorch(B, M, N):
    class CustomSoftmax(Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input):
            return extensions().custom_softmax_forward(input)

    input = torch.randn(2, 4, 5).cuda()

    custom_softmax = CustomSoftmax().cuda().eval()
    assert(torch.allclose(custom_softmax(input), torch.softmax(input, dim=-1)))

@pytest.mark.parametrize("B,M,N", [(1, 5, 15), (1, 6, 256), (1, 5, 1024), (1, 5, 32)])
def test_opt_softmax_matches_pytorch(B, M, N):
    class OptSoftmax(Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input):
            return extensions().opt_softmax_forward(input)

    input = torch.randn(2, 4, 5).cuda()

    opt_softmax = OptSoftmax().cuda().eval()
    assert(torch.allclose(opt_softmax(input), torch.softmax(input, dim=-1)))


@pytest.mark.parametrize("B,M,N,Val", [(1, 5, 5, 5.0)])
def test_eltwise_matches_pytorch(B, M, N, Val):
    class CustomEltwise(Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, input, val):
            return extensions().custom_eltwise_div_forward(input, val)

    input = torch.randn(2, 4, 5).cuda()

    elt = CustomEltwise().cuda().eval()
    out = elt(input, Val)
    ref = input / torch.tensor(Val, dtype=torch.float32)

    assert(torch.allclose(out, ref))


# write code to see prints here
#test_eltwise_matches_pytorch(1, 5, 5, 5.0)