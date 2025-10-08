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



@pytest.mark.parametrize("B,M,N,K", [(1, 5, 7, 8)])
def test_matmul_matches_pytorch(B, M, N, K):
    class CustomMatmul(Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, a, b):
            return extensions().custom_matmul_forward(a, b)

    a = torch.randn(2, 4, 5).cuda()
    b = torch.randn(2, 5, 6).cuda()

    mm = CustomMatmul().cuda().eval()
    assert torch.allclose(mm(a, b), torch.matmul(a, b))