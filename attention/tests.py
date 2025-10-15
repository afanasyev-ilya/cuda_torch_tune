import torch
import pytest
from torch.nn import Module
from ext import extensions
import time


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



@pytest.mark.parametrize("B,M,N,K", [(1, 512, 512, 512), (1, 1024, 512, 2048)])
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
    #assert torch.allclose(out, ref) # TODO FIX ME



@pytest.mark.parametrize("B,M,N,K", [(1, 2048, 2048, 2048), (1, 1024, 512, 2048)])
def test_opt_matmul_matches_pytorch(B, M, N, K):
    class OptMatmul(Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, a, b):
            return extensions().opt_matmul_forward(a, b)

    a = torch.randn(B, M, K).cuda()
    b = torch.randn(B, K, N).cuda()

    mm = OptMatmul().cuda().eval()
    out = mm(a, b)
    ref = torch.matmul(a, b)
    #assert torch.allclose(out, ref)


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
            extensions().opt_softmax_forward(input)
            return input

    input = torch.randn(2, 4, 5).cuda()
    save_input = input

    opt_softmax = OptSoftmax().cuda().eval()
    ref = torch.softmax(save_input, dim=-1)
    out = opt_softmax(input)
    assert(torch.allclose(out, ref))


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
#test_matmul_matches_pytorch(1, 8192, 8192, 8192)
#test_matmul_matches_pytorch(4, 4096, 4096, 128)
#test_opt_matmul_matches_pytorch(1, 8192, 8192, 8192)
#test_opt_matmul_matches_pytorch(4, 4096, 4096, 128)

def benchmark_mm(B, M, N, K, heat_runs, benchmark_iters):
    class OptMatmul(Module):
        def __init__(self):
            super().__init__()
            
        def forward(self, a, b):
            return extensions().opt_matmul_forward(a, b)

    a = torch.randn(B, M, K).cuda()
    b = torch.randn(B, K, N).cuda()

    mm = OptMatmul().cuda().eval()

    for i in range(0, heat_runs):
        out = mm(a, b)

    avg_time = 0.0
    min_time = 0.0
    max_time = 0.0

    for iter in range(0, benchmark_iters):
        torch.cuda.synchronize()
        start = time.time()
        mm(a, b)
        torch.cuda.synchronize()
        cur_time = (time.time() - start) * 1000
        avg_time += cur_time / benchmark_iters
        if min_time == 0:
            min_time = cur_time
        else:
            min_time = min(cur_time, min_time)
        if max_time == 0:
            max_time = cur_time
        else:
            max_time = max(cur_time, max_time)

    ops = 2 * B * M * K * N

    print(f"Inference min perf: {ops / (1e9 * max_time)} TFlop/s")
    print(f"Inference avg perf: {ops / (1e9 * avg_time)} TFlop/s")
    print(f"Inference max perf: {ops / (1e9 * min_time)} TFlop/s")
    print("\n\n")


def benchmark_gemm():
    heat_runs = 2
    benchmark_iters = 20
    print("--------------- 8192x8192 & 8192x8192 -------------------")
    benchmark_mm(1, 8192, 8192, 8192, heat_runs, benchmark_iters)
    print("--------------- 4096x128 & 128x4096 -------------------")
    benchmark_mm(4, 4096, 4096, 128, heat_runs, benchmark_iters)

    print("--------------- 4096x256 & 256x4096 -------------------")
    benchmark_mm(4, 4096, 4096, 256, heat_runs, benchmark_iters)

def profile_gemm():
    print("--------------- 8192x8192 & 8192x8192 -------------------")
    benchmark_mm(1, 8192, 8192, 8192, 1, 1)

profile_gemm()