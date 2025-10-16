import torch
import torch.nn as nn
import time
from torch.nn import Module
from ext import extensions
import math


DEBUG_MODE = False

if DEBUG_MODE:
    batch_size = 1
    seq_len = 5
    embed_dim = 4
    num_heads = 1
else:
    batch_size = 8
    seq_len = 4096 # as in llama2
    embed_dim = 128 # as in llama2
    num_heads = 1


def benchmark(name, model, *inputs):
    warmup_iters = 1
    for iter in range(0, warmup_iters):
        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(*inputs)

    avg_time = 0.0
    min_time = 0.0
    max_time = 0.0
    benchmark_iters = 3
    for iter in range(0, benchmark_iters):
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            outputs = model(*inputs)
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

    #print(f"Inference ({name}) min time: {min_time:.2f} ms")
    print(f"Inference ({name}) avg time: {avg_time:.2f} ms")
    #print(f"Inference ({name}) max time: {max_time:.2f} ms")
    print("\n\n")
    return outputs


def torch_attention(Q, K, V):
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True, dropout=0.0).cuda().eval()

    with torch.no_grad():
        eye = torch.eye(embed_dim)

        mha.in_proj_weight.zero_()
        mha.in_proj_bias.zero_()

        mha.in_proj_weight[:embed_dim, :] = eye
        mha.in_proj_weight[embed_dim:2*embed_dim, :] = eye
        mha.in_proj_weight[2*embed_dim:, :] = eye
        mha.out_proj.weight.copy_(eye)
        mha.out_proj.bias.zero_()
    
    # Perform the forward pass
    attn_output, attn_output_weights = benchmark("torch mha", mha, Q, K, V)

    return attn_output


def layerwise_torch_attention(Q, K, V):
    class LayerwiseSDPA(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, query, key, value):
            # 1. Calculate dot product of query and key
            # (batch_size, query_seq_len, embed_dim) @ (batch_size, embed_dim, key_seq_len)
            # -> (batch_size, query_seq_len, key_seq_len)
            scores = torch.matmul(query, torch.transpose(key, -2, -1))

            # 2. scaling
            dk = torch.tensor(self.embed_dim, dtype=torch.float32).cuda()
            sqrt_dk = torch.sqrt(dk)

            scores = scores / sqrt_dk

            # 3. Apply softmax to get attention weights
            attention_weights = torch.nn.functional.softmax(scores, dim=-1)

            # 4. *= K
            attention_output = torch.matmul(attention_weights, value);

            return attention_output, attention_weights
    
    mha = LayerwiseSDPA(embed_dim).cuda().eval()

    attn_output, attn_output_weights = benchmark("layerwise attention", mha, Q, K, V)

    return attn_output


def cuda_naive_attention(Q, K, V):
    class CudaNaiveSDPA(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        # since we write the model on our own, can jsut create such functions instead of search and replace
        def cuda_transpose(self, input):
            return extensions().custom_transpose_forward(input)
        
        def cuda_mamtul(self, a, b):
            return extensions().custom_matmul_forward(a, b)

        def cuda_softmax(self, input):
            return extensions().custom_softmax_forward(input)

        def cuda_eltwise_div(self, input, val):
            return extensions().custom_eltwise_div_forward(input, val)

        def forward(self, query, key, value):
            # 1. Calculate dot product of query and key^T
            scores = self.cuda_mamtul(query, self.cuda_transpose(key))

            # 2. Scaling scores by embed_size sqrt (scalar)
            dk_sqrt = math.sqrt(self.embed_dim)
            scores = self.cuda_eltwise_div(scores, dk_sqrt)

            # 3. compute attention weights
            attention_weights = self.cuda_softmax(scores)

            # 4. Multiply by K
            attention_output = self.cuda_mamtul(attention_weights, value);

            return attention_output
    
    sdpa = CudaNaiveSDPA(embed_dim).cuda().eval()

    attn_output = benchmark("cuda naive", sdpa, Q, K, V)

    return attn_output



def cuda_opt_layerwise_attention(Q, K, V):
    class CudaOptLayerwiseSDPA(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def cuda_softmax(self, input):
            return extensions().opt_softmax_forward(input)

        def cublas_qkt(self, query_input, key_input, scale_input):
            return extensions().qkt_cublas_forward(query_input, key_input, scale_input)

        def cublas_pv(self, weights, value):
            return extensions().pv_cublas_forward(weights, value)

        def forward(self, query, key, value):
            # 1. use cublas to transpose and multiply
            scores = self.cublas_qkt(query, key, 1.0/math.sqrt(self.embed_dim))

            # 2. compute attention weights
            self.cuda_softmax(scores)

            # 3. Multiply by V
            attention_output = self.cublas_pv(scores, value);

            return attention_output
    
    sdpa = CudaOptLayerwiseSDPA(embed_dim).cuda().eval()

    attn_output = benchmark("cuda opt layerwise", sdpa, Q, K, V)

    return attn_output


def flash_attention(Q, K, V):
    class FlashAttention(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def cuda_flash_attention(self, query_input, key_input, value, scale_input):
            return extensions().flash_attention_forward(query_input, key_input, value, scale_input)

        def forward(self, query, key, value):
            attention_output = self.cuda_flash_attention(query, key, value, 1.0/math.sqrt(self.embed_dim))
            return attention_output
    
    attn = FlashAttention(embed_dim).cuda().eval()

    attn_output = benchmark("flash attention", attn, Q, K, V)

    return attn_output


def check(name, outs, ref):
    all_close = torch.allclose(outs, ref, rtol=1e-5, atol=1e-5)
    print(f"{name} test all close? {all_close}")
    if not all_close:
        max_diff = torch.max(torch.abs(outs - ref))
        print(f"{name} maximum difference between the tensors is: {max_diff}")


def run_all():
    Q = torch.randn(batch_size, seq_len, embed_dim).cuda()
    K = torch.randn(batch_size, seq_len, embed_dim).cuda()
    V = torch.randn(batch_size, seq_len, embed_dim).cuda()

    print(f"Q/K/V shape: {Q.shape}")

    torch_res = torch_attention(Q, K, V)
    custom_res = layerwise_torch_attention(Q, K, V)
    cuda_naive_res = cuda_naive_attention(Q, K, V)
    cuda_opt_layerwise_res = cuda_opt_layerwise_attention(Q, K, V)
    flash_attention_res = flash_attention(Q, K, V)

    check("custom torch", torch_res, custom_res)
    check("cuda naive", torch_res, cuda_naive_res)
    check("cuda layerwise opt", torch_res, cuda_opt_layerwise_res)
    check("flash attention", torch_res, flash_attention_res)

    if DEBUG_MODE:
        print("build-in torch (ref): ------------ ")
        print(torch_res)
        print("custom torch (ref): ------------ ")
        print(custom_res)
        print("CUDA naive: ------------ ")
        print(cuda_naive_res)
        print("CUDA opt: ------------ ")
        print(cuda_opt_layerwise_res)


def run():
    Q = torch.randn(batch_size, seq_len, embed_dim).cuda()
    K = torch.randn(batch_size, seq_len, embed_dim).cuda()
    V = torch.randn(batch_size, seq_len, embed_dim).cuda()

    torch_res = torch_attention(Q, K, V)

    flash_attention_res = flash_attention(Q, K, V)
    check("flash attention", torch_res, flash_attention_res)


if __name__ == "__main__":
    run()