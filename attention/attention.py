import torch
import torch.nn as nn
import time

DEBUG_MODE = False

if DEBUG_MODE:
    batch_size = 1
    seq_len = 3
    embed_dim = 5
    num_heads = 1
else:
    batch_size = 128
    seq_len = 100
    embed_dim = 4096
    num_heads = 1


def benchmark(name, model, *inputs):
    warmup_iters = 5
    for iter in range(0, warmup_iters):
        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(*inputs)

    avg_time = 0.0
    min_time = 0.0
    max_time = 0.0
    benchmark_iters = 10
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

    print(f"Inference ({name}) min time: {min_time:.2f} ms")
    print(f"Inference ({name}) avg time: {avg_time:.2f} ms")
    print(f"Inference ({name}) max time: {max_time:.2f} ms")
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

    # Print the shapes of the outputs
    print(f"Shape of attention output: {attn_output.shape}")
    print(f"Shape of attention weights: {attn_output_weights.shape}")

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

    print(f"Shape of attention output: {attn_output.shape}")
    print(f"Shape of attention weights: {attn_output_weights.shape}")

    return attn_output


if __name__ == "__main__":
    Q = torch.randn(batch_size, seq_len, embed_dim).cuda()
    K = torch.randn(batch_size, seq_len, embed_dim).cuda()
    V = torch.randn(batch_size, seq_len, embed_dim).cuda()

    print(f"Q/K/V shape: {Q.shape}")

    torch_res = torch_attention(Q, K, V)
    custom_res = layerwise_torch_attention(Q, K, V)

    are_all_close = torch.allclose(torch_res, custom_res)
    print(f"Are all elements close (default tolerances)? {are_all_close}")

    if DEBUG_MODE:
        print(torch_res)
        print(custom_res)