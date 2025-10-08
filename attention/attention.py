import torch
import torch.nn as nn


batch_size = 1
seq_len = 5
embed_dim = 7
num_heads = 1


def torch_attention(Q, K, V):
    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, bias=True, dropout=0.0).eval()

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
    attn_output, attn_output_weights = mha(Q, K, V)

    # Print the shapes of the outputs
    print(f"Shape of attention output: {attn_output.shape}")
    print(f"Shape of attention weights: {attn_output_weights.shape}")

    return attn_output


def custom_torch_attention(Q, K, V):
    class CustomScaledDotProductAttention(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.embed_dim = embed_dim

        def forward(self, query, key, value):
            # 1. Calculate dot product of query and key
            # (batch_size, query_seq_len, embed_dim) @ (batch_size, embed_dim, key_seq_len)
            # -> (batch_size, query_seq_len, key_seq_len)
            scores = torch.matmul(query, torch.transpose(key, -2, -1))

            # 2. scaling
            dk = torch.tensor(self.embed_dim, dtype=torch.float32)
            sqrt_dk = torch.sqrt(dk)

            scores = scores / sqrt_dk

            # 3. Apply softmax to get attention weights
            attention_weights = torch.nn.functional.softmax(scores, dim=-1)

            # 4. *= K
            attention_output = torch.matmul(attention_weights, value);

            return attention_output, attention_weights
    
    custom_attention = CustomScaledDotProductAttention(embed_dim)

    attn_output, attn_output_weights = custom_attention(Q, K, V)

    print(f"Shape of attention output: {attn_output.shape}")
    print(f"Shape of attention weights: {attn_output_weights.shape}")

    return attn_output


if __name__ == "__main__":
    Q = torch.randn(batch_size, seq_len, embed_dim)
    K = torch.randn(batch_size, seq_len, embed_dim)
    V = torch.randn(batch_size, seq_len, embed_dim)

    print(f"Q/K/V shape: {Q.shape}")

    torch_res = torch_attention(Q, K, V)
    custom_res = custom_torch_attention(Q, K, V)

    are_all_close = torch.allclose(torch_res, custom_res)
    print(f"Are all elements close (default tolerances)? {are_all_close}")

    print(torch_res)
    print(custom_res)