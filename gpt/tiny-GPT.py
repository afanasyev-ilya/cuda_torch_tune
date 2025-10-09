
# tiny_gpt2_min.py
import math, os, torch, torch.nn as nn
from torch.nn import functional as F

# ----------------------------
# Data: tiny, character-level
# ----------------------------
# Put a plain text file at ./data/input.txt (e.g., "tiny_shakespeare.txt")
with open("./data/tiny_shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi  = {ch:i for i,ch in enumerate(chars)}
itos  = {i:ch for ch,i in stoi.items()}
vocab_size = len(chars)

def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
def decode(t): return "".join([itos[int(i)] for i in t])

data = encode(text)
n = int(0.9*len(data))
train_data, val_data = data[:n], data[n:]

# batching helper
def get_batch(split, batch_size, block_size, device):
    src = train_data if split=="train" else val_data
    ix = torch.randint(len(src)-block_size-1, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# ----------------------------
# Your attention block (causal)
# ----------------------------
class LayerwiseSDPA(nn.Module):
    """Your block with a causal mask (decoder-style) and scaling."""
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = 1.0 / math.sqrt(embed_dim)

    def forward(self, query, key, value, *, causal: bool = True):
        # query:[B, Sq, D], key/value:[B, Sk, D]; here Sq==Sk=S for self-attn
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale  # [B,Sq,Sk]

        if causal:
            Sq, Sk = query.size(-2), key.size(-2)
            mask = torch.triu(torch.ones(Sq, Sk, dtype=torch.bool, device=scores.device), diagonal=1)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, value)  # [B,Sq,D]
        return out

# ----------------------------
# Multi-head wrapper around your SDPA
# ----------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.d_head = n_embd // n_head

        # Q, K, V projections (why: see explanation above)
        self.q_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn = LayerwiseSDPA(self.d_head)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, S, E = x.shape
        H, Dh = self.n_head, self.d_head

        q = self.q_proj(x).view(B, S, H, Dh).transpose(1, 2).contiguous().view(B*H, S, Dh)
        k = self.k_proj(x).view(B, S, H, Dh).transpose(1, 2).contiguous().view(B*H, S, Dh)
        v = self.v_proj(x).view(B, S, H, Dh).transpose(1, 2).contiguous().view(B*H, S, Dh)

        y = self.attn(q, k, v, causal=True)                         # [B*H,S,Dh]
        y = y.view(B, H, S, Dh).transpose(1, 2).contiguous().view(B, S, E)  # merge heads
        y = self.drop(self.out_proj(y))
        return y

# ----------------------------
# GPT-2 style block (pre-norm)
# ----------------------------
class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.0, expansion=4):
        super().__init__()
        self.fc = nn.Linear(n_embd, expansion*n_embd)
        self.proj = nn.Linear(expansion*n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc(x)
        x = F.gelu(x)
        x = self.drop(self.proj(x))
        return x

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# ----------------------------
# Tiny GPT-2
# ----------------------------
class TinyGPT2(nn.Module):
    def __init__(self, vocab_size, n_layer=4, n_head=4, n_embd=256, block_size=256, dropout=0.0):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb   = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying (classic GPT-2 trick)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx, targets=None):
        B, S = idx.shape
        assert S <= self.block_size, "Sequence length > block_size"

        tok = self.token_emb(idx)                            # [B,S,E]
        pos = self.pos_emb(torch.arange(S, device=idx.device))[None, :, :]  # [1,S,E]
        x = self.drop(tok + pos)

        for blk in self.blocks:
            x = blk(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)                             # [B,S,V]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]             # crop to block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                        # last step
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

# ----------------------------
# Train (tiny defaults)
# ----------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1337)

    # small-ish config to check everything runs
    block_size = 256
    n_layer = 4
    n_head  = 4
    n_embd  = 256
    dropout = 0.1
    batch_size = 64
    max_steps = 2000
    lr = 3e-4

    model = TinyGPT2(vocab_size, n_layer, n_head, n_embd, block_size, dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()

    for step in range(1, max_steps+1):
        x, y = get_batch("train", batch_size, block_size, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            with torch.no_grad():
                vx, vy = get_batch("val", batch_size, block_size, device)
                _, vloss = model(vx, vy)
            print(f"step {step:5d} | train loss {loss.item():.3f} | val loss {vloss.item():.3f}")

    # quick sample
    prompt = "Once upon a time"
    ctx = encode(prompt).unsqueeze(0).to(device)
    out = model.generate(ctx, max_new_tokens=200)[0].tolist()
    print(decode(out))

if __name__ == "__main__":
    os.makedirs("./data", exist_ok=True)
    # if you don't have a file yet, create a super tiny fallback
    if not os.path.exists("./data/tiny_shakespeare.txt"):
        with open("./data/tiny_shakespeare.txt", "w", encoding="utf-8") as f:
            f.write("Tiny dataset.\n")
    main()