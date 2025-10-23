import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple

########################################################################################################

# MHA model and it's config
@dataclass
class MiniGPTConfig:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    block_size: int = 128
    dropout: float = 0.1

    aux_loss_weight: float = 0.01


class MHA(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # causal mask prepared once for maximum block size
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # shape into heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y

class DenseFFN(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MHA(config)
        self.ff = DenseFFN(config.n_embd, config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x, x.new_zeros(())  # zero aux

class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.fc = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        aux_total = x.new_zeros(())
        for block in self.blocks:
            x, aux = block(x)
            aux_total = aux_total + aux

        x = self.ln_f(x)
        logits = self.fc(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, aux_total

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

########################################################################################################

@dataclass
class MoEGPTConfig(MiniGPTConfig):
    vocab_size: int
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    block_size: int = 128
    dropout: float = 0.1

    aux_loss_weight: float = 0.01

    num_experts: int = 8  # Number of experts
    expert_dim: int = 256  # Hidden dimension of experts


class Router(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.num_experts)  # Output size = num_experts

    def forward(self, x):
        # (B, T, C) -> (B, T, num_experts)
        logits = self.fc(x)  # Raw logits to determine routing for each token
        probs = F.softmax(logits, dim=-1)  # (B, T, num_experts)
        return probs

class MoEExpert(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.expert_fc = nn.Sequential(
            nn.Linear(config.n_embd, config.expert_dim),
            nn.GELU(),
            nn.Linear(config.expert_dim, config.n_embd),
        )

    def forward(self, x):
        return self.expert_fc(x)

class MoELayer(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.router = Router(config)
        self.experts = nn.ModuleList([MoEExpert(config) for _ in range(config.num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        expert_probs = self.router(x)  # (B, T, num_experts)
        expert_outputs = []

        # Use softmax probabilities to apply weighted expert outputs
        for i, expert in enumerate(self.experts):
            expert_output = expert(x)  # (B, T, C)
            expert_outputs.append(expert_output * expert_probs[:, :, i:i+1])  # Weighted by probability

        # Aggregate the expert outputs (sum the weighted outputs)
        moe_output = torch.stack(expert_outputs, dim=-1).sum(dim=-1)  # (B, T, C)
        return moe_output

class MoEGPT(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.mha_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer - 1)])
        self.moe_layer = MoELayer(config)
        self.ln_f = nn.LayerNorm(config.n_embd)  # Final layer norm
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        aux_total = x.new_zeros(())
        
        for mha_block in self.mha_blocks:
            x, aux = mha_block(x)
            aux_total = aux_total + aux
        
        x = self.moe_layer(x)  # Apply MoE layer
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss, aux_total

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _, _ = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

########################################################################################################

# --- Main Training and Inference Script ---
def load_text(path: str) -> str:
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print("No dataset found")
        exit(1)


def get_batch(data, batch_size, block_size):
    # Non-naive batching: efficient handling
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i + block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i + 1:i + 1 + block_size]) for i in ix])
    return x, y


def get_batch(data_ids, block_size, batch_size, device):
    # sample random offsets
    ix = torch.randint(0, data_ids.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i:i+block_size] for i in ix])
    y = torch.stack([data_ids[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)


def print_model_info(model):
    # Print model parameter count and estimated size
    param_count = sum(p.numel() for p in model.parameters())
    estimated_size = param_count * 4 / (1024 ** 2)  # size in MB assuming float32 (4 bytes)
    print(f"Model has {param_count:,} parameters.")
    print(f"Estimated model size: {estimated_size:.2f} MB")


def train(model, data_ids, batch_size=64, block_size=128, epochs=3, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        xb, yb = get_batch(data_ids, block_size, batch_size, 'cuda')
        logits, loss_ce, aux_loss = model(xb, yb)
        loss = loss_ce + cfg.aux_loss_weight * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            dt = time.time() - t0
            print(f"epoch {epoch:5d}/{epochs} | loss {loss.item():.4f} (ce {loss_ce.item():.4f} + aux {aux_loss.item():.4f}) | {dt:.1f}s")

    training_time = time.time() - t0
    print(f"Training time: {training_time:.2f} seconds")


class CharTokenizer:
    def __init__(self, text: str):
        vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


def inference(model, tok, max_new_tokens=100):
    model.eval()

    # prepare empty context for now
    ctx = torch.zeros((1, 1), dtype=torch.long).to('cuda')

    # infer
    out = model.generate(ctx, max_new_tokens)[0].tolist()
    generated = tok.decode(out)

    return generated


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["miniGPT", "MoE"], required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    args = parser.parse_args()

    # Load data
    text = load_text("./shakespeare.txt")

    # tokenize data
    tok = CharTokenizer(text)
    data_ids = torch.tensor(tok.encode(text), dtype=torch.long)

    # Load model
    if args.model == "miniGPT":
        print("miniGPT")
        cfg = MiniGPTConfig(vocab_size=tok.vocab_size)
        model = MiniGPT(cfg).cuda()
    elif args.model == "MoE":
        print("using MoE model")
        cfg = MoEGPTConfig(vocab_size=tok.vocab_size)
        model = MoEGPT(cfg).cuda()

    # Print model info
    print_model_info(model)

    # Train the model
    train(model, data_ids, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

    # Generate text with the trained model
    output = inference(model, tok, max_new_tokens=args.max_new_tokens)
    print("Generated text:", output)

