#!/usr/bin/env python3
"""
Tiny-MoE-GPT: a single-file educational implementation of a GPT-style language model
with a Switch-style Mixture-of-Experts (MoE) feed-forward block.

Goals:
- Be short, readable, and hackable (one file, ~300 lines).
- Show exactly how MoE differs from a dense MLP in a Transformer block.
- Provide toggles to compare MoE vs. dense, top-k routing, and an auxiliary load-balance loss.

This script trains a tiny character-level language model on a small text corpus
(you can pass your own text file). It is intentionally simple and not optimized.

Usage examples:
  # Train dense tiny GPT (no MoE)
  python tiny-moe-gpt.py --data "./input.txt" --steps 2000

  # Train with MoE, 4 experts, top-1 routing
  python tiny-moe-gpt.py --data "./input.txt" --moe --experts 4 --steps 2000

  # Use a built-in tiny corpus if you don't pass --data
  python tiny-moe-gpt.py --moe --experts 4 --steps 1000

Notes:
- MoE here replaces the standard MLP inside each Transformer block.
- We implement Switch-style top-1 routing (optionally top-2) with a simple
  load-balancing auxiliary loss to encourage uniform expert utilization.
- There is no capacity limit/drop routing; this keeps the code small.
- For speed and simplicity, we loop over experts when scattering/gathering.
- This is an educational toy; don't expect SotA performance.
"""

import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 1337):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------
# Data: char-level tokenizer
# ------------------------------

BUILTIN_TEXT = (
    """
    To build a tiny model you first choose a tiny goal.
    This file demonstrates how Mixture-of-Experts (MoE) differs from a dense MLP
    inside a Transformer block. We route each token's hidden state to one of several
    experts using a learned router. Compare this to the Multi-Head Attention (MHA),
    which mixes information across tokens; MoE mixes across parameters by selecting
    different feed-forward networks.
    """
    .strip()
)

class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


def load_text(path: Optional[str]) -> str:
    if path and os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print("No dataset found")
        exit(1)


# ------------------------------
# Model config
# ------------------------------

@dataclass
class Config:
    vocab_size: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0

    # MoE knobs
    moe: bool = False
    num_experts: int = 4
    topk: int = 1  # 1 or 2 supported (top-2 returns weighted sum of two experts)
    aux_loss_weight: float = 0.01


# ------------------------------
# Attention and MoE blocks
# ------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config):
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


class ExpertMLP(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


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


class MoEFFN(nn.Module):
    """Switch-style MoE feed-forward block.

    - Router chooses top-k experts per token (k=1 or 2).
    - Each expert is an independent MLP.
    - No capacity limit; simple scatter/gather.
    - Returns (output, aux_loss) so caller can add load-balance penalty.
    """
    def __init__(self, n_embd: int, num_experts: int = 4, topk: int = 1, dropout: float = 0.0):
        super().__init__()
        assert topk in (1, 2), "Only top-1 or top-2 supported in this tiny demo"
        self.num_experts = num_experts
        self.topk = topk
        self.experts = nn.ModuleList([ExpertMLP(n_embd) for _ in range(num_experts)])
        self.router = nn.Linear(n_embd, num_experts)
        self.dropout = nn.Dropout(dropout)

    @torch.no_grad()
    def _tokens_per_expert(self, assignments: torch.Tensor) -> torch.Tensor:
        # assignments: (N,) ints in [0, E)
        return torch.bincount(assignments, minlength=self.num_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        N = B * T
        x_flat = x.view(N, C)

        logits = self.router(x_flat)              # (N, E)
        gates = F.softmax(logits, dim=-1)        # (N, E)

        if self.topk == 1:
            probs, idx = gates.max(dim=-1)       # (N,), (N,)
            out = torch.zeros_like(x_flat)
            # per-expert processing
            for e in range(self.num_experts):
                sel = (idx == e).nonzero(as_tuple=False).squeeze(-1)
                if sel.numel() == 0:
                    continue
                xe = x_flat.index_select(0, sel)
                ye = self.experts[e](xe)
                # weight by gate prob (Switch Transformer multiplies by gate)
                ye = ye * probs.index_select(0, sel).unsqueeze(-1)
                out.index_copy_(0, sel, ye)
            # auxiliary loss: encourage uniform assignment across experts
            with torch.no_grad():
                counts = self._tokens_per_expert(idx).float().to(x.device)
                frac = counts / max(counts.sum(), torch.tensor(1.0, device=x.device))
            uniform = torch.full_like(frac, 1.0 / self.num_experts)
            aux = F.mse_loss(frac, uniform)
            return self.dropout(out.view(B, T, C)), aux
        else:
            # top-2 routing: weighted sum of two experts
            probs, idx = torch.topk(gates, k=2, dim=-1)  # (N,2),(N,2)
            out = torch.zeros_like(x_flat)
            for e in range(self.num_experts):
                # tokens where expert e is first or second choice
                sel1 = (idx[:, 0] == e).nonzero(as_tuple=False).squeeze(-1)
                sel2 = (idx[:, 1] == e).nonzero(as_tuple=False).squeeze(-1)
                if sel1.numel() + sel2.numel() == 0:
                    continue
                # concatenate for a single forward for expert e
                sel = torch.cat([sel1, sel2], dim=0) if sel1.numel() and sel2.numel() else (sel1 if sel1.numel() else sel2)
                xe = x_flat.index_select(0, sel)
                ye = self.experts[e](xe)
                # scatter-add weighted outputs to out
                weights = torch.cat([
                    probs.index_select(0, sel1)[:, 0] if sel1.numel() else torch.empty(0, device=x.device),
                    probs.index_select(0, sel2)[:, 1] if sel2.numel() else torch.empty(0, device=x.device)
                ])
                out.index_add_(0, sel, ye * weights.unsqueeze(-1))
            # aux loss based on mean gate probs per expert
            mean_gates = gates.mean(dim=0)  # (E,)
            uniform = torch.full_like(mean_gates, 1.0 / self.num_experts)
            aux = F.mse_loss(mean_gates, uniform)
            return self.dropout(out.view(B, T, C)), aux


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        if config.moe:
            self.ff = MoEFFN(config.n_embd, config.num_experts, config.topk, config.dropout)
        else:
            self.ff = DenseFFN(config.n_embd, config.dropout)
        self.moe_enabled = config.moe

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        if self.moe_enabled:
            y, aux = self.ff(self.ln2(x))
            x = x + y
            return x, aux
        else:
            x = x + self.ff(self.ln2(x))
            return x, x.new_zeros(())  # zero aux


class TinyMoEGPT(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

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


# ------------------------------
# Training loop
# ------------------------------

def get_batch(data_ids, block_size, batch_size, device):
    # sample random offsets
    ix = torch.randint(0, data_ids.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data_ids[i:i+block_size] for i in ix])
    y = torch.stack([data_ids[i+1:i+1+block_size] for i in ix])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='path to a text file (optional)')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--moe', action='store_true', help='enable MoE feed-forward')
    parser.add_argument('--experts', type=int, default=4)
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--aux', type=float, default=0.01, help='aux load-balance loss weight')
    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--gen', type=int, default=200, help='tokens to generate after training')
    args = parser.parse_args()

    set_seed(args.seed)
    device = default_device()
    print(f"device: {device}")

    text = load_text(args.data)
    tok = CharTokenizer(text)
    data_ids = torch.tensor(tok.encode(text), dtype=torch.long)

    cfg = Config(
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
        dropout=args.dropout,
        moe=args.moe,
        num_experts=args.experts,
        topk=args.topk,
        aux_loss_weight=args.aux,
    )

    model = TinyMoEGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start_time = time.time()

    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        xb, yb = get_batch(data_ids, cfg.block_size, args.batch, device)
        logits, loss_ce, aux_loss = model(xb, yb)
        loss = loss_ce + cfg.aux_loss_weight * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % max(1, args.steps // 10) == 0 or step == 1:
            dt = time.time() - t0
            print(f"step {step:5d}/{args.steps} | loss {loss.item():.4f} (ce {loss_ce.item():.4f} + aux {aux_loss.item():.4f}) | {dt:.1f}s")
            t0 = time.time()

    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print(f"The code block took {elapsed_time:.4f} seconds to execute.")

    # sample
    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(ctx, args.gen)[0].tolist()
    print("\n=== sample ===")
    print(tok.decode(out))


if __name__ == '__main__':
    main()
