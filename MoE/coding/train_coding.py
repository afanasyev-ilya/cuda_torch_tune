import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple

########################################################################################################

# Byte-level BPE tokenizer (uses `tokenizers` lib) ----
# pip install tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

class BPETokenizer:
    def __init__(self, tokenizer_path: str = None):
        """
        If tokenizer_path exists, loads it. Otherwise call train(...) first.
        """
        if tokenizer_path is not None and os.path.exists(tokenizer_path):
            self.tk = Tokenizer.from_file(tokenizer_path)
        else:
            self.tk = None  # call train() to create

        self._update_special_ids()

    def _update_special_ids(self):
        if self.tk is None:
            self.pad_id = self.bos_id = self.eos_id = None
            return
        self.pad_id = self.tk.token_to_id("<pad>")
        self.bos_id = self.tk.token_to_id("<bos>")
        self.eos_id = self.tk.token_to_id("<eos>")

    @property
    def vocab_size(self):
        if self.tk is None:
            raise ValueError("Tokenizer not initialized. Call train() or load a file.")
        return self.tk.get_vocab_size()

    def train(self, files, vocab_size=32000, min_freq=2, save_path="tokenizer.json"):
        """
        Train on a list of file paths (e.g., ['input.txt']) and save to tokenizer.json.
        """
        self.tk = Tokenizer(BPE(unk_token=None))  # byte-level covers all bytes; no UNK needed
        self.tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tk.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=["<pad>", "<bos>", "<eos>"]
        )
        self.tk.train(files=files, trainer=trainer)
        self.tk.save(save_path)
        self._update_special_ids()
        return save_path

    def encode(self, s: str, add_bos=False, add_eos=False):
        ids = self.tk.encode(s).ids
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.tk.decode(ids)

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

    pos_encoding: str = "rope"      # "rope" or "learned"
    rope_base: float = 10000.0      # theta
    rope_scale: float = 1.0         # >1.0 = NTK-like scaling (allows longer ctx)

# ---- RoPE helper ----
class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0, scale: float = 1.0, max_seq_len: int = 4096):
        super().__init__()
        assert dim % 2 == 0, "Rotary dim must be even"
        self.dim = dim
        self.base = base
        self.scale = scale
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build_cache(self, seq_len: int, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype) * self.scale
        freqs = torch.einsum("t,d->td", t, self.inv_freq)  # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)            # [T, dim]
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        # Store as [T, dim] - we'll reshape when applying
        self.cos_cached = cos
        self.sin_cached = sin
        self.max_seq_len_cached = seq_len

    def get_cos_sin(self, seq_len: int, device, dtype):
        if seq_len > self.max_seq_len_cached or self.cos_cached.device != device:
            self._build_cache(max(seq_len, self.max_seq_len_cached + 1), device, dtype)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

class MHA(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.scale = self.head_dim ** -0.5

        # we use n_embed x n_embed because these are joint projections of qkv among all heads
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)

        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        # causal mask prepared once for maximum block size
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size)
        )

        self.use_rope = (getattr(config, "pos_encoding", "rope") == "rope")
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base, scale=config.rope_scale, max_seq_len=config.block_size)

    @staticmethod
    def _rotate_half(x):
        # [..., dim] with dim even
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, q, k):
        # q,k: [B, H, T, D]
        B, H, T, D = q.shape
        
        # Get cos/sin with shape [T, D]
        cos, sin = self.rope.get_cos_sin(T, q.device, q.dtype)
        
        # Reshape cos/sin to [1, 1, T, D] for broadcasting with [B, H, T, D]
        cos = cos.view(1, 1, T, D)
        sin = sin.view(1, 1, T, D)
        
        # Apply RoPE to queries and keys
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_rotated, k_rotated

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # shape into heads
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q, k = self._apply_rope(q, k)

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
        self.pos_emb = None
        if config.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.fc = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)
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
    
    print("[STATUS] training started....")
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


def inference(model, tok, prompt = "", max_new_tokens=100):
    model.eval()

    # prepare empty context for now
    ctx = torch.zeros((1, 1), dtype=torch.long).to('cuda')

    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device='cuda')

    # infer
    out = model.generate(ctx, max_new_tokens)[0].tolist()
    generated = tok.decode(out)

    return generated


# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # LLM settings
    parser.add_argument("--model", type=str, choices=["miniGPT", "MoE"], required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--pos_encoding", type=str, choices=["rope", "learned"], default="rope")
    # tokenizer settings
    parser.add_argument("--tok_type", type=str, choices=["char", "bpe"], default="bpe")
    parser.add_argument("--tok_path", type=str, default="./tokenizer.json")  # load/save here
    parser.add_argument("--vocab_size", type=int, default=32000)             # for BPE training
    args = parser.parse_args()

    # Load data
    text = load_text(args.train_data)
    print("[STATUS] data loaded.")

    # tokenize data
    if args.tok_type == "char":
        tok = CharTokenizer(text)
    else:
        tok = BPETokenizer(tokenizer_path=args.tok_path if os.path.exists(args.tok_path) else None)
        if tok.tk is None:
            print(f"Training byte-level BPE tokenizer (vocab={args.vocab_size}) on input.txt ...")
            tok.train(files=["./input.txt"], vocab_size=args.vocab_size, save_path=args.tok_path)
            print(f"Saved tokenizer to {args.tok_path}")
    data_ids = torch.tensor(tok.encode(text), dtype=torch.long)
    print("[STATUS] data tokenized.")

    # Load model
    if args.model == "miniGPT":
        print("miniGPT")
        cfg = MiniGPTConfig(vocab_size=tok.vocab_size)
        model = MiniGPT(cfg).cuda()
    elif args.model == "MoE":
        print("using MoE model")
        cfg = MoEGPTConfig(vocab_size=tok.vocab_size)
        model = MoEGPT(cfg).cuda()
    print("[STATUS] model created and moved to CUDA.")

    # Print model info
    print_model_info(model)

    # Train the model
    train(model, data_ids, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

    # Generate text with the trained model
    output = inference(model, tok, "def binary_search(arr, target):\n", max_new_tokens=args.max_new_tokens)
    print("Generated text:\n", output)

