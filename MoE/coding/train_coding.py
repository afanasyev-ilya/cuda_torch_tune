import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple
from dataclasses import dataclass, field
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

########################################################################################################

CONTEXT_SIZE = 512

@dataclass
class BaseGPTConfig:
    vocab_size: int
    
    block_size: int = CONTEXT_SIZE
    dropout: float = 0.1

    aux_loss_weight: float = 0.01

    # rope settings
    pos_encoding: str = "rope"
    rope_base: float = 10000.0
    rope_scale: float = 1.0

@dataclass
class MiniGPTConfig(BaseGPTConfig):
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256

@dataclass
class MoEGPTConfig(BaseGPTConfig):
    # MHA settings
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256

    # MoE settings
    num_experts: int = 8
    expert_dim: int = 256

# Factory functions
def create_minigpt_small(vocab_size: int, **kwargs) -> MiniGPTConfig:
    """Small model for quick experiments (~4GB VRAM)"""
    return MiniGPTConfig(
        vocab_size=vocab_size,
        n_layer=4, n_head=8, n_embd=256, block_size=CONTEXT_SIZE,
        **kwargs
    )

def create_minigpt_large(vocab_size: int, **kwargs) -> MiniGPTConfig:
    """Large model for RTX A5000 (~16GB VRAM)"""
    return MiniGPTConfig(
        vocab_size=vocab_size, 
        n_layer=12, n_head=12, n_embd=768, block_size=CONTEXT_SIZE,
        **kwargs
    )

def create_moegpt_small(vocab_size: int, **kwargs) -> MoEGPTConfig:
    """Small MoE model (~6GB VRAM)"""
    return MoEGPTConfig(
        vocab_size=vocab_size,
        n_layer=4, n_head=8, n_embd=512, block_size=CONTEXT_SIZE,
        num_experts=8, expert_dim=512,
        **kwargs
    )

def create_moegpt_large(vocab_size: int, **kwargs) -> MoEGPTConfig:
    """Large MoE model for RTX A5000 (~20GB VRAM)"""
    return MoEGPTConfig(
        vocab_size=vocab_size,
        n_layer=8, n_head=12, n_embd=768, block_size=CONTEXT_SIZE,
        num_experts=16, expert_dim=768,
        **kwargs
    )

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
        self.pos_emb = None
        if config.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.mha_blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer - 1)])
        self.moe_layer = MoELayer(config)
        self.ln_f = nn.LayerNorm(config.n_embd)  # Final layer norm
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

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

class StreamingDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, seq_length=CONTEXT_SIZE, max_samples=1000):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.max_samples = max_samples
        self._current_buffer = []
        self._fill_buffer()
    
    def _fill_buffer(self):
        """Fill buffer with tokenized samples"""
        self._current_buffer = []
        for i, row in enumerate(self.hf_dataset):
            if i >= self.max_samples:
                break
            # Tokenize and split into sequences
            tokens = self.tokenizer.encode(row["content"])
            # Split into chunks of seq_length
            for i in range(0, len(tokens), self.seq_length):
                chunk = tokens[i:i + self.seq_length]
                if len(chunk) == self.seq_length:  # Only use complete sequences
                    self._current_buffer.append(torch.tensor(chunk, dtype=torch.long))
    
    def __len__(self):
        return len(self._current_buffer)
    
    def __getitem__(self, idx):
        return self._current_buffer[idx]

def get_batch_from_dataloader(dataloader):
    """Get batch from DataLoader instead of random sampling"""
    for batch in dataloader:
        x = batch[:, :-1]  # Input sequence
        y = batch[:, 1:]   # Target sequence (shifted by one)
        return x, y
    return None, None

def print_model_info(model):
    # Print model parameter count and estimated size
    param_count = sum(p.numel() for p in model.parameters())
    estimated_size = param_count * 4 / (1024 ** 2)  # size in MB assuming float32 (4 bytes)
    print(f"Model has {param_count:,} parameters.")
    print(f"Estimated model size: {estimated_size:.2f} MB")


def print_memory_stats(step_name=""):
    print(f"\n--- Memory Stats {step_name} ---")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Max Allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    torch.cuda.reset_peak_memory_stats()  # Reset max counter


def train(model, dataset, batch_size=64, epochs=3, lr=3e-4):
    # Create streaming dataset
    stream_dataset = StreamingDataset(dataset, tok, seq_length=CONTEXT_SIZE)
    
    # Create DataLoader with multiple workers
    dataloader = DataLoader(
        stream_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True  # Faster transfer to GPU
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    print("[STATUS] training started....")
    model.train()
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        xb, yb = get_batch_from_dataloader(dataloader)
        xb = xb.to('cuda')
        yb = yb.to('cuda')
        if xb is None:  # Reset if dataset is exhausted
            dataloader.dataset._fill_buffer()
            continue

        logits, loss_ce, aux_loss = model(xb, yb)
        loss = loss_ce + cfg.aux_loss_weight * aux_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % max(1, epochs // 10) == 0 or epoch == 1:
            dt = time.time() - t0
            print(f"epoch {epoch:5d}/{epochs} | loss {loss.item():.4f} (ce {loss_ce.item():.4f} + aux {aux_loss.item():.4f}) | {dt:.1f}s")
            print_memory_stats()

    training_time = time.time() - t0
    print(f"Training time: {training_time:.2f} seconds")


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
    # parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--pos_encoding", type=str, choices=["rope", "learned"], default="rope")
    # tokenizer settings
    parser.add_argument("--tok_path", type=str, default="./tokenizer.json")  # load/save here
    parser.add_argument("--vocab_size", type=int, default=32000)             # for BPE training
    args = parser.parse_args()

    # Load data
    print("[STATUS] preparing dataset...")
    #dataset = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    dataset = load_dataset("/home/i.afanasyev/codeparrot-clean", split="train", streaming=True)
    print("[STATUS] data loaded.")

    # tokenize data
    tok = BPETokenizer(tokenizer_path=args.tok_path if os.path.exists(args.tok_path) else None)
    if tok.tk is None:
        print(f"Training byte-level BPE tokenizer (vocab={args.vocab_size}) on input.txt ...")
        tok.train(files=["./input.txt"], vocab_size=args.vocab_size, save_path=args.tok_path)
        print(f"Saved tokenizer to {args.tok_path}")
    print("[STATUS] tokenizer prepared.")

    # Load model
    if args.model == "miniGPT":
        print("miniGPT")
        cfg = create_minigpt_small(vocab_size=tok.vocab_size)
        model = MiniGPT(cfg).cuda()
    elif args.model == "MoE":
        print("using MoE model")
        cfg = create_moegpt_small(vocab_size=tok.vocab_size)
        model = MoEGPT(cfg).cuda()
    print("[STATUS] model created and moved to CUDA.")

    # Print model info
    print_model_info(model)

    # Train the model
    train(model, dataset, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

    # Generate text with the trained model
    output = inference(model, tok, "def binary_search(arr, target):\n", max_new_tokens=args.max_new_tokens)
    print("Generated text:\n", output)

