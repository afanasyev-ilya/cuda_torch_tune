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
from checkpoints import *

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
class MoEGPTConfig(BaseGPTConfig):
    # MHA settings
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256

    # MoE settings
    num_experts: int = 8
    expert_dim: int = 256

def create_moegpt_deepseek_style(vocab_size: int, **kwargs) -> MoEGPTConfig:
    """DeepSeek-style alternating MHA -> MoE architecture"""
    return MoEGPTConfig(
        vocab_size=vocab_size,
        # For DeepSeek style, n_layer means number of (MHA + MoE) blocks
        n_layer=6,           # Total blocks: 12 MHA + 12 MoE layers
        n_head=16,           
        n_embd=1024,         
        block_size=CONTEXT_SIZE,
        # Each MoE layer gets these settings
        num_experts=16,      # Slightly fewer experts per layer but more layers
        expert_dim=2048,     
        dropout=0.1,
        aux_loss_weight=0.01,
        pos_encoding="rope",
        **kwargs
    )

########################################################################################################

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

    def train(self, files=None, dataset=None, vocab_size=32000, min_freq=2, save_path="tokenizer.json", max_samples=1000):
        """
        Train on either files or a dataset.
        """
        self.tk = Tokenizer(BPE(unk_token=None))
        self.tk.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tk.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=["<pad>", "<bos>", "<eos>"]
        )
        
        if files is not None:
            # Original file-based training
            self.tk.train(files=files, trainer=trainer)
        elif dataset is not None:
            # New dataset-based training
            self._train_from_dataset(dataset, trainer, max_samples)
        else:
            raise ValueError("Either 'files' or 'dataset' must be provided")
            
        self.tk.save(save_path)
        self._update_special_ids()
        return save_path

    def _train_from_dataset(self, dataset, trainer, max_samples=1000):
        """Train tokenizer from a Hugging Face dataset"""
        def batch_iterator(batch_size=1000):
            samples_processed = 0
            for example in dataset:
                if samples_processed >= max_samples:
                    break
                yield example["content"]
                samples_processed += 1
        
        # Train using the iterator
        self.tk.train_from_iterator(
            batch_iterator(), 
            trainer=trainer, 
            length=max_samples  # Helps with progress reporting
        )

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


import torch.backends.cuda as cuda
cuda.enable_flash_sdp(True)  # Enable Flash Attention if available

class MHA(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Use fused QKV projection for better memory efficiency
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.use_rope = (getattr(config, "pos_encoding", "rope") == "rope")
        if self.use_rope:
            self.rope = RotaryEmbedding(self.head_dim, base=config.rope_base, 
                                      scale=config.rope_scale, max_seq_len=config.block_size)

    @staticmethod
    def _rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rope(self, q, k):
        # Same RoPE implementation as before
        B, H, T, D = q.shape
        cos, sin = self.rope.get_cos_sin(T, q.device, q.dtype)
        cos = cos.view(1, 1, T, D)
        sin = sin.view(1, 1, T, D)
        
        q_rotated = (q * cos) + (self._rotate_half(q) * sin)
        k_rotated = (k * cos) + (self._rotate_half(k) * sin)
        return q_rotated, k_rotated

    def forward(self, x):
        B, T, C = x.shape
        
        # Fused QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, nh, T, hd)

        if self.use_rope:
            q, k = self._apply_rope(q, k)

        # Use PyTorch's built-in scaled_dot_product_attention (uses Flash Attention when available)
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=self.dropout.p if self.training else 0,
                is_causal=True
            )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y

######################################

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
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MHA(config)
        self.ff = DenseFFN(config.n_embd, config.dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x, x.new_zeros(())  # zero aux

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
        self.config = config
        self.router = nn.Linear(config.n_embd, config.num_experts, bias=False)
        self.experts = nn.ModuleList([MoEExpert(config) for _ in range(config.num_experts)])
        self.top_k = 2
        self.noise_epsilon = 1e-2

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(B * T, C)
        
        # Router with noise for load balancing
        router_logits = self.router(x_flat)
        if self.training:
            router_logits = router_logits + torch.randn_like(router_logits) * self.noise_epsilon
        
        # Get top-k experts - VECTORIZED
        router_probs = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(router_probs, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # Create expert masks - VECTORIZED
        expert_mask = torch.zeros(B * T, self.config.num_experts, device=x.device)
        expert_mask.scatter_(1, topk_indices, topk_weights)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert in batch - VECTORIZED
        for expert_idx, expert in enumerate(self.experts):
            # Get tokens that use this expert
            mask = expert_mask[:, expert_idx] > 0
            if mask.any():
                expert_input = x_flat[mask]
                expert_output = expert(expert_input)
                
                # Apply weights - VECTORIZED
                weights = expert_mask[mask, expert_idx].unsqueeze(-1)
                output[mask] += weights * expert_output
        
        output = output.reshape(B, T, C)
        
        # Aux loss
        aux_loss = self._compute_aux_loss(router_probs, topk_indices) if self.training else 0.0
        
        return output, aux_loss

    def _compute_aux_loss(self, router_probs, topk_indices):
        # Expert usage statistics - VECTORIZED
        expert_usage = torch.zeros(self.config.num_experts, device=router_probs.device)
        for expert_idx in range(self.config.num_experts):
            expert_usage[expert_idx] = (topk_indices == expert_idx).float().mean()
        
        target_usage = torch.ones_like(expert_usage) / self.config.num_experts
        aux_loss = F.mse_loss(expert_usage, target_usage)
        return aux_loss

# NEW: DeepSeek-style alternating MHA -> MoE blocks
class MHAThenMoEBlock(nn.Module):
    """DeepSeek-style block: MHA followed by MoE (replaces dense FFN)"""
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = MHA(config)
        self.moe = MoELayer(config)  # Replace dense FFN with MoE

    def forward(self, x):
        # MHA part
        x = x + self.attn(self.ln1(x))
        # MoE part (replaces FFN)
        moe_out, aux_loss = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss


class MoEGPT(nn.Module):
    def __init__(self, config: MoEGPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = None
        if config.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # NEW: Use alternating MHA -> MoE blocks (DeepSeek style)
        self.blocks = nn.ModuleList([
            MHAThenMoEBlock(config) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
            x = x + self.pos_emb(pos)

        x = self.drop(x)
        aux_total = 0.0
        
        # NEW: Process all MHA->MoE blocks
        for block in self.blocks:
            x, aux_loss = block(x)
            aux_total = aux_total + aux_loss
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss, aux_total

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature=0.8):
        # Use autocast in generation loop
        with torch.cuda.amp.autocast(dtype=torch.float16):
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.config.block_size:]
                logits, _, _ = self.forward(idx_cond)
                logits = logits[:, -1, :] / temperature
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


def analyze_memory_usage(model, batch_size=16, seq_length=512):
    """Analyze memory usage by layer and provide scaling recommendations"""
    print("=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)
    
    total_params = 0
    layer_breakdown = {}
    
    # Analyze embedding layer
    emb_params = sum(p.numel() for p in model.tok_emb.parameters())
    if model.pos_emb is not None:
        emb_params += sum(p.numel() for p in model.pos_emb.parameters())
    total_params += emb_params
    layer_breakdown["Embedding"] = emb_params
    print(f"Embedding layers: {emb_params:,} parameters")
    
    # Analyze blocks (now MHA+MoE blocks)
    block_params = 0
    for i, block in enumerate(model.blocks):
        block_param_count = sum(p.numel() for p in block.parameters())
        block_params += block_param_count
        layer_breakdown[f"Block {i+1} (MHA+MoE)"] = block_param_count
        print(f"Block {i+1} (MHA+MoE): {block_param_count:,} parameters")
    
    total_params += block_params
    print(f"Total blocks: {block_params:,} parameters")
    
    # Final layers
    final_params = sum(p.numel() for p in model.ln_f.parameters()) + sum(p.numel() for p in model.head.parameters())
    total_params += final_params
    layer_breakdown["Final Layers"] = final_params
    print(f"Final layers: {final_params:,} parameters")
    
    print("-" * 60)
    print(f"TOTAL MODEL: {total_params:,} parameters")
    
    # Memory calculations
    param_memory_mb = total_params * 4 / (1024 ** 2)  # FP32
    param_memory_mb_fp16 = total_params * 2 / (1024 ** 2)  # FP16
    
    # Activation memory estimation (rough)
    activation_memory_mb = (batch_size * seq_length * model.config.n_embd * 10) / (1024 ** 2)  # Conservative estimate
    
    # Optimizer memory (AdamW: 2x for moments + 1x for params)
    optimizer_memory_mb = param_memory_mb * 3
    
    total_training_memory_mb = param_memory_mb_fp16 + activation_memory_mb + optimizer_memory_mb
    
    print("\nMEMORY BREAKDOWN:")
    print(f"Parameters (FP32): {param_memory_mb:.2f} MB")
    print(f"Parameters (FP16): {param_memory_mb_fp16:.2f} MB")
    print(f"Activations (est.): {activation_memory_mb:.2f} MB")
    print(f"Optimizer (AdamW): {optimizer_memory_mb:.2f} MB")
    print(f"TOTAL TRAINING: {total_training_memory_mb:.2f} MB")
    print(f"RTX A5000 VRAM: 24,000 MB")
    print(f"AVAILABLE HEADROOM: {24000 - total_training_memory_mb:.2f} MB")
    
    return total_params, layer_breakdown


def train(model, dataset, batch_size=16, epochs=3, lr=3e-4, grad_accum_steps=2, 
          checkpoint_dir="checkpoints", save_every=10, resume_from=None):
    """Training with checkpoint saving and resuming"""
    
    stream_dataset = StreamingDataset(dataset, tok, seq_length=CONTEXT_SIZE)
    dataloader = DataLoader(stream_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scaler = torch.cuda.amp.GradScaler()
    
    start_epoch = 1
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if resume_from == "latest":
        latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            start_epoch, best_loss = load_checkpoint(model, optimizer, latest_checkpoint)
            start_epoch += 1  # Start from next epoch
            print(f"Resuming training from epoch {start_epoch}")
    elif resume_from and os.path.exists(resume_from):
        start_epoch, best_loss = load_checkpoint(model, optimizer, resume_from)
        start_epoch += 1
        print(f"Resuming training from epoch {start_epoch}")
    
    print(f"[STATUS] Training started from epoch {start_epoch}...")
    model.train()
    
    for epoch in range(start_epoch, epochs + 1):
        total_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(dataloader):
            xb = batch[:, :-1].to('cuda', non_blocking=True)
            yb = batch[:, 1:].to('cuda', non_blocking=True)

            with torch.cuda.amp.autocast():
                logits, loss_ce, aux_loss = model(xb, yb)
                loss = loss_ce + model.config.aux_loss_weight * aux_loss
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * grad_accum_steps
            num_batches += 1

            if step % 100 == 0:
                avg_loss = total_loss / (step + 1)
                print(f"epoch {epoch:3d}/{epochs} | step {step:4d} | loss {avg_loss:.4f}")

        avg_epoch_loss = total_loss / num_batches
        print(f"epoch {epoch:3d}/{epochs} | avg_loss {avg_epoch_loss:.4f}")
        
        # Save checkpoint
        if epoch % save_every == 5 or epoch == epochs:
            save_checkpoint(model, optimizer, epoch, avg_epoch_loss, checkpoint_dir)
            
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, epoch, avg_epoch_loss, checkpoint_dir)
            # Also save as best model
            best_path = f"{checkpoint_dir}/checkpoint_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'config': model.config.__dict__,
            }, best_path)
            print(f"New best model saved with loss: {best_loss:.4f}")
        
        # Generate sample to monitor progress
        if epoch % 20 == 0:
            model.eval()
            sample = inference(model, tok, "def binary_search(arr, target):\n", max_new_tokens=100)
            print(f"\n--- Epoch {epoch} Sample ---")
            print(sample[:500] + "..." if len(sample) > 500 else sample)
            print("---" + "-" * 20)
            model.train()


def inference(model, tok, prompt = "", max_new_tokens=100):
    model = model.half()
    model.eval()

    # prepare empty context for now
    ctx = torch.zeros((1, 1), dtype=torch.long).to('cuda')

    ids = tok.encode(prompt)
    ctx = torch.tensor([ids], dtype=torch.long, device='cuda')

    # infer
    out = model.generate(ctx, max_new_tokens)[0].tolist()
    generated = tok.decode(out)

    return generated


def generate_multiple_samples(model, tok, prompt, num_samples=3, max_new_tokens=150, temperature=0.7):
    """Generate multiple samples and pick the best one"""
    samples = []
    
    for i in range(num_samples):
        # Vary temperature slightly for diversity
        current_temp = temperature * (0.8 + 0.4 * (i / num_samples))
        sample = inference(model, tok, prompt, max_new_tokens, temperature=current_temp)
        samples.append(sample)
        
        print(f"--- Sample {i+1} (temp={current_temp:.2f}) ---")
        print(sample)
        print()
    
    # Simple heuristic: pick the one with most complete function structure
    best_sample = max(samples, key=lambda s: (
        s.count('def '),
        s.count('return '),
        s.count(':') - s.count('":'),  # Count colons but not in strings
        len(s)
    ))
    
    return best_sample, samples


def print_torch_stats():
    # Check if FlashAttention is available in your PyTorch installation
    print("FlashAttention available:", torch.backends.cuda.flash_sdp_enabled())

    # List available backends
    from torch.backends.cuda import SDPBackend
    print("\nAvailable SDP backends:")
    with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=True):
        print("All backends enabled for context")


# --- Main ---
if __name__ == "__main__":
    print_torch_stats()

    parser = argparse.ArgumentParser()
    # LLM settings
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_new_tokens", type=int, default=1000)
    parser.add_argument("--pos_encoding", type=str, choices=["rope", "learned"], default="rope")
    parser.add_argument("--model_arch", type=str, choices=["deepseek", "optimized"], default="deepseek")
    
    # Checkpoint settings
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, 
                       help="Resume from 'latest', or path to specific checkpoint")
    parser.add_argument("--save_final_model", action="store_true", default=True, 
                       help="Save final model after training")
    
    # tokenizer settings
    parser.add_argument("--tok_path", type=str, default="./tokenizer.json")
    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()

    # Load data
    print("[STATUS] preparing dataset...")
    dataset = load_dataset("/home/i.afanasyev/codeparrot-clean", split="train", streaming=True)
    print("[STATUS] data loaded.")

    # tokenize data
    tok = BPETokenizer(tokenizer_path=args.tok_path if os.path.exists(args.tok_path) else None)
    if tok.tk is None:
        print(f"[STATUS] Training byte-level BPE tokenizer (vocab={args.vocab_size}) on dataset...")
        tok.train(
            dataset=dataset,
            vocab_size=args.vocab_size, 
            save_path=args.tok_path,
            max_samples=1000
        )
        print(f"Saved tokenizer to {args.tok_path}")
    print("[STATUS] tokenizer prepared.")

    # Load or create model
    if args.resume_from and args.resume_from != "latest" and os.path.exists(args.resume_from):
        # Load existing model from checkpoint
        model = load_model(MoEGPT, args.resume_from)
        print(f"[STATUS] Model loaded from checkpoint: {args.resume_from}")
    else:
        # Create new model
        if args.model_arch == "deepseek":
            cfg = create_moegpt_deepseek_style(vocab_size=tok.vocab_size)
            print("[ARCHITECTURE] Using DeepSeek-style: alternating MHA -> MoE blocks")
        else:
            cfg = create_moegpt_a5000_optimized(vocab_size=tok.vocab_size)
            print("[ARCHITECTURE] Using optimized single-MoE architecture")
        
        model = MoEGPT(cfg).cuda()
        print("[STATUS] New model created and moved to CUDA.")

    # Print model info
    total_params, breakdown = analyze_memory_usage(model, batch_size=args.batch_size, seq_length=CONTEXT_SIZE)
    
    print(f"\nARCHITECTURE SUMMARY:")
    print(f"• Total blocks: {model.config.n_layer}")
    print(f"• Each block: MHA -> MoE")
    print(f"• Attention: {model.config.n_head} heads, {model.config.n_embd} dim")
    print(f"• MoE per block: {model.config.num_experts} experts, {model.config.expert_dim} dim")
    print(f"• Context: {model.config.block_size} tokens")

    # Train the model
    train(model, dataset, 
          batch_size=args.batch_size, 
          epochs=args.epochs, 
          lr=args.lr,
          checkpoint_dir=args.checkpoint_dir,
          save_every=args.save_every,
          resume_from=args.resume_from)

    # Save final model
    if args.save_final_model:
        model_dir = "saved_models"
        save_model(model, model_dir)
        save_tokenizer(tok, model_dir)
        print(f"Final model and tokenizer saved to {model_dir}")

    # Generate text with the trained model
    print("\n" + "="*50)
    print("GENERATION EXAMPLES")
    print("="*50)
    
    prompts = [
        "def binary_search(arr, target):\n",
        "def quicksort(arr):\n",
        "def fibonacci(n):\n",
        "class LinkedList:\n    def __init__(self):\n",
    ]
    
    for prompt in prompts:
        print(f"\n--- Generating for: {prompt.strip()} ---")
        output = inference(model, tok, prompt, max_new_tokens=args.max_new_tokens)
        print(output)
        print("-" * 40)

    # Also try multiple samples for the main prompt
    print("\n--- Multiple samples for binary_search ---")
    best_code, all_samples = generate_multiple_samples(
        model, tok, 
        "def binary_search(arr, target):\n",
        num_samples=3
    )
    
    print("\n--- BEST SAMPLE ---")
    print(best_code)
