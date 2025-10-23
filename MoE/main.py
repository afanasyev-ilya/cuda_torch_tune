import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# --- Model Definitions ---
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size, dropout=0.1):
        super(TinyGPT, self).__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size

        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks (MHA layers)
        self.blocks = nn.ModuleList([self._build_block() for _ in range(n_layer)])

        # Output layer
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
    def _build_block(self):
        """Builds a single transformer block with MHA and a feed-forward layer"""
        return nn.ModuleList([
            nn.LayerNorm(self.n_embd),
            nn.MultiheadAttention(self.n_embd, self.n_head),
            nn.LayerNorm(self.n_embd),
            nn.Linear(self.n_embd, 4 * self.n_embd),
            nn.GELU(),
            nn.Linear(4 * self.n_embd, self.n_embd),
            nn.Dropout(0.1)
        ])
    
    def forward(self, x):
        B, T = x.size()

        # Embedding lookup
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.drop(x)

        # Forward through transformer blocks
        for block in self.blocks:
            x = self._forward_block(x, block)

        # Final layer normalization and output
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    def _forward_block(self, x, block):
        ln1, mha, ln2, ff1, act, ff2, drop = block
        # Self-attention
        x_res = x
        x = ln1(x)
        attn_out, _ = mha(x, x, x)
        x = x_res + attn_out

        # Feed-forward
        x_res = x
        x = ln2(x)
        x = ff2(act(ff1(x)))
        x = x_res + drop(x)
        
        return x

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

class MoE_MHA(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, n_head, block_size, num_experts, topk, dropout=0.1):
        super(MoE_MHA, self).__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.block_size = block_size
        self.num_experts = num_experts
        self.topk = topk

        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks (MHA + MoE)
        self.blocks = nn.ModuleList([self._build_block() for _ in range(n_layer)])

        # Output layer
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        
    def _build_block(self):
        """Builds a single transformer block with MHA and MoE"""
        return nn.ModuleList([
            nn.LayerNorm(self.n_embd),
            nn.MultiheadAttention(self.n_embd, self.n_head),
            nn.LayerNorm(self.n_embd),
            MoEFFN(self.n_embd, self.num_experts, self.topk),
        ])
    
    def forward(self, x):
        B, T = x.size()

        # Embedding lookup
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.drop(x)

        # Forward through transformer blocks
        for block in self.blocks:
            x = self._forward_block(x, block)

        # Final layer normalization and output
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    def _forward_block(self, x, block):
        ln1, mha, ln2, moe = block
        # Self-attention
        x_res = x
        x = ln1(x)
        attn_out, _ = mha(x, x, x)
        x = x_res + attn_out

        # MoE (feed-forward)
        x_res = x
        x = ln2(x)
        x = moe(x)
        x = x_res + x
        
        return x

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

class MoE(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, num_experts, dropout=0.1):
        super(MoE, self).__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.num_experts = num_experts

        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(512, n_embd)  # Adjusted position embedding length
        self.drop = nn.Dropout(dropout)

        # MoE blocks (no attention)
        self.blocks = nn.ModuleList([self._build_block() for _ in range(n_layer)])

        # Output layer
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def _build_block(self):
        """Builds a single MoE block (no attention)"""
        return nn.ModuleList([
            MoEFFN(self.n_embd, self.num_experts, 1),  # Just MoE (no attention)
        ])
    
    def forward(self, x):
        B, T = x.size()

        # Embedding lookup
        pos = torch.arange(0, T, device=x.device).unsqueeze(0)
        x = self.tok_emb(x) + self.pos_emb(pos)
        x = self.drop(x)

        # Forward through MoE blocks
        for block in self.blocks:
            x = self._forward_block(x, block)

        # Final layer normalization and output
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
    
    def _forward_block(self, x, block):
        moe = block[0]
        return moe(x)

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

class MoEFFN(nn.Module):
    def __init__(self, n_embd, num_experts, topk):
        super(MoEFFN, self).__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.experts = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(num_experts)])
        self.router = nn.Linear(n_embd, num_experts)
        
    def forward(self, x):
        B, T, C = x.size()
        logits = self.router(x)
        gates = F.softmax(logits, dim=-1)
        
        out = torch.zeros_like(x)
        for e in range(self.num_experts):
            sel = (gates[:, :, e] > 0.5).nonzero(as_tuple=False).squeeze(-1)
            if sel.numel() > 0:
                out.index_add_(0, sel, self.experts[e](x[sel]))
        
        return out

# --- Main Training and Inference Script ---
def load_data(file_path, block_size=128):
    with open(file_path, 'r') as f:
        text = f.read()
    # Basic tokenization by character
    vocab = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for i, ch in enumerate(vocab)}
    data = [stoi[c] for c in text]
    return data, stoi, itos

def get_batch(data, batch_size, block_size):
    # Non-naive batching: efficient handling
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([torch.tensor(data[i:i + block_size]) for i in ix])
    y = torch.stack([torch.tensor(data[i + 1:i + 1 + block_size]) for i in ix])
    return x, y

def train(model, data, batch_size=64, block_size=128, epochs=3, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            x_batch, y_batch = get_batch(data, batch_size, block_size)
            x_batch = x_batch.cuda()  # Move batch to GPU
            y_batch = y_batch.cuda()  # Move batch to GPU

            optimizer.zero_grad()
            logits = model(x_batch)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item()}")

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

def inference(model, prompt, stoi, itos, max_new_tokens=100):
    model.eval()
    idx = torch.tensor([stoi[c] for c in prompt]).unsqueeze(0).cuda()  # Move to GPU
    generated = model.generate(idx, max_new_tokens)
    return ''.join([itos[i.item()] for i in generated[0]])

# --- Main ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["tinyGPT", "MoE_MHA", "MoE"], required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    # Load model
    if args.model == "tinyGPT":
        model = TinyGPT(vocab_size=256, n_embd=128, n_layer=4, n_head=4, block_size=128).cuda()
    elif args.model == "MoE_MHA":
        model = MoE_MHA(vocab_size=256, n_embd=128, n_layer=4, n_head=4, block_size=128, num_experts=4, topk=1).cuda()
    elif args.model == "MoE":
        model = MoE(vocab_size=256, n_embd=128, n_layer=4, num_experts=4).cuda()

    # Load data
    data, stoi, itos = load_data("./input.txt")

    # Train the model
    train(model, data, batch_size=args.batch_size, epochs=args.epochs, lr=args.lr)

    # Generate text with the trained model
    prompt = "What is the capital of France?"
    output = inference(model, prompt, stoi, itos)
    print("Generated text:", output)
