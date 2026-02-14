"""
Week 4: Slide 23 - Complete Language Model Training Pipeline

Description: 
    Self-contained implementation of complete language model training.
    Includes data preparation, model definition, training loop, and text generation.
    Uses character-level tokenization with Shakespeare text.

Dependencies:
    - torch
    - matplotlib (for visualization)
    - urllib (for downloading data)

Usage:
    python slide23_language_model_training.py
    
Note: Downloads Shakespeare text automatically and trains a mini GPT model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

# ===== Model Architecture from Slide 21 =====

class GPTBlock(nn.Module):
    """Single transformer block for GPT"""
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        x = x + self.sa(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTModel(nn.Module):
    """GPT Language Model"""
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[GPTBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# ===== Dataset Class =====

class TextDataset(torch.utils.data.Dataset):
    """Dataset for language modeling"""
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.block_size + 1]
        return chunk


# ===== Training Functions =====

def compute_loss(model, batch):
    """Compute language modeling loss"""
    x = batch[:, :-1]
    y = batch[:, 1:]
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return loss


def train_language_model(model, train_dataset, val_dataset, epochs=10, batch_size=64, 
                        learning_rate=3e-4, device='cpu'):
    """Complete training loop for language model"""
    model = model.to(device)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader)
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = compute_loss(model, batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_perplexity = torch.exp(torch.tensor(avg_val_loss))
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_lm_model.pth')
    
    return train_losses, val_losses


# ===== Text Generation =====

def generate(model, idx, max_new_tokens, temperature=1.0, device='cpu'):
    """Generate text autoregressively"""
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ===== Main Training Script =====

if __name__ == "__main__":
    print("="*70)
    print("LANGUAGE MODEL TRAINING - Character-Level GPT")
    print("="*70)
    
    # Download Shakespeare text
    print("\nDownloading Shakespeare text...")
    try:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
        print(f"✓ Downloaded {len(text):,} characters")
    except Exception as e:
        print(f"⚠ Download failed: {e}")
        print("Using sample text instead...")
        text = "This is sample text for language modeling. " * 1000
    
    # Create character-level tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
    
    print(f"Vocabulary size: {vocab_size} unique characters")
    
    # Tokenize
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    print(f"Total tokens: {len(data):,}")
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    # Create datasets
    block_size = 128
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=4,
        n_layer=4,
        block_size=block_size
    )
    
    print(f"\nModel Configuration:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  Block size: {block_size}")
    print(f"  Embedding dim: 256")
    print(f"  Layers: 4")
    
    # Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nTraining on: {device}")
    
    train_losses, val_losses = train_language_model(
        model, train_dataset, val_dataset,
        epochs=10, batch_size=32, learning_rate=3e-4, device=device
    )
    
    # Generate samples
    print("\n" + "="*70)
    print("GENERATED SAMPLES")
    print("="*70)
    
    model.eval()
    for temp in [0.5, 1.0, 1.5]:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        generated = generate(model, context, max_new_tokens=200, temperature=temp, device=device)
        generated_text = decode(generated[0].tolist())
        print(f"\nTemperature={temp}:")
        print(generated_text[:200])
    
    print("\n✓ Training complete!")
