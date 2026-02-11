# Week 4: Deep Learning and the GenAI General Algorithm - Slides Batch 3 (Slides 21-30)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 11, 2026

---

## Slide 21: GPT Architecture

### Generative Pre-trained Transformer

**GPT = Decoder-Only Transformer**

**Key Characteristics:**
- Uses only the decoder part of transformer
- Masked self-attention (causal)
- Pre-trained on massive text data
- Fine-tuned for specific tasks

**Architecture:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GPTBlock(nn.Module):
    """Single transformer block for GPT"""
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
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
        # Self-attention with causal mask
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
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            GPTBlock(n_embd, n_head, dropout) 
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight tying
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
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, max is {self.block_size}"
        
        # Token + position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Generate logits
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        return logits


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("GPT MODEL EXAMPLE")
    print("="*60)
    
    # Mini GPT config (scaled down for demonstration)
    vocab_size = 5000
    n_embd = 256
    n_head = 4
    n_layer = 4
    block_size = 128
    
    # Create model
    model = GPTModel(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size
    )
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary Size: {vocab_size:,}")
    print(f"  Embedding Dimension: {n_embd}")
    print(f"  Number of Heads: {n_head}")
    print(f"  Number of Layers: {n_layer}")
    print(f"  Block Size (max context): {block_size}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Test forward pass
    batch_size = 4
    seq_length = 50
    
    # Random token IDs (simulating tokenized text)
    sample_tokens = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    print(f"\n\nInput:")
    print(f"  Shape: {sample_tokens.shape}")
    print(f"  Sample tokens (first sequence): {sample_tokens[0, :10].tolist()}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(sample_tokens)
    
    print(f"\n\nOutput:")
    print(f"  Logits Shape: {logits.shape}")
    print(f"  (batch_size, sequence_length, vocab_size)")
    
    # Get next token predictions
    next_token_logits = logits[:, -1, :]  # Take last position
    probs = F.softmax(next_token_logits, dim=-1)
    next_tokens = torch.argmax(probs, dim=-1)
    
    print(f"\n\nNext Token Predictions:")
    for i in range(batch_size):
        print(f"  Sequence {i+1}: Next token ID = {next_tokens[i].item()}, "
              f"Probability = {probs[i, next_tokens[i]].item():.3f}")
    
    print("\n" + "="*60)
    print("✓ GPT model successfully processes and predicts next tokens!")
    print("Use case: Text generation, completion, translation")
    print("="*60)
    
    # Show GPT-2 comparison
    print("\n\nGPT-2 Configuration Comparison:")
    configs = {
        'GPT-2 Small': {'params': '124M', 'n_layer': 12, 'n_embd': 768, 'n_head': 12},
        'GPT-2 Medium': {'params': '350M', 'n_layer': 24, 'n_embd': 1024, 'n_head': 16},
        'GPT-2 Large': {'params': '774M', 'n_layer': 36, 'n_embd': 1280, 'n_head': 20},
        'GPT-2 XL': {'params': '1.5B', 'n_layer': 48, 'n_embd': 1600, 'n_head': 25},
    }
    
    for name, config in configs.items():
        print(f"  {name}: {config['params']} parameters")
        print(f"    Layers: {config['n_layer']}, Embedding: {config['n_embd']}, Heads: {config['n_head']}")
```

---

## Slide 22: How GPT Generates Text

### Autoregressive Generation

**Process:**
1. Start with prompt tokens
2. Model predicts next token
3. Add predicted token to sequence
4. Repeat until done

**Implementation:**
```python
import torch
import torch.nn.functional as F
from typing import Optional

def generate(model: torch.nn.Module, 
             idx: torch.Tensor, 
             max_new_tokens: int, 
             temperature: float = 1.0, 
             top_k: Optional[int] = None,
             block_size: int = 1024) -> torch.Tensor:
    """
    Generate text autoregressively using a language model.
    
    Args:
        model: Trained language model (e.g., GPT)
        idx: Initial context tokens of shape (batch, sequence_length)
        max_new_tokens: Number of new tokens to generate
        temperature: Controls randomness (higher = more random)
                    - temperature < 1: More focused/deterministic
                    - temperature = 1: Normal sampling
                    - temperature > 1: More random/creative
        top_k: If specified, only sample from top k most likely tokens
        block_size: Maximum context length the model can handle
    
    Returns:
        Generated token sequence of shape (batch, sequence_length + max_new_tokens)
    
    Example:
        >>> model = GPTModel(...)
        >>> prompt_tokens = torch.tensor([[1, 2, 3, 4]])  # "Hello world"
        >>> generated = generate(model, prompt_tokens, max_new_tokens=20)
        >>> # Decode generated tokens back to text
    """
    model.eval()  # Set to evaluation mode
    
    with torch.no_grad():  # No gradient computation needed
        for _ in range(max_new_tokens):
            # Crop context if it exceeds model's maximum length
            idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
            
            # Forward pass: get predictions for next token
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # Focus on last position: (batch, vocab_size)
            
            # Apply temperature scaling
            # Lower temperature → more confident predictions
            # Higher temperature → more diverse predictions
            logits = logits / temperature
            
            # Optional: Top-k sampling (only consider top k tokens)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # (batch, vocab_size)
            
            # Sample next token from probability distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Append sampled token to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq_len + 1)
    
    return idx


# Complete usage example
if __name__ == "__main__":
    from transformers import GPT2Tokenizer, GPT2LMHeadModel
    
    # Load pre-trained model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    # Encode prompt
    prompt = "The future of AI is"
    tokens = tokenizer.encode(prompt, return_tensors='pt')
    print(f"Prompt: {prompt}")
    print(f"Token IDs: {tokens}")
    
    # Generate with different temperatures
    print("\n--- Temperature = 0.5 (More focused) ---")
    generated_low = generate(model, tokens, max_new_tokens=30, temperature=0.5)
    text_low = tokenizer.decode(generated_low[0].tolist())
    print(text_low)
    
    print("\n--- Temperature = 1.0 (Balanced) ---")
    generated_mid = generate(model, tokens, max_new_tokens=30, temperature=1.0)
    text_mid = tokenizer.decode(generated_mid[0].tolist())
    print(text_mid)
    
    print("\n--- Temperature = 1.5 (More creative) ---")
    generated_high = generate(model, tokens, max_new_tokens=30, temperature=1.5)
    text_high = tokenizer.decode(generated_high[0].tolist())
    print(text_high)
    
    print("\n--- With top_k = 10 ---")
    generated_topk = generate(model, tokens, max_new_tokens=30, temperature=1.0, top_k=10)
    text_topk = tokenizer.decode(generated_topk[0].tolist())
    print(text_topk)
```

---

## Slide 23: Training Language Models

### Pre-training and Fine-tuning

**Pre-training Objective:**
- Next token prediction (language modeling)
- Learn from massive unlabeled data
- Captures language patterns and knowledge

**Complete Training Implementation:**
```python
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

class TextDataset(Dataset):
    """
    Dataset for language modeling.
    Creates input-target pairs for next-token prediction.
    """
    def __init__(self, tokens: torch.Tensor, block_size: int):
        """
        Args:
            tokens: Tensor of token IDs
            block_size: Maximum sequence length
        """
        self.tokens = tokens
        self.block_size = block_size
    
    def __len__(self):
        return len(self.tokens) - self.block_size
    
    def __getitem__(self, idx):
        # Get block_size + 1 tokens
        chunk = self.tokens[idx:idx + self.block_size + 1]
        return chunk


def compute_loss(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """
    Compute language modeling loss (next token prediction).
    
    Args:
        model: Language model
        batch: Token sequences of shape (batch_size, block_size + 1)
    
    Returns:
        Cross-entropy loss
    
    How it works:
        Input:  [1, 2, 3, 4, 5] → model predicts → [2, 3, 4, 5, 6]
        Target: [2, 3, 4, 5, 6]
    """
    # Split into input and target
    x = batch[:, :-1]  # Input: first T tokens
    y = batch[:, 1:]   # Target: next T tokens (shifted by 1)
    
    # Forward pass
    logits = model(x)  # Shape: (batch_size, seq_len, vocab_size)
    
    # Reshape for cross-entropy loss
    # From: (B, T, V) → (B*T, V)
    # From: (B, T) → (B*T,)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )
    
    return loss


def train_language_model(
    model: torch.nn.Module,
    train_dataset: TextDataset,
    val_dataset: TextDataset,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Complete training loop for language model.
    
    Args:
        model: Language model to train
        train_dataset: Training dataset
        val_dataset: Validation dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
    """
    # Move model to device
    model = model.to(device)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Optimizer (AdamW is standard for transformers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * len(train_loader)
    )
    
    # Training loop
    best_val_loss = float('inf')
    step = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute loss
            loss = compute_loss(model, batch)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Track loss
            train_loss += loss.item()
            step += 1
            
            # Logging
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, "
                      f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = compute_loss(model, batch)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_perplexity = torch.exp(torch.tensor(avg_val_loss))
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Perplexity: {val_perplexity:.2f}\n")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_model.pth')
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")


# Complete example usage
if __name__ == "__main__":
    # 1. Prepare data
    import urllib.request
    
    # Download sample text (or use your own)
    try:
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
    except:
        # Fallback to sample text
        text = "This is sample text. " * 1000
    
    # Create character-level tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    
    # Tokenize
    data = torch.tensor(encode(text), dtype=torch.long)
    
    # Split train/val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # 2. Create datasets
    block_size = 128
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    
    # 3. Create model (using GPTModel defined in Slide 21)
    # NOTE: Copy the GPTModel and GPTBlock classes from Slide 21 above,
    # or run this in the same script where they are defined
    
    model = GPTModel(
        vocab_size=vocab_size,
        n_embd=256,
        n_head=4,
        n_layer=4,
        block_size=block_size
    )
    
    print(f"\nModel created:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 4. Train
    train_language_model(
        model,
        train_dataset,
        val_dataset,
        epochs=10,
        batch_size=32,
        learning_rate=3e-4
    )
    
    # 5. Visualize training progress
    import matplotlib.pyplot as plt
    
    def plot_training_progress(train_losses: list, val_losses: list):
        """
        Visualize training and validation loss over time.
        
        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Perplexity curves
        train_perp = [np.exp(loss) for loss in train_losses]
        val_perp = [np.exp(loss) for loss in val_losses]
        
        ax2.plot(epochs, train_perp, 'b-', linewidth=2, label='Training Perplexity')
        ax2.plot(epochs, val_perp, 'r-', linewidth=2, label='Validation Perplexity')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Perplexity', fontsize=12)
        ax2.set_title('Model Perplexity', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved training visualization to 'training_progress.png'")
        plt.show()
```

**Fine-tuning:**
- Adapt pre-trained model to specific task
- Much less data needed
- Few epochs of training

---

## Slide 24: Tokenization

### Breaking Text into Pieces

**Why Tokenization?**
- Models work with numbers, not text
- Need to convert text → tokens → numbers
- Balance between vocabulary size and coverage

**Types of Tokenization:**

**1. Word-level** (Simple but limited)
```python
# Split by spaces
text = "Hello world"
tokens = text.split()  # ['Hello', 'world']
```

**2. Character-level** (Too granular)
```python
text = "Hello"
tokens = list(text)  # ['H', 'e', 'l', 'l', 'o']
```

**3. Subword (BPE) - Used by GPT**
```python
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "Hello world! This is tokenization."
tokens = tokenizer.encode(text)
print(tokens)  # [15496, 995, 0, 770, 318, 11241, 1634, 13]

# Decode back
decoded = tokenizer.decode(tokens)
print(decoded)  # Original text
```

**Building a Simple Tokenizer:**
```python
class SimpleTokenizer:
    def __init__(self, texts):
        # Build vocabulary
        self.vocab = set()
        for text in texts:
            self.vocab.update(text.split())
        
        # Create mappings
        self.vocab = sorted(list(self.vocab))
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
    
    def encode(self, text):
        tokens = text.split()
        return [self.token_to_id.get(token, 0) for token in tokens]
    
    def decode(self, ids):
        tokens = [self.id_to_token.get(id, '<UNK>') for id in ids]
        return ' '.join(tokens)
```

---

## Slide 25: Scaling Laws

### Bigger Models = Better Performance

**Key Findings:**
- Performance scales predictably with:
  - Model size (number of parameters)
  - Dataset size
  - Compute budget

**Scaling Equation (simplified):**
```
Loss ∝ 1 / (N^α)

Where:
N = number of parameters
α ≈ 0.076 (empirically determined)
```

**Model Evolution:**

| Model | Parameters | Training Data | Year |
|-------|-----------|---------------|------|
| GPT-1 | 117M | 5GB | 2018 |
| GPT-2 | 1.5B | 40GB | 2019 |
| GPT-3 | 175B | 570GB | 2020 |
| GPT-4 | ~1.8T | Unknown | 2023 |

**Compute Requirements:**
```python
def estimate_training_cost(num_params, tokens, flops_per_param=6):
    """
    Rough estimate of training cost
    
    num_params: model parameters
    tokens: training tokens
    flops_per_param: FLOPs per parameter per token (approx 6)
    """
    total_flops = num_params * tokens * flops_per_param
    
    # A100 GPU: ~312 TFLOPS
    gpu_flops = 312e12
    
    # Seconds of compute
    seconds = total_flops / gpu_flops
    
    # Convert to GPU-days
    gpu_days = seconds / (24 * 3600)
    
    return gpu_days

# GPT-3 training estimate
params = 175e9
tokens = 300e9
days = estimate_training_cost(params, tokens)
print(f"Estimated: {days:,.0f} GPU-days")
print(f"Cost (at $2/GPU-hour): ${days * 24 * 2:,.0f}")
```

---

## Slide 26: Emergent Abilities

### Capabilities That Appear at Scale

**What Are Emergent Abilities?**

Emergent abilities are one of the most fascinating and mysterious aspects of large language models. These are capabilities that don't exist in smaller models but suddenly "emerge" when models reach a certain scale - like a phase transition in physics where water suddenly becomes ice at 0°C.

**The Emergence Phenomenon:**

Imagine teaching multiplication to a child. At first, they can only do 2×3. With more practice, they can do 5×7. But at some point, something clicks - they suddenly understand the concept and can multiply any numbers, even ones they've never seen before. Large language models experience similar "aha moments" at scale.

**Key Characteristics:**
- **Not present in smaller models**: A 1B parameter model can't do it, but a 10B parameter model can
- **Appear suddenly**: Performance goes from near-zero to strong at a threshold scale
- **Not explicitly trained for**: No one taught the model these specific capabilities
- **Unpredictable**: We can't always predict what abilities will emerge at what scale

**Major Emergent Abilities:**

**1. Few-Shot Learning (In-Context Learning)**

The model learns new tasks from just a few examples in the prompt, without any parameter updates. It's like learning to play a new board game just by watching a few rounds.

```python
prompt = """
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>"""

# Model completes: "fromage"
```

**How It Works:**
- You provide examples in the prompt (the "context")
- The model identifies the pattern from these examples
- It applies this pattern to new inputs
- No retraining or fine-tuning required!

**Business Application:**
You can adapt ChatGPT to your company's style guide just by showing it examples - no technical ML expertise needed.

**2. Chain-of-Thought (CoT) Reasoning**

Perhaps the most remarkable emergent ability: models can "think step by step" to solve complex problems, dramatically improving accuracy on reasoning tasks.

```python
prompt = """
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls.
Each can has 3 tennis balls. How many tennis balls does he have now?

A: Let's think step by step.
Roger started with 5 balls.
He bought 2 cans of 3 balls each.
2 cans × 3 balls = 6 balls
5 + 6 = 11 balls
Therefore, Roger has 11 tennis balls.

Q: The cafeteria had 23 apples. If they used 20 to make lunch and 
bought 6 more, how many apples do they have?

A: Let's think step by step."""

# Model generates step-by-step reasoning:
# "The cafeteria started with 23 apples.
# They used 20 for lunch: 23 - 20 = 3 apples remaining.
# They bought 6 more: 3 + 6 = 9 apples.
# Therefore, they have 9 apples."
```

**Why CoT Is Revolutionary:**
- **Accuracy boost**: 50%+ improvement on math/logic tasks
- **Interpretability**: Can see model's "reasoning"
- **Error detection**: Can spot where reasoning goes wrong
- **Generalizes**: Works across many domains (math, logic, common sense)

**The Magic Phrase**: Simply adding "Let's think step by step" can dramatically improve performance!

**3. Multi-Step Task Decomposition**

Large models can break complex tasks into subtasks automatically:

```python
User: "Plan a week-long trip to Japan"

GPT-4: 
"I'll break this down into steps:
1. Determine budget and dates
2. Book flights
3. Arrange accommodations
4. Plan daily itineraries
5. Make restaurant reservations
6. Organize transportation

Let's start with step 1. What's your budget and preferred travel dates?"
```

**4. Zero-Shot Reasoning**

The ability to perform tasks with NO examples, just instructions:

```python
"Explain quantum computing to a 10-year-old"
# Model generates appropriate explanation without examples
```

**5. Cross-Lingual Transfer**

Models trained primarily on English can perform tasks in other languages they've barely seen:

```python
# Model trained mostly on English
"Translate this technical document from English to Swahili"
# Model performs reasonably well despite limited Swahili training
```

**6. Instruction Following**

Understanding and executing complex, multi-part instructions:

```python
"Read this contract, identify potential legal issues, summarize them 
in bullet points, then draft an email to our lawyer with concerns"
# Model handles all parts correctly
```

**The Scale Factor:**

Research shows these abilities emerge at predictable but surprising thresholds:

| Ability | Emerges Around | Model Examples |
|---------|---------------|----------------|
| Few-shot learning | 1-10B params | GPT-2 → GPT-3 |
| Chain-of-thought | 50-100B params | GPT-3 → GPT-3.5 |
| Complex reasoning | 100B+ params | GPT-4, PaLM 2 |

**Implications for Business:**

1. **Future-proofing**: Today's impossible tasks may become trivial with next-gen models
2. **Investment timing**: Sometimes waiting for larger models is better than engineering solutions
3. **Capability discovery**: Test new models for unexpected abilities
4. **Competitive advantage**: Early adopters of emergent capabilities gain significant advantages

**The Mystery:**

We still don't fully understand WHY these abilities emerge. Current theories:
- **Compression hypothesis**: Models discover efficient representations that generalize
- **Latent knowledge activation**: Capabilities exist but need scale to be accessible
- **Phase transitions**: Like physical systems, behavior changes sharply at thresholds

**Practical Example - Business Use:**

```python
# Small model (GPT-2, 1.5B params): Fails
prompt = "Our Q3 revenue was $2.3M, up 15% from Q2. Q2 was down 5% from Q1. 
         What was our Q1 revenue? Show your work."
# Output: Gibberish or wrong answer

# Large model (GPT-4, ~1.8T params): Succeeds
# Output: "Let me work backwards:
# Q3 revenue: $2.3M
# Q3 was 15% more than Q2, so: Q2 = $2.3M / 1.15 = $2.0M
# Q2 was 5% less than Q1, so: Q1 = $2.0M / 0.95 = $2.105M
# Therefore, Q1 revenue was approximately $2.11M"
```

This ability to reason through multi-step financial calculations emerged only at large scale!

---

## Slide 27: The GenAI Training Pipeline

### From Data to Deployment

**Complete Training Pipeline:**

```python
# 1. Data Collection & Preparation
class DataPipeline:
    def __init__(self):
        self.raw_data = []
        self.processed_data = []
    
    def collect_data(self, sources):
        """Scrape, download, or access data"""
        for source in sources:
            data = load_from_source(source)
            self.raw_data.extend(data)
    
    def clean_data(self):
        """Remove duplicates, filter quality"""
        seen = set()
        for doc in self.raw_data:
            if doc not in seen:
                if self.is_high_quality(doc):
                    self.processed_data.append(doc)
                    seen.add(doc)
    
    def tokenize_data(self, tokenizer):
        """Convert text to tokens"""
        tokenized = []
        for doc in self.processed_data:
            tokens = tokenizer.encode(doc)
            tokenized.append(tokens)
        return tokenized

# 2. Model Architecture
model = GPTModel(
    vocab_size=50257,
    n_embd=768,
    n_head=12,
    n_layer=12,
    block_size=1024
)

# 3. Training Configuration
config = {
    'batch_size': 64,
    'learning_rate': 3e-4,
    'epochs': 10,
    'gradient_accumulation_steps': 4,
    'warmup_steps': 1000,
    'max_grad_norm': 1.0
}

# 4. Training Loop
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=config['warmup_steps'],
    num_training_steps=total_steps
)

for epoch in range(config['epochs']):
    for batch_idx, batch in enumerate(train_loader):
        # Forward pass
        loss = compute_loss(model, batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config['gradient_accumulation_steps'] == 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['max_grad_norm']
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Logging
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            wandb.log({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
    
    # Save checkpoint
    torch.save(model.state_dict(), f'checkpoint_epoch_{epoch}.pth')

# 5. Evaluation
model.eval()
with torch.no_grad():
    for batch in val_loader:
        loss = compute_loss(model, batch)
        perplexity = torch.exp(loss)
        print(f"Validation Perplexity: {perplexity:.2f}")
```

---

## Slide 28: Fine-tuning Techniques

### Adapting Pre-trained Models

**1. Full Fine-tuning**
```python
# Update all parameters
model = GPT2LMHeadModel.from_pretrained('gpt2')
optimizer = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

**2. LoRA (Low-Rank Adaptation)**
```python
from peft import get_peft_model, LoraConfig

# Only train small adapter matrices
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1
)

model = get_peft_model(model, lora_config)
print(f"Trainable params: {model.print_trainable_parameters()}")
# Output: trainable params: 0.5M || all params: 124M || trainable%: 0.4%
```

**3. Prompt Tuning**
```python
# Learn soft prompts instead of model weights
class PromptTuning(nn.Module):
    def __init__(self, n_tokens, token_dim):
        super().__init__()
        self.soft_prompt = nn.Parameter(torch.randn(n_tokens, token_dim))
    
    def forward(self, input_embeds):
        # Prepend learnable tokens
        batch_size = input_embeds.shape[0]
        soft_prompt = self.soft_prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([soft_prompt, input_embeds], dim=1)
```

**4. RLHF (Reinforcement Learning from Human Feedback)**
```python
# Used by ChatGPT
# 1. Collect human preferences
# 2. Train reward model
# 3. Fine-tune with PPO

from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    batch_size=16,
    learning_rate=1.41e-5
)

ppo_trainer = PPOTrainer(
    model=model,
    config=ppo_config,
    tokenizer=tokenizer
)

# Training loop
for batch in dataset:
    query_tensors = batch['input_ids']
    
    # Generate responses
    response_tensors = ppo_trainer.generate(query_tensors)
    
    # Get rewards from reward model
    rewards = reward_model(query_tensors, response_tensors)
    
    # Update policy
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
```

---

## Slide 29: Evaluation Metrics

### Measuring Model Performance

**1. Perplexity**
```python
def calculate_perplexity(model, dataset):
    """
    Lower is better
    Perplexity = exp(average cross-entropy loss)
    """
    total_loss = 0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for batch in dataset:
            outputs = model(**batch, labels=batch['input_ids'])
            total_loss += outputs.loss.item() * batch['input_ids'].numel()
            total_tokens += batch['input_ids'].numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

ppl = calculate_perplexity(model, val_loader)
print(f"Perplexity: {ppl:.2f}")
```

**2. BLEU Score (for translation)**
```python
from nltk.translate.bleu_score import sentence_bleu

reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'test']

score = sentence_bleu(reference, candidate)
print(f"BLEU: {score:.4f}")
```

**3. Human Evaluation**
- Fluency
- Coherence
- Factual accuracy
- Helpfulness

**4. Automated Benchmarks**
```python
# Common benchmarks
benchmarks = {
    'MMLU': 'Multitask understanding',
    'HellaSwag': 'Commonsense reasoning',
    'TruthfulQA': 'Truthfulness',
    'HumanEval': 'Code generation',
    'GSM8K': 'Math problem solving'
}
```

---

## Slide 30: Hands-on Lab Preview

### Building a Mini Language Model

**Lab Objectives:**
1. Implement a small transformer from scratch
2. Train on a text corpus
3. Generate text
4. Experiment with hyperparameters

**Starter Code:**
```python
import torch
import torch.nn as nn
import urllib.request

# Mini GPT configuration
config = {
    'vocab_size': 10000,
    'n_embd': 256,
    'n_head': 4,
    'n_layer': 4,
    'block_size': 128,
    'dropout': 0.1
}

# Load data from URL
print("Downloading Shakespeare text...")
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

try:
    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8')
    print(f"✓ Downloaded {len(text):,} characters")
except Exception as e:
    print(f"Error downloading: {e}")
    print("Using sample text instead...")
    text = "This is sample text for demonstration. " * 100

# Build character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size} unique characters")

# Character-level tokenizer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Test tokenizer
sample = "Hello"
encoded = encode(sample)
decoded = decode(encoded)
print(f"\nTokenizer test:")
print(f"  Original: '{sample}'")
print(f"  Encoded: {encoded}")
print(f"  Decoded: '{decoded}'")

# Prepare dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"\nDataset prepared:")
print(f"  Total tokens: {len(data):,}")
print(f"  Training tokens: {len(train_data):,}")
print(f"  Validation tokens: {len(val_data):,}")

# TODO: Implement model, training, generation
# (Lab exercise)
```

**Lab Tasks:**
1. Complete the TransformerBlock implementation
2. Implement the training loop
3. Generate text with different temperatures
4. Experiment with model size
5. Compare results

---

**End of Batch 3 (Slides 21-30)**

*Continue to Batch 4 for Business Applications and Wrap-up*
