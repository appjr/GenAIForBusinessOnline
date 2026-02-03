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
class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head) 
            for _ in range(n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
    
    def forward(self, idx):
        B, T = idx.shape
        
        # Token + position embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Generate logits
        logits = self.lm_head(x)
        
        return logits

# GPT-2 Small config
model = GPTModel(
    vocab_size=50257,
    n_embd=768,
    n_head=12,
    n_layer=12,
    block_size=1024
)
print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
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
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Generate text autoregressively
    
    idx: (B, T) current context
    max_new_tokens: how many tokens to generate
    temperature: randomness (higher = more random)
    top_k: sample from top k tokens only
    """
    
    for _ in range(max_new_tokens):
        # Crop context if needed
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        
        # Get predictions
        logits = model(idx_cond)
        logits = logits[:, -1, :]  # Get last token
        
        # Apply temperature
        logits = logits / temperature
        
        # Optionally crop logits to top k
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx

# Usage
prompt = "The future of AI is"
tokens = tokenizer.encode(prompt)
idx = torch.tensor([tokens])
generated = generate(model, idx, max_new_tokens=50)
text = tokenizer.decode(generated[0].tolist())
print(text)
```

---

## Slide 23: Training Language Models

### Pre-training and Fine-tuning

**Pre-training Objective:**
- Next token prediction (language modeling)
- Learn from massive unlabeled data
- Captures language patterns and knowledge

**Pre-training Loss:**
```python
def compute_loss(model, batch):
    # batch shape: (B, T+1)
    # Split into input and target
    x = batch[:, :-1]  # Input: first T tokens
    y = batch[:, 1:]   # Target: next T tokens
    
    # Forward pass
    logits = model(x)  # (B, T, vocab_size)
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        y.view(-1)
    )
    
    return loss

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = compute_loss(model, batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
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
- Capabilities not present in smaller models
- Suddenly appear at certain scales
- Not explicitly trained for

**Examples:**

**1. Few-Shot Learning**
```python
prompt = """
Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>"""

# Model completes: "fromage"
```

**2. Chain-of-Thought Reasoning**
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

# Model generates step-by-step reasoning
```

**3. In-Context Learning**
- Learning from examples in the prompt
- No gradient updates needed
- Adapts to new tasks on-the-fly

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

# Mini GPT configuration
config = {
    'vocab_size': 10000,
    'n_embd': 256,
    'n_head': 4,
    'n_layer': 4,
    'block_size': 128,
    'dropout': 0.1
}

# Load data
text = open('shakespeare.txt', 'r').read()
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character-level tokenizer
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Prepare dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

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
