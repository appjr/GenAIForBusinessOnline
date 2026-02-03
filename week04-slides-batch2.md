# Week 4: Deep Learning and the GenAI General Algorithm - Slides Batch 2 (Slides 11-20)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 11, 2026

---

## Slide 11: Convolutional Neural Networks (CNNs)

### Specialized for Image Data

**Why CNNs?**
- Traditional neural networks don't preserve spatial structure
- Too many parameters for images
- CNNs use local connectivity and parameter sharing

**Key Components:**

**1. Convolutional Layer**
```python
import torch.nn as nn

# 2D Convolution
conv_layer = nn.Conv2d(
    in_channels=3,      # RGB image
    out_channels=32,    # 32 filters
    kernel_size=3,      # 3x3 filter
    stride=1,
    padding=1
)

# How it works:
# - Slide filter across image
# - Compute dot product at each position
# - Creates feature map
```

**2. Pooling Layer**
```python
# Max pooling (most common)
pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)

# Reduces spatial dimensions by half
# Provides translation invariance
```

**3. Fully Connected Layer**
```python
# Final classification
fc = nn.Linear(in_features=512, out_features=10)
```

---

## Slide 12: Building a CNN

### Image Classification Example

**Complete CNN Architecture:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Create model
model = SimpleCNN(num_classes=10)
print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
```

---

## Slide 13: Recurrent Neural Networks (RNNs)

### For Sequential Data

**Why RNNs?**
- Handle variable-length sequences
- Maintain memory of previous inputs
- Used for text, time series, speech

**RNN Structure:**
```
Input: x₁ → x₂ → x₃ → x₄
        ↓     ↓     ↓     ↓
Hidden: h₁ → h₂ → h₃ → h₄
        ↓     ↓     ↓     ↓
Output: y₁   y₂   y₃   y₄
```

**Basic RNN Implementation:**
```python
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # RNN layer
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch, sequence_length, input_size)
        
        # RNN forward pass
        out, hidden = self.rnn(x)
        
        # Use last output
        out = self.fc(out[:, -1, :])
        
        return out

# Example usage
model = SimpleRNN(input_size=10, hidden_size=128, output_size=5)
```

---

## Slide 14: LSTM and GRU

### Solving the Vanishing Gradient Problem

**Problem with Basic RNNs:**
- Struggle with long sequences
- Vanishing gradients
- Can't remember distant past

**LSTM (Long Short-Term Memory):**
```python
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=2,
            dropout=0.5,
            batch_first=True
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, text):
        # Embed words
        embedded = self.embedding(text)
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Use final hidden state
        return self.fc(hidden[-1])

model = LSTMModel(
    vocab_size=10000,
    embedding_dim=100,
    hidden_dim=256,
    output_dim=1
)
```

**GRU (Gated Recurrent Unit):**
```python
# Simpler than LSTM, often works just as well
self.gru = nn.GRU(
    input_size=embedding_dim,
    hidden_size=hidden_dim,
    num_layers=2,
    dropout=0.5,
    batch_first=True
)
```

---

## Slide 15: Introduction to Attention

### The Key Innovation for GenAI

**Problem with RNNs:**
- Fixed-size context vector is a bottleneck
- Information loss for long sequences
- Can't focus on relevant parts

**Attention Mechanism:**
- Model decides which parts of input are important
- Creates weighted representation
- Different for each output step

**Simple Attention Example:**
```python
import torch
import torch.nn.functional as F

def attention(query, keys, values):
    """
    query: (batch, query_dim)
    keys: (batch, seq_len, key_dim)
    values: (batch, seq_len, value_dim)
    """
    
    # Calculate attention scores
    scores = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2))
    # Shape: (batch, 1, seq_len)
    
    # Apply softmax to get weights
    attention_weights = F.softmax(scores, dim=-1)
    
    # Apply weights to values
    context = torch.matmul(attention_weights, values)
    # Shape: (batch, 1, value_dim)
    
    return context.squeeze(1), attention_weights

# Example
batch_size = 32
seq_len = 10
dim = 64

query = torch.randn(batch_size, dim)
keys = torch.randn(batch_size, seq_len, dim)
values = torch.randn(batch_size, seq_len, dim)

context, weights = attention(query, keys, values)
print(f"Context shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")
```

---

## Slide 16: Self-Attention

### Looking at the Entire Sequence

**Self-Attention Concept:**
- Each word attends to all other words
- Discovers relationships within the sequence
- Parallel computation (unlike RNNs)

**Self-Attention Implementation:**
```python
import torch.nn as nn
import math

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # x shape: (batch, seq_len, embed_dim)
        
        # Generate Q, K, V
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.embed_dim)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

# Usage
attention_layer = SelfAttention(embed_dim=512)
x = torch.randn(32, 10, 512)  # batch, seq_len, embed_dim
output, weights = attention_layer(x)
```

---

## Slide 17: Multi-Head Attention

### Multiple Perspectives

**Why Multiple Heads?**
- Learn different types of relationships
- Capture various aspects of the data
- More expressive representation

**Multi-Head Attention:**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear layers for Q, K, V
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V for all heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # (batch, heads, seq_len, 3*head_dim)
        
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        out = torch.matmul(attention, v)
        
        # Concatenate heads
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_dim)
        
        # Final linear layer
        out = self.fc_out(out)
        
        return out

# Example
mha = MultiHeadAttention(embed_dim=512, num_heads=8)
x = torch.randn(32, 100, 512)
output = mha(x)
print(f"Output shape: {output.shape}")
```

---

## Slide 18: Positional Encoding

### Adding Position Information

**Problem:**
- Attention has no notion of order
- "I love cats" vs "Cats love I"
- Need to inject position information

**Positional Encoding:**
```python
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1)]
        return x

# Usage
pos_encoding = PositionalEncoding(d_model=512)
x = torch.randn(32, 100, 512)
x_with_pos = pos_encoding(x)
```

---

## Slide 19: The Transformer Block

### Building Block of Modern AI

**Transformer Layer Components:**

**1. Multi-Head Attention**
**2. Add & Norm (Residual Connection + Layer Normalization)**
**3. Feed-Forward Network**
**4. Add & Norm**

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Multi-head attention with residual
        attention_out = self.attention(x)
        x = self.ln1(x + self.dropout(attention_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        
        return x

# Stack multiple blocks
class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers):
        super(Transformer, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)
        
        # Stack transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x
```

---

## Slide 20: Transformer Architecture Overview

### The Full Picture

**Encoder-Decoder Architecture:**

```
Encoder (Left Side):
    Input Embedding
    + Positional Encoding
    ↓
    [Multi-Head Attention]
    ↓
    [Feed Forward]
    ↓
    (Repeat N times)
    
Decoder (Right Side):
    Output Embedding
    + Positional Encoding
    ↓
    [Masked Multi-Head Attention]
    ↓
    [Cross-Attention to Encoder]
    ↓
    [Feed Forward]
    ↓
    (Repeat N times)
    ↓
    Linear + Softmax
```

**Key Differences:**

**Encoder:**
- Processes input sequence
- Bidirectional attention
- Creates rich representations

**Decoder:**
- Generates output sequence
- Masked attention (can't see future)
- Attends to encoder outputs
- Autoregressive generation

**Applications:**
- **Translation**: Encoder-Decoder
- **Text Generation**: Decoder-only (GPT)
- **Understanding**: Encoder-only (BERT)

**Why Transformers Won:**
1. Parallel computation (vs RNN sequential)
2. Better at long-range dependencies
3. Scalable to massive datasets
4. Transfer learning capabilities

---

**End of Batch 2 (Slides 11-20)**

*Continue to Batch 3 for GenAI Algorithms and Training*
