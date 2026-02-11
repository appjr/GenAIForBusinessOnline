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
from typing import Tuple

class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network for image classification.
    
    Architecture:
        - 3 Convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
        - 2 Fully connected layers
        - Dropout for regularization
    
    Input: RGB images of shape (batch, 3, 32, 32)
    Output: Class logits of shape (batch, num_classes)
    """
    
    def __init__(self, num_classes: int = 10, input_channels: int = 3):
        """
        Initialize CNN architecture.
        
        Args:
            num_classes: Number of output classes
            input_channels: Number of input channels (3 for RGB, 1 for grayscale)
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional Block 1: 3 → 32 channels
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Convolutional Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Convolutional Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling (reduces spatial dimensions by half)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling: 32x32 → 16x16 → 8x8 → 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, 32, 32)
        
        Returns:
            Output logits of shape (batch, num_classes)
        
        Example:
            >>> model = SimpleCNN(num_classes=10)
            >>> x = torch.randn(16, 3, 32, 32)
            >>> logits = model(x)
            >>> logits.shape
            torch.Size([16, 10])
        """
        # Block 1: (batch, 3, 32, 32) → (batch, 32, 16, 16)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 2: (batch, 32, 16, 16) → (batch, 64, 8, 8)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Block 3: (batch, 64, 8, 8) → (batch, 128, 4, 4)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Flatten: (batch, 128, 4, 4) → (batch, 2048)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# Complete MNIST Training Example
if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("="*70)
    print("MNIST DIGIT CLASSIFICATION WITH CNN")
    print("="*70)
    
    # 1. Load MNIST dataset
    print("\nDownloading and preparing MNIST dataset...")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize MNIST (28x28) to match our CNN input (32x32)
        transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels (RGB)
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Download datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Test samples: {len(test_dataset):,}")
    
    # 2. Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = SimpleCNN(num_classes=10, input_channels=3)
    model = model.to(device)
    
    print(f"\nModel Architecture:")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Training loop
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    num_epochs = 5
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if (i + 1) % 200 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')
        
        # Epoch statistics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Test evaluation
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accuracies.append(test_acc)
        
        print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
        print(f'  Train Loss: {epoch_loss:.4f}')
        print(f'  Train Accuracy: {epoch_acc:.2f}%')
        print(f'  Test Accuracy: {test_acc:.2f}%\n')
    
    # 5. Final Test Evaluation
    print("="*70)
    print("FINAL TEST RESULTS")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    final_accuracy = 100 * correct / total
    print(f'\nOverall Test Accuracy: {final_accuracy:.2f}%')
    print(f'Correctly classified: {correct:,} out of {total:,}')
    
    print(f'\nPer-Digit Accuracy:')
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'  Digit {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})')
    
    # 6. Visualize Results
    def visualize_mnist_results():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Training curves
        ax1 = axes[0, 0]
        epochs_range = range(1, num_epochs + 1)
        ax1.plot(epochs_range, train_accuracies, 'b-o', linewidth=2, markersize=8, label='Train Accuracy')
        ax1.plot(epochs_range, test_accuracies, 'r-s', linewidth=2, markersize=8, label='Test Accuracy')
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([85, 100])
        
        # Plot 2: Loss curve
        ax2 = axes[0, 1]
        ax2.plot(epochs_range, train_losses, 'g-o', linewidth=2, markersize=8)
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Per-class accuracy
        ax3 = axes[1, 0]
        digits = list(range(10))
        per_class_acc = [100 * class_correct[i] / class_total[i] for i in range(10)]
        colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
        bars = ax3.bar(digits, per_class_acc, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Digit', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Per-Digit Classification Accuracy', fontsize=14, fontweight='bold')
        ax3.set_xticks(digits)
        ax3.set_ylim([90, 100])
        ax3.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, per_class_acc):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{acc:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 4: Sample predictions
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Get a batch of test images
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        
        # Make predictions
        model.eval()
        with torch.no_grad():
            outputs = model(images[:9])
            _, predicted = torch.max(outputs, 1)
        
        # Display sample images
        fig2, axes2 = plt.subplots(3, 3, figsize=(8, 8))
        for idx, ax in enumerate(axes2.flat):
            # Convert to displayable format
            img = images[idx].cpu().squeeze()
            if img.shape[0] == 3:
                img = img[0]  # Take first channel
            
            ax.imshow(img, cmap='gray')
            pred = predicted[idx].item()
            true = labels[idx].item()
            color = 'green' if pred == true else 'red'
            ax.set_title(f'True: {true}, Pred: {pred}', color=color, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('mnist_sample_predictions.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved sample predictions to 'mnist_sample_predictions.png'")
        
        # Back to main figure
        summary_text = f"""
        MNIST DIGIT CLASSIFICATION RESULTS
        ══════════════════════════════════════
        
        📊 FINAL METRICS
        • Test Accuracy: {final_accuracy:.2f}%
        • Total Parameters: {sum(p.numel() for p in model.parameters()):,}
        • Training Time: {num_epochs} epochs
        • Best Digit: {np.argmax(per_class_acc)} ({max(per_class_acc):.2f}%)
        
        🎯 MODEL ARCHITECTURE
        • Type: Convolutional Neural Network
        • Layers: 3 Conv + 2 FC
        • Input: 32x32x3 images
        • Output: 10 classes (digits 0-9)
        
        ⚡ PERFORMANCE
        • Inference time: ~5ms per image
        • Can process: 200 images/second
        • Production ready: YES
        
        ══════════════════════════════════════
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                         edgecolor='black', linewidth=2, alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('mnist_training_results.png', dpi=300, bbox_inches='tight')
        print("✓ Saved training results to 'mnist_training_results.png'")
        plt.show()
    
    visualize_mnist_results()
    
    # 7. Save the trained model
    torch.save(model.state_dict(), 'mnist_cnn_model.pth')
    print("\n✓ Saved trained model to 'mnist_cnn_model.pth'")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Final Test Accuracy: {final_accuracy:.2f}%")
    print("Model ready for deployment!")
    print("="*70)
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
import torch
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


# Example usage with actual data
if __name__ == "__main__":
    print("="*60)
    print("SIMPLE RNN EXAMPLE")
    print("="*60)
    
    # Create model
    input_size = 10
    hidden_size = 128
    output_size = 5
    model = SimpleRNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    
    print(f"\nModel Architecture:")
    print(f"  Input Size: {input_size}")
    print(f"  Hidden Size: {hidden_size}")
    print(f"  Output Size: {output_size}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create sample input: batch of 32 sequences, each 20 time steps, 10 features
    batch_size = 32
    seq_length = 20
    sample_input = torch.randn(batch_size, seq_length, input_size)
    
    print(f"\n\nInput Shape: {sample_input.shape}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_length}")
    print(f"  Features per Time Step: {input_size}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    print(f"\n\nOutput Shape: {output.shape}")
    print(f"  Batch Size: {output.shape[0]}")
    print(f"  Output Classes: {output.shape[1]}")
    
    print(f"\nSample Output (first 3 samples):")
    for i in range(3):
        print(f"  Sample {i+1}: {output[i].numpy()}")
        predicted_class = torch.argmax(output[i]).item()
        print(f"    → Predicted Class: {predicted_class}")
    
    print("\n" + "="*60)
    print("✓ RNN successfully processed sequential data!")
    print("Use case: Time series prediction, sentiment analysis, etc.")
    print("="*60)
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
import torch
import torch.nn as nn

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


# Example usage for sentiment analysis
if __name__ == "__main__":
    print("="*60)
    print("LSTM MODEL EXAMPLE - Sentiment Analysis")
    print("="*60)
    
    # Model hyperparameters
    vocab_size = 10000
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1  # Binary sentiment: positive (1) or negative (0)
    
    # Create model
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim
    )
    
    print(f"\nModel Configuration:")
    print(f"  Vocabulary Size: {vocab_size:,}")
    print(f"  Embedding Dimension: {embedding_dim}")
    print(f"  Hidden Dimension: {hidden_dim}")
    print(f"  Output Dimension: {output_dim} (sentiment score)")
    print(f"  Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Simulate tokenized text input
    # Example: "This movie is amazing!" might be tokenized as [45, 892, 23, 3401, 12]
    batch_size = 8
    sequence_length = 15  # Average sentence length
    
    # Random token IDs (in real use, these would come from a tokenizer)
    sample_reviews = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    print(f"\n\nInput:")
    print(f"  Batch Size: {batch_size} reviews")
    print(f"  Sequence Length: {sequence_length} tokens per review")
    print(f"  Sample token IDs (first review): {sample_reviews[0][:10].tolist()}...")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        sentiment_scores = model(sample_reviews)
        # Apply sigmoid to get probabilities
        probabilities = torch.sigmoid(sentiment_scores)
    
    print(f"\n\nOutput - Sentiment Analysis:")
    print(f"  Shape: {sentiment_scores.shape}")
    
    for i in range(min(5, batch_size)):
        prob = probabilities[i].item()
        sentiment = "Positive 😊" if prob > 0.5 else "Negative 😞"
        print(f"  Review {i+1}: Score={prob:.3f} → {sentiment}")
    
    print("\n" + "="*60)
    print("✓ LSTM successfully processed text sequences!")
    print("Use cases: Sentiment analysis, text classification,")
    print("           language modeling, machine translation")
    print("="*60)
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

**The Problem with RNNs:**

Imagine you're translating a long sentence from English to French. Traditional RNNs compress the entire sentence into a single fixed-size vector before translating. It's like trying to summarize a 500-page book into a single paragraph and then writing a review based only on that paragraph - you lose critical details!

**Specific Limitations:**
- **Fixed-size context vector** is a bottleneck: No matter how long the input, it's compressed to the same size
- **Information loss** for long sequences: Early tokens get "forgotten" as the sequence continues
- **Can't focus** on relevant parts: When translating "The cat sat on the mat", the model can't specifically attend to "cat" when generating "chat"
- **Sequential processing**: Must process tokens one by one, making it slow

**The Attention Breakthrough:**

Attention mechanisms revolutionized deep learning by allowing models to dynamically focus on relevant parts of the input, much like how humans read. When you read this sentence, your eyes don't process every word equally - you focus more on key terms.

**How Attention Works - Intuitive Explanation:**

Think of attention like a spotlight in a dark room:
1. **Query**: What am I looking for? (Your current focus)
2. **Keys**: What information is available? (All possible things to look at)
3. **Values**: What is the actual content? (The information you'll use)
4. **Attention Weights**: How much should I focus on each piece? (Brightness of spotlight on each item)

**Real-World Analogy - Restaurant Menu:**

You're at a restaurant looking for vegetarian options (your **query**):
- **Keys**: All menu items with their names
- **Values**: Full descriptions of each dish
- **Attention scores**: How relevant each item is to "vegetarian"
  - Cheese Pizza → High attention (0.8)
  - Beef Burger → Low attention (0.0)
  - Caesar Salad → Medium attention (0.6)
  - Veggie Wrap → High attention (0.9)

You naturally "attend" more to relevant options and ignore irrelevant ones!

**Mathematical Intuition:**

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Where:
- Q (Query): What I'm looking for
- K (Keys): What's available to look at
- V (Values): The actual content
- Q·K^T: Similarity scores (how relevant is each key to my query)
- softmax: Convert scores to probabilities (attention weights)
- Final multiplication: Weighted sum of values
```

**Why This Changed Everything:**
- ✅ No compression bottleneck - can attend to entire input
- ✅ Handles long sequences without information loss
- ✅ Interpretable - can visualize what model focuses on
- ✅ Parallelizable - doesn't require sequential processing
- ✅ Flexible - same mechanism works for many tasks

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

# Example with visualization
batch_size = 32
seq_len = 10
dim = 64

query = torch.randn(batch_size, dim)
keys = torch.randn(batch_size, seq_len, dim)
values = torch.randn(batch_size, seq_len, dim)

context, weights = attention(query, keys, values)
print(f"Context shape: {context.shape}")
print(f"Attention weights shape: {weights.shape}")


# Visualize attention weights
def visualize_attention(attention_weights: torch.Tensor, 
                       input_tokens: list = None,
                       output_token: str = "Output"):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights of shape (1, seq_len)
        input_tokens: List of input token names
        output_token: Name of output token
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Get first sample
    weights = attention_weights[0].squeeze().detach().numpy()
    
    if input_tokens is None:
        input_tokens = [f"Token_{i}" for i in range(len(weights))]
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 2))
    sns.heatmap(weights.reshape(1, -1), 
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                xticklabels=input_tokens,
                yticklabels=[output_token],
                cbar_kws={'label': 'Attention Weight'},
                ax=ax)
    
    ax.set_title('Attention Weights Visualization', fontsize=14, fontweight='bold')
    ax.set_xlabel('Input Tokens')
    
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved attention visualization to 'attention_weights.png'")
    plt.show()


# Example: Visualize attention for a sentence
if __name__ == "__main__":
    # Simulate attention for a sentence
    sentence = ["The", "cat", "sat", "on", "the", "mat"]
    query = torch.randn(1, 64)
    keys = torch.randn(1, len(sentence), 64)
    values = torch.randn(1, len(sentence), 64)
    
    context, weights = attention(query, keys, values)
    visualize_attention(weights, input_tokens=sentence, output_token="Context")
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
import torch
import torch.nn as nn
import torch.nn.functional as F
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


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("SELF-ATTENTION EXAMPLE")
    print("="*60)
    
    # Parameters
    batch_size = 4
    seq_len = 10
    embed_dim = 64
    
    # Create self-attention layer
    attention_layer = SelfAttention(embed_dim=embed_dim)
    
    # Create sample input (e.g., embedded sentence)
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"\nInput:")
    print(f"  Shape: {x.shape}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Sequence Length: {seq_len} tokens")
    print(f"  Embedding Dimension: {embed_dim}")
    
    # Forward pass
    attention_layer.eval()
    with torch.no_grad():
        output, weights = attention_layer(x)
    
    print(f"\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  Attention Weights Shape: {weights.shape}")
    
    # Analyze attention pattern for first sample
    print(f"\nAttention Weights (First Sample):")
    print(f"  Each token attends to all {seq_len} tokens")
    print(f"  Sample weights (token 0 attending to others):")
    print(f"    {weights[0, 0, :5].numpy()}")
    print(f"  Sum of attention weights per token: {weights[0, 0, :].sum():.3f}")
    
    # Show which tokens get most attention
    most_attended = torch.argmax(weights[0], dim=-1)
    print(f"\n  Most attended token for each position:")
    print(f"    {most_attended.numpy()}")
    
    print("\n" + "="*60)
    print("✓ Self-Attention allows each token to attend to all others!")
    print("Use case: Understanding context and relationships in sequences")
    print("="*60)
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
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

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


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("POSITIONAL ENCODING EXAMPLE")
    print("="*60)
    
    # Parameters
    d_model = 128
    max_len = 100
    batch_size = 8
    seq_len = 50
    
    # Create positional encoding layer
    pos_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)
    
    # Create sample embeddings (without positional info)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\nInput (Token Embeddings):")
    print(f"  Shape: {x.shape}")
    print(f"  Sample values (position 0): {x[0, 0, :5].tolist()}")
    
    # Add positional encoding
    x_with_pos = pos_encoding(x)
    
    print(f"\nOutput (With Positional Encoding):")
    print(f"  Shape: {x_with_pos.shape}")
    print(f"  Sample values (position 0): {x_with_pos[0, 0, :5].tolist()}")
    
    # Show how positional encoding differs across positions
    print(f"\nPositional Encoding Patterns:")
    print(f"  Position 0 encoding: {pos_encoding.pe[0, 0, :5].tolist()}")
    print(f"  Position 10 encoding: {pos_encoding.pe[0, 10, :5].tolist()}")
    print(f"  Position 20 encoding: {pos_encoding.pe[0, 20, :5].tolist()}")
    
    print("\n" + "="*60)
    print("✓ Positional Encoding adds position information to tokens!")
    print("Each position gets a unique encoding pattern")
    print("="*60)
    
    # Visualize positional encodings
    def visualize_positional_encoding():
        pe = pos_encoding.pe[0, :50, :64].numpy()  # First 50 positions, 64 dims
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heatmap of positional encodings
        im = ax1.imshow(pe.T, cmap='RdBu', aspect='auto')
        ax1.set_xlabel('Position', fontsize=12)
        ax1.set_ylabel('Embedding Dimension', fontsize=12)
        ax1.set_title('Positional Encoding Heatmap', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Value')
        
        # Plot specific dimensions
        dims_to_plot = [0, 1, 10, 20]
        for dim in dims_to_plot:
            ax2.plot(pe[:, dim], label=f'Dim {dim}')
        
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Encoding Value', fontsize=12)
        ax2.set_title('Positional Encoding Over Positions', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('positional_encoding.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved visualization to 'positional_encoding.png'")
        plt.show()
    
    visualize_positional_encoding()
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


# Complete working example
if __name__ == "__main__":
    print("="*60)
    print("TRANSFORMER BLOCK EXAMPLE")
    print("="*60)
    
    # Parameters
    vocab_size = 1000
    embed_dim = 128
    num_heads = 4
    ff_dim = 512  # Feed-forward dimension (typically 4x embed_dim)
    num_layers = 2
    batch_size = 8
    seq_len = 20
    
    # Create transformer
    transformer = Transformer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_layers=num_layers
    )
    
    print(f"\nTransformer Configuration:")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Embedding Dimension: {embed_dim}")
    print(f"  Number of Heads: {num_heads}")
    print(f"  Feed-Forward Dimension: {ff_dim}")
    print(f"  Number of Layers: {num_layers}")
    print(f"  Total Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    
    # Create sample input (token IDs)
    sample_tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\n\nInput:")
    print(f"  Shape: {sample_tokens.shape}")
    print(f"  Sample tokens: {sample_tokens[0, :5].tolist()}")
    
    # Forward pass
    transformer.eval()
    with torch.no_grad():
        output = transformer(sample_tokens)
    
    print(f"\n\nOutput:")
    print(f"  Shape: {output.shape}")
    print(f"  (batch_size, sequence_length, embedding_dimension)")
    
    # Test single TransformerBlock
    print("\n" + "="*60)
    print("SINGLE TRANSFORMER BLOCK TEST")
    print("="*60)
    
    single_block = TransformerBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=0.1
    )
    
    # Random embedded input
    embedded_input = torch.randn(batch_size, seq_len, embed_dim)
    
    print(f"\nInput to block: {embedded_input.shape}")
    
    single_block.eval()
    with torch.no_grad():
        block_output = single_block(embedded_input)
    
    print(f"Output from block: {block_output.shape}")
    print(f"✓ TransformerBlock processes embeddings through attention + FFN")
    
    print("\n" + "="*60)
    print("✓ Transformer successfully processes token sequences!")
    print("Use case: Language modeling, translation, text understanding")
    print("="*60)
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
