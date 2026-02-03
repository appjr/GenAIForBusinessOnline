# Week 4: Deep Learning and the GenAI General Algorithm - Slides Batch 1 (Slides 1-10)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 11, 2026  
**Duration:** 2.5 hours

---

## Slide 1: Week 4 Title Slide

### Deep Learning and the GenAI General Algorithm

**Today's Focus:**
- Understanding neural networks from the ground up
- Exploring deep learning architectures
- Demystifying how GenAI models actually work
- Transformer architecture and attention mechanisms
- Building intuition for modern AI systems

**Prerequisites Recap:**
- Basic Python programming
- Linear algebra fundamentals
- Understanding of ML concepts from Week 3

---

## Slide 2: Today's Agenda

### Class Overview

1. **Neural Networks Fundamentals** (30 min)
2. **Deep Learning Architectures** (25 min)
3. **Break** (10 min)
4. **Attention Mechanisms** (30 min)
5. **Transformer Architecture** (30 min)
6. **GenAI Algorithm Deep Dive** (20 min)
7. **Hands-on Lab & Q&A** (25 min)

---

## Slide 3: Learning Objectives

### By the End of This Class, You Will:

✅ **Understand** how neural networks learn and make predictions  
✅ **Explain** the architecture of deep learning models  
✅ **Comprehend** attention mechanisms and their importance  
✅ **Describe** how transformers revolutionized AI  
✅ **Recognize** the components of GenAI algorithms  
✅ **Apply** concepts to real-world business scenarios

---

## Slide 4: What is a Neural Network?

### Inspired by the Human Brain

**Biological Neuron:**
- Receives signals from dendrites
- Processes in the cell body
- Sends output through axon
- Connects to other neurons

**Artificial Neuron:**
- Receives inputs (x₁, x₂, ..., xₙ)
- Applies weights (w₁, w₂, ..., wₙ)
- Sums weighted inputs + bias
- Applies activation function
- Produces output

**Mathematical Representation:**
```
Output = Activation(Σ(wᵢ × xᵢ) + bias)
```

**Simple Example:**
```python
import numpy as np

def neuron(inputs, weights, bias):
    # Weighted sum
    z = np.dot(inputs, weights) + bias
    
    # Activation function (sigmoid)
    output = 1 / (1 + np.exp(-z))
    
    return output

# Example
inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([0.5, -0.3, 0.8])
bias = 0.1

result = neuron(inputs, weights, bias)
print(f"Neuron output: {result:.4f}")
```

---

## Slide 5: Activation Functions

### Adding Non-Linearity to Networks

**Why Activation Functions?**
- Without them, networks would just be linear combinations
- Enable learning complex, non-linear patterns
- Introduce decision boundaries

**Common Activation Functions:**

**1. Sigmoid (Logistic)**
```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Range: (0, 1)
# Use: Output layer for binary classification
```

**2. ReLU (Rectified Linear Unit)**
```python
def relu(x):
    return np.maximum(0, x)

# Range: [0, ∞)
# Use: Hidden layers (most popular)
# Advantage: Fast, avoids vanishing gradient
```

**3. Tanh (Hyperbolic Tangent)**
```python
def tanh(x):
    return np.tanh(x)

# Range: (-1, 1)
# Use: Hidden layers, better than sigmoid
```

**4. Softmax (for multi-class)**
```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Range: (0, 1), sum = 1
# Use: Output layer for multi-class classification
```

---

## Slide 6: Building a Simple Neural Network

### From Neurons to Networks

**Network Architecture:**
```
Input Layer (3 nodes)
    ↓
Hidden Layer (4 nodes)
    ↓
Output Layer (2 nodes)
```

**Implementation:**
```python
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Layer 2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        
        return self.a2
    
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# Usage
network = SimpleNeuralNetwork(input_size=3, hidden_size=4, output_size=2)
X = np.array([[1.0, 2.0, 3.0]])
predictions = network.forward(X)
print(f"Predictions: {predictions}")
```

---

## Slide 7: How Neural Networks Learn

### The Training Process

**4-Step Learning Cycle:**

**1. Forward Propagation**
- Input flows through network
- Each layer transforms data
- Final layer produces prediction

**2. Loss Calculation**
- Compare prediction to actual label
- Calculate error (loss)
- Common loss functions:
  - Binary Cross-Entropy (binary classification)
  - Categorical Cross-Entropy (multi-class)
  - Mean Squared Error (regression)

**3. Backward Propagation**
- Calculate gradients of loss w.r.t. weights
- Use chain rule to propagate error backwards
- Determine how much each weight contributed to error

**4. Weight Update**
- Adjust weights to reduce loss
- Use optimization algorithm (e.g., SGD, Adam)
- Learning rate controls step size

**Training Loop:**
```python
def train(network, X_train, y_train, epochs=100, learning_rate=0.01):
    for epoch in range(epochs):
        # Forward pass
        predictions = network.forward(X_train)
        
        # Calculate loss
        loss = -np.mean(y_train * np.log(predictions + 1e-8))
        
        # Backward pass (compute gradients)
        # ... gradient calculations ...
        
        # Update weights
        network.W1 -= learning_rate * dW1
        network.W2 -= learning_rate * dW2
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

---

## Slide 8: Gradient Descent Visualization

### Optimizing the Loss Function

**Concept:**
- Loss function is like a hilly landscape
- Weights are your position
- Goal: Find the lowest point (minimum loss)
- Gradient points "uphill"
- Move opposite to gradient (downhill)

**Gradient Descent Variants:**

**1. Batch Gradient Descent**
```python
# Use all training data
for epoch in range(epochs):
    gradients = compute_gradients(X_train, y_train)
    weights -= learning_rate * gradients
```

**2. Stochastic Gradient Descent (SGD)**
```python
# Use one sample at a time
for epoch in range(epochs):
    for i in range(len(X_train)):
        gradients = compute_gradients(X_train[i], y_train[i])
        weights -= learning_rate * gradients
```

**3. Mini-Batch Gradient Descent** (Most Common)
```python
# Use small batches (best of both)
batch_size = 32
for epoch in range(epochs):
    for batch in get_batches(X_train, y_train, batch_size):
        gradients = compute_gradients(batch)
        weights -= learning_rate * gradients
```

**Learning Rate Impact:**
- **Too high**: Overshooting, oscillation, divergence
- **Too low**: Slow convergence, stuck in local minima
- **Just right**: Smooth, efficient convergence

---

## Slide 9: Deep Neural Networks

### Going Deeper

**What Makes a Network "Deep"?**
- Multiple hidden layers (typically 3+)
- Each layer learns increasingly abstract features
- Early layers: Simple patterns (edges, colors)
- Middle layers: Combinations of patterns (shapes, textures)
- Deep layers: High-level concepts (objects, faces)

**Example Architecture:**
```python
import torch
import torch.nn as nn

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(DeepNeuralNetwork, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # Regularization
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Create a deep network
model = DeepNeuralNetwork(
    input_size=100,
    hidden_sizes=[512, 256, 128, 64],  # 4 hidden layers
    output_size=10
)

print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
```

---

## Slide 10: Challenges in Deep Learning

### Common Issues and Solutions

**1. Vanishing Gradient Problem**
- **Problem**: Gradients become very small in early layers
- **Why**: Chain rule multiplication of small numbers
- **Solutions**: 
  - ReLU activation (instead of sigmoid)
  - Batch normalization
  - Residual connections (ResNet)

**2. Overfitting**
- **Problem**: Model memorizes training data, poor generalization
- **Solutions**:
  ```python
  # Dropout
  nn.Dropout(0.5)
  
  # L2 Regularization
  optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01)
  
  # Early stopping
  if val_loss > best_val_loss:
      patience_counter += 1
      if patience_counter > patience:
          break
  ```

**3. Exploding Gradient**
- **Problem**: Gradients become very large
- **Solutions**:
  - Gradient clipping
  - Careful weight initialization
  - Batch normalization

**4. Slow Training**
- **Solutions**:
  - Use GPUs/TPUs
  - Batch normalization
  - Better optimizers (Adam, AdamW)
  - Learning rate scheduling

**Best Practices:**
```python
# Modern deep learning setup
model = DeepNeuralNetwork(...)

# Good optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

# Training with best practices
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    scheduler.step(val_loss)
    
    # Save best model
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'best_model.pth')
```

---

**End of Batch 1 (Slides 1-10)**

*Continue to Batch 2 for Deep Learning Architectures and CNNs*
