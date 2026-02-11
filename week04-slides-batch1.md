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
from typing import Union

def neuron(inputs: np.ndarray, weights: np.ndarray, bias: float) -> float:
    """
    Simulates a single artificial neuron with sigmoid activation.
    
    Args:
        inputs: Input feature vector of shape (n,)
        weights: Weight vector of shape (n,)
        bias: Scalar bias term
    
    Returns:
        Activation output between 0 and 1
    
    Example:
        >>> inputs = np.array([1.0, 2.0, 3.0])
        >>> weights = np.array([0.5, -0.3, 0.8])
        >>> bias = 0.1
        >>> output = neuron(inputs, weights, bias)
        >>> print(f"Neuron output: {output:.4f}")
        Neuron output: 0.9866
    """
    # Weighted sum: z = w₁x₁ + w₂x₂ + ... + wₙxₙ + bias
    z = np.dot(inputs, weights) + bias
    
    # Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
    # Maps any value to range (0, 1)
    output = 1 / (1 + np.exp(-z))
    
    return output

# Example usage
if __name__ == "__main__":
    # Create sample inputs
    inputs = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, -0.3, 0.8])
    bias = 0.1
    
    # Calculate neuron output
    result = neuron(inputs, weights, bias)
    print(f"Neuron output: {result:.4f}")
    
    # Interpretation: High output (close to 1) means strong activation
```

---

## Slide 5: Activation Functions

### Adding Non-Linearity to Networks

**Why Activation Functions?**
- Without them, networks would just be linear combinations
- Enable learning complex, non-linear patterns
- Introduce decision boundaries

**Common Activation Functions:**

```python
import numpy as np
from typing import Union
import matplotlib.pyplot as plt

def sigmoid(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Sigmoid (Logistic) activation function.
    
    Formula: σ(x) = 1 / (1 + e^(-x))
    
    Args:
        x: Input value(s)
    
    Returns:
        Output in range (0, 1)
    
    Use case: Output layer for binary classification
    Properties:
        - Smooth gradient
        - Output interpretable as probability
        - Suffers from vanishing gradient problem
    """
    return 1 / (1 + np.exp(-x))


def relu(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Formula: f(x) = max(0, x)
    
    Args:
        x: Input value(s)
    
    Returns:
        Output in range [0, ∞)
    
    Use case: Hidden layers (most popular choice)
    Advantages:
        - Computationally efficient
        - Avoids vanishing gradient problem
        - Sparse activation (many zeros)
    
    Disadvantage:
        - "Dead ReLU" problem when x < 0
    """
    return np.maximum(0, x)


def tanh(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Tanh (Hyperbolic Tangent) activation function.
    
    Formula: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
    
    Args:
        x: Input value(s)
    
    Returns:
        Output in range (-1, 1)
    
    Use case: Hidden layers, preferred over sigmoid
    Advantages:
        - Zero-centered output
        - Stronger gradients than sigmoid
    """
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation function for multi-class classification.
    
    Formula: softmax(xᵢ) = e^xᵢ / Σⱼ e^xⱼ
    
    Args:
        x: Input vector or matrix (last dimension is classes)
    
    Returns:
        Probability distribution over classes (sums to 1)
    
    Use case: Output layer for multi-class classification
    Properties:
        - Outputs interpretable as probabilities
        - Numerically stable with max subtraction
    
    Example:
        >>> logits = np.array([2.0, 1.0, 0.1])
        >>> probs = softmax(logits)
        >>> print(probs)
        [0.659 0.242 0.099]
        >>> print(probs.sum())
        1.0
    """
    # Subtract max for numerical stability (prevents overflow)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Demonstration and visualization of activation functions
if __name__ == "__main__":
    x = np.linspace(-5, 5, 100)
    
    print("Activation Function Examples:")
    print(f"sigmoid(0) = {sigmoid(0):.4f}")
    print(f"relu(-2) = {relu(-2):.4f}")
    print(f"tanh(0) = {tanh(0):.4f}")
    
    logits = np.array([2.0, 1.0, 0.1])
    probs = softmax(logits)
    print(f"softmax([2.0, 1.0, 0.1]) = {probs}")
    
    # Visualize activation functions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_title('Sigmoid Activation', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Input')
    axes[0, 0].set_ylabel('Output')
    axes[0, 0].text(0.05, 0.95, 'Range: (0, 1)\nUse: Binary classification',
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ReLU
    axes[0, 1].plot(x, relu(x), 'r-', linewidth=2)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title('ReLU Activation', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Input')
    axes[0, 1].set_ylabel('Output')
    axes[0, 1].text(0.05, 0.95, 'Range: [0, ∞)\nUse: Hidden layers (most common)',
                    transform=axes[0, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Tanh
    axes[1, 0].plot(x, tanh(x), 'g-', linewidth=2)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axhline(y=-1, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_title('Tanh Activation', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Input')
    axes[1, 0].set_ylabel('Output')
    axes[1, 0].text(0.05, 0.95, 'Range: (-1, 1)\nUse: Hidden layers, zero-centered',
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Softmax visualization
    classes = ['Class A', 'Class B', 'Class C']
    logits_viz = np.array([2.0, 1.0, 0.1])
    probs_viz = softmax(logits_viz)
    
    axes[1, 1].bar(classes, probs_viz, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1, 1].set_title('Softmax Output', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('Probability')
    axes[1, 1].set_ylim([0, 1])
    for i, (c, p) in enumerate(zip(classes, probs_viz)):
        axes[1, 1].text(i, p + 0.02, f'{p:.3f}', ha='center', fontweight='bold')
    axes[1, 1].text(0.5, 0.5, f'Sum = {probs_viz.sum():.3f}',
                    transform=axes[1, 1].transAxes, ha='center',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved visualization to 'activation_functions.png'")
    plt.show()
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
from typing import Tuple

class SimpleNeuralNetwork:
    """
    A basic 2-layer neural network for classification.
    
    Architecture:
        Input → Hidden (ReLU) → Output (Softmax)
    
    Attributes:
        W1: Weight matrix for hidden layer (input_size × hidden_size)
        b1: Bias vector for hidden layer (1 × hidden_size)
        W2: Weight matrix for output layer (hidden_size × output_size)
        b2: Bias vector for output layer (1 × output_size)
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize network with random weights.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output classes
        """
        # Initialize weights with small random values (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        # Store activations for backpropagation
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation: max(0, x)"""
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax activation for multi-class classification.
        Numerically stable version with max subtraction.
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through the network.
        
        Args:
            X: Input data of shape (batch_size, input_size)
        
        Returns:
            Probability distribution over classes (batch_size, output_size)
        
        Example:
            >>> X = np.array([[1.0, 2.0, 3.0]])  # 1 sample, 3 features
            >>> probs = network.forward(X)
            >>> probs.shape
            (1, 2)
        """
        # Layer 1: Input → Hidden
        self.z1 = np.dot(X, self.W1) + self.b1  # (batch, hidden_size)
        self.a1 = self.relu(self.z1)             # (batch, hidden_size)
        
        # Layer 2: Hidden → Output
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (batch, output_size)
        self.a2 = self.softmax(self.z2)                # (batch, output_size)
        
        return self.a2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Args:
            X: Input data of shape (batch_size, input_size)
        
        Returns:
            Predicted class indices (batch_size,)
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


# Example usage
if __name__ == "__main__":
    # Create network
    network = SimpleNeuralNetwork(input_size=3, hidden_size=4, output_size=2)
    
    # Single sample prediction
    X = np.array([[1.0, 2.0, 3.0]])  # Shape: (1, 3)
    predictions = network.forward(X)
    print(f"Output probabilities: {predictions}")
    print(f"Predicted class: {network.predict(X)[0]}")
    
    # Batch prediction
    X_batch = np.array([
        [1.0, 2.0, 3.0],
        [0.5, 1.5, 2.5],
        [2.0, 3.0, 4.0]
    ])  # Shape: (3, 3)
    batch_predictions = network.forward(X_batch)
    print(f"\nBatch predictions:\n{batch_predictions}")
    print(f"Predicted classes: {network.predict(X_batch)}")
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

**Complete Training Implementation:**
```python
import numpy as np
from typing import Tuple

def train(network: SimpleNeuralNetwork, X_train: np.ndarray, y_train: np.ndarray, 
          epochs: int = 100, learning_rate: float = 0.01) -> list:
    """
    Train neural network using backpropagation.
    
    Args:
        network: SimpleNeuralNetwork instance
        X_train: Training data of shape (n_samples, n_features)
        y_train: One-hot encoded labels of shape (n_samples, n_classes)
        epochs: Number of training iterations
        learning_rate: Step size for gradient descent
    
    Returns:
        List of loss values per epoch
    
    Example:
        >>> network = SimpleNeuralNetwork(3, 4, 2)
        >>> X = np.random.randn(100, 3)
        >>> y = np.eye(2)[np.random.randint(0, 2, 100)]
        >>> losses = train(network, X, y, epochs=50)
    """
    losses = []
    n_samples = X_train.shape[0]
    
    for epoch in range(epochs):
        # 1. Forward pass
        predictions = network.forward(X_train)
        
        # 2. Calculate loss (Cross-Entropy)
        loss = -np.mean(y_train * np.log(predictions + 1e-8))
        losses.append(loss)
        
        # 3. Backward pass (compute gradients using chain rule)
        
        # Output layer gradient
        # dL/dz2 = predictions - y_train (for softmax + cross-entropy)
        dz2 = predictions - y_train  # Shape: (n_samples, output_size)
        
        # Gradient for W2 and b2
        dW2 = np.dot(network.a1.T, dz2) / n_samples  # (hidden_size, output_size)
        db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples  # (1, output_size)
        
        # Hidden layer gradient
        # Backpropagate through hidden layer
        dz1 = np.dot(dz2, network.W2.T)  # (n_samples, hidden_size)
        
        # Apply ReLU derivative (1 if z1 > 0, else 0)
        dz1[network.z1 <= 0] = 0
        
        # Gradient for W1 and b1
        dW1 = np.dot(X_train.T, dz1) / n_samples  # (input_size, hidden_size)
        db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples  # (1, hidden_size)
        
        # 4. Update weights (Gradient Descent)
        network.W1 -= learning_rate * dW1
        network.b1 -= learning_rate * db1
        network.W2 -= learning_rate * dW2
        network.b2 -= learning_rate * db2
        
        # Logging
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")
    
    return losses


# Example usage with synthetic data
if __name__ == "__main__":
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 200
    n_features = 3
    n_classes = 2
    
    # Generate random data
    X_train = np.random.randn(n_samples, n_features)
    y_labels = np.random.randint(0, n_classes, n_samples)
    y_train = np.eye(n_classes)[y_labels]  # One-hot encode
    
    # Initialize network
    network = SimpleNeuralNetwork(
        input_size=n_features,
        hidden_size=4,
        output_size=n_classes
    )
    
    # Train network
    print("Training neural network...")
    losses = train(network, X_train, y_train, epochs=100, learning_rate=0.1)
    
    # Evaluate
    predictions = network.predict(X_train)
    accuracy = np.mean(predictions == y_labels)
    print(f"\nFinal Training Accuracy: {accuracy:.2%}")
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
