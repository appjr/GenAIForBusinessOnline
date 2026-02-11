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

**The Biological Inspiration:**

The human brain contains approximately 86 billion neurons, each connected to thousands of other neurons. When you recognize a face, understand speech, or make a decision, billions of neurons are firing in coordinated patterns. Artificial neural networks attempt to capture this parallel, distributed processing power in mathematical form.

**Biological Neuron:**
- **Dendrites**: Receive electrical signals from other neurons
- **Cell Body (Soma)**: Integrates incoming signals and decides whether to fire
- **Axon**: Transmits electrical impulse to other neurons if threshold is reached
- **Synapses**: Connection points where signals pass between neurons with varying strengths

**Artificial Neuron (Perceptron):**
- **Receives inputs** (x₁, x₂, ..., xₙ): Multiple numerical features (e.g., pixels, measurements)
- **Applies weights** (w₁, w₂, ..., wₙ): Each connection has a strength that amplifies or diminishes the signal
- **Sums weighted inputs + bias**: Computes z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
- **Applies activation function**: Non-linear transformation determines if neuron "fires"
- **Produces output**: Single number that becomes input for next layer

**Mathematical Representation:**
```
Output = Activation(Σ(wᵢ × xᵢ) + bias)

Where:
- xᵢ = input features
- wᵢ = learned weights (importance of each feature)
- bias = learned offset (shifts decision boundary)
- Activation = non-linear function (enables complex patterns)
```

**Intuitive Example - Email Spam Detection:**

Think of a neuron deciding if an email is spam:
- **Input x₁**: Number of exclamation marks (value: 5)
- **Input x₂**: Presence of word "FREE" (value: 1 for yes, 0 for no)
- **Input x₃**: Sender in contacts (value: 0 for no, 1 for yes)

If the neuron has learned:
- **Weight w₁ = 0.3**: Exclamation marks slightly indicate spam
- **Weight w₂ = 0.8**: "FREE" strongly indicates spam
- **Weight w₃ = -0.9**: Known sender strongly indicates NOT spam
- **Bias b = 0.1**: Slight tendency toward not spam

Calculation:
```
z = (5 × 0.3) + (1 × 0.8) + (0 × -0.9) + 0.1
z = 1.5 + 0.8 + 0 + 0.1 = 2.4

Output = sigmoid(2.4) = 0.917 (91.7% confidence it's spam)
```

**Key Insight:** The weights and bias are what the network "learns" during training. Initially random, they're adjusted through thousands of examples until the neuron makes good predictions.

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


# Complete IRIS Dataset Classification Example
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("IRIS FLOWER CLASSIFICATION WITH NEURAL NETWORK")
    print("="*70)
    
    # 1. Load IRIS dataset
    print("\nLoading IRIS dataset...")
    iris = load_iris()
    X = iris.data  # 4 features: sepal length, sepal width, petal length, petal width
    y_labels = iris.target  # 3 classes: setosa (0), versicolor (1), virginica (2)
    
    print(f"Dataset size: {X.shape[0]} samples")
    print(f"Features: {X.shape[1]} ({iris.feature_names})")
    print(f"Classes: {len(iris.target_names)} ({iris.target_names})")
    print(f"Class distribution: {np.bincount(y_labels)}")
    
    # 2. Split and normalize data
    X_train, X_test, y_train_labels, y_test_labels = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    # Normalize features (important for neural networks!)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # One-hot encode labels
    n_classes = len(iris.target_names)
    y_train = np.eye(n_classes)[y_train_labels]
    y_test = np.eye(n_classes)[y_test_labels]
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # 3. Initialize network
    network = SimpleNeuralNetwork(
        input_size=X_train.shape[1],  # 4 features
        hidden_size=8,  # 8 neurons in hidden layer
        output_size=n_classes  # 3 classes
    )
    
    print(f"\nNetwork Architecture:")
    print(f"  Input layer: {X_train.shape[1]} neurons")
    print(f"  Hidden layer: 8 neurons (ReLU)")
    print(f"  Output layer: {n_classes} neurons (Softmax)")
    
    # Calculate total parameters
    total_params = (
        network.W1.size + network.b1.size +
        network.W2.size + network.b2.size
    )
    print(f"  Total parameters: {total_params}")
    
    # 4. Train network
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    losses = train(network, X_train, y_train, epochs=200, learning_rate=0.1)
    
    # 5. Evaluate on train and test sets
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    # Training accuracy
    train_predictions = network.predict(X_train)
    train_accuracy = np.mean(train_predictions == y_train_labels)
    print(f"\nTraining Accuracy: {train_accuracy:.2%}")
    
    # Test accuracy
    test_predictions = network.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test_labels)
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    # Per-class accuracy
    print(f"\nPer-Class Test Accuracy:")
    for i, class_name in enumerate(iris.target_names):
        class_mask = y_test_labels == i
        class_acc = np.mean(test_predictions[class_mask] == y_test_labels[class_mask])
        n_samples = np.sum(class_mask)
        print(f"  {class_name:12s}: {class_acc:.2%} ({n_samples} samples)")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test_labels, test_predictions)
    print(f"\nConfusion Matrix:")
    print("           Predicted:")
    print("           ", "  ".join([f"{name[:4]:>4s}" for name in iris.target_names]))
    for i, row in enumerate(cm):
        print(f"  {iris.target_names[i][:10]:10s} {row}")
    
    # 6. Visualize results
    def visualize_iris_results():
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid for subplots
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Training loss curve
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(losses, 'b-', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confusion Matrix Heatmap
        ax2 = fig.add_subplot(gs[0, 2])
        im = ax2.imshow(cm, cmap='Blues')
        ax2.set_xticks(range(n_classes))
        ax2.set_yticks(range(n_classes))
        ax2.set_xticklabels([name[:4] for name in iris.target_names])
        ax2.set_yticklabels(iris.target_names)
        ax2.set_xlabel('Predicted', fontweight='bold')
        ax2.set_ylabel('True', fontweight='bold')
        ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        
        # Add text annotations
        for i in range(n_classes):
            for j in range(n_classes):
                text = ax2.text(j, i, cm[i, j],
                               ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black",
                               fontweight='bold')
        
        plt.colorbar(im, ax=ax2)
        
        # Plot 3-6: Feature distributions by class
        feature_pairs = [(0, 1), (2, 3)]
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            ax = fig.add_subplot(gs[1, idx])
            
            for i, class_name in enumerate(iris.target_names):
                mask = y_labels == i
                ax.scatter(X[mask, feat1], X[mask, feat2], 
                          label=class_name, alpha=0.6, s=50)
            
            ax.set_xlabel(iris.feature_names[feat1], fontweight='bold')
            ax.set_ylabel(iris.feature_names[feat2], fontweight='bold')
            ax.set_title(f'Feature Space: {iris.feature_names[feat1][:10]} vs {iris.feature_names[feat2][:10]}',
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Plot 7: Accuracy comparison
        ax7 = fig.add_subplot(gs[1, 2])
        accuracies = [train_accuracy, test_accuracy]
        colors = ['#4ECDC4', '#45B7D1']
        bars = ax7.bar(['Train', 'Test'], accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax7.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax7.set_title('Model Performance', fontsize=14, fontweight='bold')
        ax7.set_ylim([0.8, 1.0])
        ax7.grid(axis='y', alpha=0.3)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Plot 8: Sample predictions
        ax8 = fig.add_subplot(gs[2, :])
        ax8.axis('off')
        
        # Show some test predictions
        n_show = min(10, len(X_test))
        pred_text = "SAMPLE PREDICTIONS\n" + "="*50 + "\n\n"
        
        for i in range(n_show):
            true_class = iris.target_names[y_test_labels[i]]
            pred_class = iris.target_names[test_predictions[i]]
            probs = network.forward(X_test[i:i+1])[0]
            
            correct = "✓" if y_test_labels[i] == test_predictions[i] else "✗"
            pred_text += f"{correct} Sample {i+1}: True={true_class:10s} | Pred={pred_class:10s} "
            pred_text += f"| Confidence: {probs[test_predictions[i]]:.1%}\n"
        
        ax8.text(0.1, 0.5, pred_text, fontsize=10, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow',
                         edgecolor='black', linewidth=2, alpha=0.9))
        
        # Summary box
        summary_text = f"""
        IRIS CLASSIFICATION RESULTS
        ══════════════════════════════════════
        
        📊 DATASET
        • Total Samples: {len(X)}
        • Features: {X.shape[1]}
        • Classes: {n_classes}
        • Train/Test Split: 80/20
        
        🎯 MODEL PERFORMANCE
        • Train Accuracy: {train_accuracy:.1%}
        • Test Accuracy: {test_accuracy:.1%}
        • Parameters: {total_params}
        
        ⚡ ARCHITECTURE
        • Input: 4 features
        • Hidden: 8 neurons (ReLU)
        • Output: 3 classes (Softmax)
        • Training: 200 epochs
        
        ══════════════════════════════════════
        """
        
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')
        ax9.text(0.5, 0.5, summary_text, fontsize=9, family='monospace',
                verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue',
                         edgecolor='black', linewidth=2, alpha=0.9))
        
        plt.suptitle('IRIS Dataset - Neural Network Classification', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.savefig('iris_classification_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved results to 'iris_classification_results.png'")
        plt.show()
    
    visualize_iris_results()
    
    # 7. Interactive prediction
    print("\n" + "="*70)
    print("INTERACTIVE PREDICTION")
    print("="*70)
    
    def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
        """Predict iris species from measurements."""
        # Prepare input
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
        
        # Get prediction
        probs = network.forward(features_scaled)[0]
        prediction = np.argmax(probs)
        
        print(f"\nInput measurements:")
        print(f"  Sepal: {sepal_length:.1f} cm × {sepal_width:.1f} cm")
        print(f"  Petal: {petal_length:.1f} cm × {petal_width:.1f} cm")
        print(f"\nPrediction: {iris.target_names[prediction]}")
        print(f"Confidence: {probs[prediction]:.1%}")
        print(f"\nAll probabilities:")
        for i, (name, prob) in enumerate(zip(iris.target_names, probs)):
            print(f"  {name:12s}: {prob:.1%} {'█' * int(prob * 50)}")
    
    # Example predictions
    print("\nExample 1: Typical Setosa")
    predict_flower(5.1, 3.5, 1.4, 0.2)
    
    print("\nExample 2: Typical Versicolor")
    predict_flower(6.0, 2.7, 5.1, 1.6)
    
    print("\nExample 3: Typical Virginica")
    predict_flower(6.5, 3.0, 5.8, 2.2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Model achieved {test_accuracy:.1%} accuracy on test set")
    print("Model ready for deployment!")
    print("="*70)
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
