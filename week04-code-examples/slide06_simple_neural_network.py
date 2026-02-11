"""
Week 4: Deep Learning and the GenAI General Algorithm
Slide 6: Building a Simple Neural Network

Description: 
    A basic 2-layer neural network for classification.
    Architecture: Input → Hidden (ReLU) → Output (Softmax)

Dependencies:
    - numpy

Usage:
    python slide06_simple_neural_network.py
"""

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
