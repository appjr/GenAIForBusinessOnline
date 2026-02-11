"""
Week 4: Deep Learning and the GenAI General Algorithm
Slide 5: Activation Functions

Description: 
    Implementation and visualization of common activation functions used in neural networks:
    - Sigmoid (for binary classification output)
    - ReLU (most common for hidden layers)
    - Tanh (zero-centered alternative to sigmoid)
    - Softmax (for multi-class classification output)

Dependencies:
    - numpy
    - matplotlib

Usage:
    python slide05_activation_functions.py
"""

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
