"""
Week 4: Slide 4 - What is a Neural Network?

Description: 
    Basic artificial neuron (perceptron) implementation.
    Demonstrates how a single neuron processes inputs with weights and bias.

Dependencies:
    - numpy

Usage:
    python slide04_neuron.py
"""

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
