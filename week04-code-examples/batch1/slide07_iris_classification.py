"""
Week 4: Slide 7 - Complete IRIS Classification with Neural Network Training

Description: 
    Complete end-to-end neural network training example using IRIS dataset.
    Includes training loop, backpropagation, evaluation, and visualization.
    This is a production-quality implementation demonstrating the full ML pipeline.

Dependencies:
    - numpy
    - scikit-learn
    - matplotlib

Usage:
    python slide07_iris_classification.py
    
Note: This file requires SimpleNeuralNetwork class from slide06.
      Either run from same directory or import it.
"""

import numpy as np
from typing import Tuple

# Import SimpleNeuralNetwork from slide06
# If running standalone, copy the class definition here
class SimpleNeuralNetwork:
    """A basic 2-layer neural network for classification."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.forward(X)
        return np.argmax(probs, axis=1)


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
        dz2 = predictions - y_train
        dW2 = np.dot(network.a1.T, dz2) / n_samples
        db2 = np.sum(dz2, axis=0, keepdims=True) / n_samples
        
        dz1 = np.dot(dz2, network.W2.T)
        dz1[network.z1 <= 0] = 0
        
        dW1 = np.dot(X_train.T, dz1) / n_samples
        db1 = np.sum(dz1, axis=0, keepdims=True) / n_samples
        
        # 4. Update weights (Gradient Descent)
        network.W1 -= learning_rate * dW1
        network.b1 -= learning_rate * db1
        network.W2 -= learning_rate * dW2
        network.b2 -= learning_rate * db2
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}, Loss: {loss:.4f}")
    
    return losses


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    
    print("="*70)
    print("IRIS FLOWER CLASSIFICATION WITH NEURAL NETWORK")
    print("="*70)
    
    # 1. Load IRIS dataset
    print("\nLoading IRIS dataset...")
    iris = load_iris()
    X = iris.data
    y_labels = iris.target
    
    print(f"Dataset size: {X.shape[0]} samples")
    print(f"Features: {X.shape[1]} ({iris.feature_names})")
    print(f"Classes: {len(iris.target_names)} ({iris.target_names})")
    print(f"Class distribution: {np.bincount(y_labels)}")
    
    # 2. Split and normalize data
    X_train, X_test, y_train_labels, y_test_labels = train_test_split(
        X, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    n_classes = len(iris.target_names)
    y_train = np.eye(n_classes)[y_train_labels]
    y_test = np.eye(n_classes)[y_test_labels]
    
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    # 3. Initialize network
    network = SimpleNeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=8,
        output_size=n_classes
    )
    
    print(f"\nNetwork Architecture:")
    print(f"  Input layer: {X_train.shape[1]} neurons")
    print(f"  Hidden layer: 8 neurons (ReLU)")
    print(f"  Output layer: {n_classes} neurons (Softmax)")
    
    total_params = (network.W1.size + network.b1.size +
                   network.W2.size + network.b2.size)
    print(f"  Total parameters: {total_params}")
    
    # 4. Train network
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    losses = train(network, X_train, y_train, epochs=200, learning_rate=0.1)
    
    # 5. Evaluate
    print("\n" + "="*70)
    print("EVALUATION")
    print("="*70)
    
    train_predictions = network.predict(X_train)
    train_accuracy = np.mean(train_predictions == y_train_labels)
    print(f"\nTraining Accuracy: {train_accuracy:.2%}")
    
    test_predictions = network.predict(X_test)
    test_accuracy = np.mean(test_predictions == y_test_labels)
    print(f"Test Accuracy: {test_accuracy:.2%}")
    
    print(f"\nPer-Class Test Accuracy:")
    for i, class_name in enumerate(iris.target_names):
        class_mask = y_test_labels == i
        class_acc = np.mean(test_predictions[class_mask] == y_test_labels[class_mask])
        n_samples = np.sum(class_mask)
        print(f"  {class_name:12s}: {class_acc:.2%} ({n_samples} samples)")
    
    cm = confusion_matrix(y_test_labels, test_predictions)
    print(f"\nConfusion Matrix:")
    print("           Predicted:")
    print("           ", "  ".join([f"{name[:4]:>4s}" for name in iris.target_names]))
    for i, row in enumerate(cm):
        print(f"  {iris.target_names[i][:10]:10s} {row}")
    
    # 6. Visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Training loss
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(losses, 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 2])
    im = ax2.imshow(cm, cmap='Blues')
    ax2.set_xticks(range(n_classes))
    ax2.set_yticks(range(n_classes))
    ax2.set_xticklabels([name[:4] for name in iris.target_names])
    ax2.set_yticklabels(iris.target_names)
    ax2.set_xlabel('Predicted', fontweight='bold')
    ax2.set_ylabel('True', fontweight='bold')
    ax2.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    for i in range(n_classes):
        for j in range(n_classes):
            ax2.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black",
                    fontweight='bold')
    plt.colorbar(im, ax=ax2)
    
    # Feature distributions
    feature_pairs = [(0, 1), (2, 3)]
    for idx, (feat1, feat2) in enumerate(feature_pairs):
        ax = fig.add_subplot(gs[1, idx])
        for i, class_name in enumerate(iris.target_names):
            mask = y_labels == i
            ax.scatter(X[mask, feat1], X[mask, feat2], 
                      label=class_name, alpha=0.6, s=50)
        ax.set_xlabel(iris.feature_names[feat1], fontweight='bold')
        ax.set_ylabel(iris.feature_names[feat2], fontweight='bold')
        ax.set_title(f'{iris.feature_names[feat1][:10]} vs {iris.feature_names[feat2][:10]}',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Accuracy comparison
    ax7 = fig.add_subplot(gs[1, 2])
    accuracies = [train_accuracy, test_accuracy]
    colors = ['#4ECDC4', '#45B7D1']
    bars = ax7.bar(['Train', 'Test'], accuracies, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=2)
    ax7.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax7.set_title('Model Performance', fontsize=14, fontweight='bold')
    ax7.set_ylim([0.8, 1.0])
    ax7.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', 
                fontweight='bold', fontsize=12)
    
    # Sample predictions
    ax8 = fig.add_subplot(gs[2, :])
    ax8.axis('off')
    
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
    
    # Summary
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
    
    # 7. Interactive prediction
    print("\n" + "="*70)
    print("INTERACTIVE PREDICTION")
    print("="*70)
    
    def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        features_scaled = scaler.transform(features)
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
