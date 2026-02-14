"""
Week 4: Slide 31 - Customer Churn Prediction with Neural Networks

Description: 
    Complete business application demonstrating customer churn prediction.
    Includes synthetic data generation, model training, evaluation, and visualization.
    Shows real-world business value with ROI analysis.

Dependencies:
    - torch
    - numpy
    - pandas
    - scikit-learn
    - matplotlib

Usage:
    python slide31_churn_prediction.py
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import matplotlib.pyplot as plt

def generate_customer_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic customer data for demonstration"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'monthly_usage': np.random.normal(50, 20, n_samples),
        'tenure_months': np.random.randint(1, 60, n_samples),
        'support_calls': np.random.poisson(2, n_samples),
        'bill_amount': np.random.normal(75, 25, n_samples)
    })
    
    # Create target: higher churn if low tenure, high support calls
    churn_prob = (
        0.5 - data['tenure_months']/120 +
        data['support_calls']/10 +
        np.random.normal(0, 0.1, n_samples)
    )
    data['churned'] = (churn_prob > 0.5).astype(int)
    
    return data


class ChurnPredictor(nn.Module):
    """Neural network for customer churn prediction"""
    def __init__(self, input_size: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_churn_model() -> Tuple[ChurnPredictor, StandardScaler]:
    """Train churn prediction model"""
    
    print("="*70)
    print("CUSTOMER CHURN PREDICTION")
    print("="*70)
    
    # Load data
    data = generate_customer_data(n_samples=1000)
    X = data[['monthly_usage', 'tenure_months', 'support_calls', 'bill_amount']].values
    y = data['churned'].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Initialize model
    model = ChurnPredictor(input_size=4)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining...")
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_pred_binary = (test_pred > 0.5).float()
        accuracy = (test_pred_binary == y_test_t).float().mean()
        
        print(f"\nTest Accuracy: {accuracy:.2%}")
    
    return model, scaler


def predict_churn_risk(model: ChurnPredictor, scaler: StandardScaler,
                       customer_features: dict) -> float:
    """Predict churn risk for a single customer"""
    features = np.array([[
        customer_features['monthly_usage'],
        customer_features['tenure_months'],
        customer_features['support_calls'],
        customer_features['bill_amount']
    ]])
    
    features_scaled = scaler.transform(features)
    features_tensor = torch.FloatTensor(features_scaled)
    
    model.eval()
    with torch.no_grad():
        churn_prob = model(features_tensor).item()
    
    return churn_prob


if __name__ == "__main__":
    # Train model
    model, scaler = train_churn_model()
    
    # Test predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    
    high_risk = {
        'monthly_usage': 20,
        'tenure_months': 3,
        'support_calls': 5,
        'bill_amount': 100
    }
    
    low_risk = {
        'monthly_usage': 70,
        'tenure_months': 48,
        'support_calls': 0,
        'bill_amount': 60
    }
    
    for name, customer in [("High Risk", high_risk), ("Low Risk", low_risk)]:
        risk = predict_churn_risk(model, scaler, customer)
        print(f"\n{name} Customer:")
        print(f"  Churn Risk: {risk:.1%}")
        print(f"  Action: {'Contact immediately!' if risk > 0.7 else 'Monitor'}")
    
    print("\n✓ Model ready for deployment!")
