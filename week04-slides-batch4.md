# Week 4: Deep Learning and the GenAI General Algorithm - Slides Batch 4 (Slides 31-40)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 11, 2026

---

## Slide 31: Business Applications of Deep Learning

### Real-World Use Cases

**1. Customer Analytics**
- Churn prediction
- Lifetime value estimation
- Segmentation and personalization

**2. NLP Applications**
- Sentiment analysis for brand monitoring
- Document classification and routing
- Automated customer support

**3. Computer Vision**
- Quality control in manufacturing
- Medical image analysis
- Retail analytics (foot traffic, shelf monitoring)

**4. Time Series Forecasting**
- Demand forecasting
- Stock price prediction
- Equipment maintenance scheduling

**Business Impact Example:**
```python
"""
Customer Churn Prediction using Neural Networks
Business Value: Proactive retention campaigns
Expected Result: 25% reduction in churn, $2M annual savings
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

# Generate synthetic customer data (replace with your actual data)
def generate_customer_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Generate synthetic customer data for demonstration.
    
    In production, replace this with:
        data = pd.read_csv('customer_data.csv')
    """
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
    """
    Neural network for customer churn prediction.
    
    Architecture: 4 → 64 → 32 → 1
    Output: Probability of customer churning
    """
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
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_churn_model() -> Tuple[ChurnPredictor, StandardScaler]:
    """Train churn prediction model."""
    
    # 1. Load and prepare data
    print("Loading customer data...")
    data = generate_customer_data(n_samples=1000)
    
    X = data[['monthly_usage', 'tenure_months', 'support_calls', 'bill_amount']].values
    y = data['churned'].values
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 4. Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)
    
    # 5. Initialize model
    model = ChurnPredictor(input_size=4)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 6. Training loop
    print("\nTraining churn predictor...")
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    # 7. Evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t)
        test_pred_binary = (test_pred > 0.5).float()
        accuracy = (test_pred_binary == y_test_t).float().mean()
        
        print(f"\nTest Accuracy: {accuracy:.2%}")
    
    return model, scaler


def predict_churn_risk(model: ChurnPredictor, scaler: StandardScaler,
                       customer_features: dict) -> float:
    """
    Predict churn risk for a single customer.
    
    Args:
        model: Trained churn predictor
        scaler: Fitted StandardScaler
        customer_features: Dict with customer data
    
    Returns:
        Churn probability (0-1)
    """
    # Prepare features
    features = np.array([[
        customer_features['monthly_usage'],
        customer_features['tenure_months'],
        customer_features['support_calls'],
        customer_features['bill_amount']
    ]])
    
    # Normalize
    features_scaled = scaler.transform(features)
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Predict
    model.eval()
    with torch.no_grad():
        churn_prob = model(features_tensor).item()
    
    return churn_prob


# Complete example usage
if __name__ == "__main__":
    # Train model
    model, scaler = train_churn_model()
    
    # Predict for new customers
    print("\n" + "="*50)
    print("CHURN RISK PREDICTIONS")
    print("="*50)
    
    high_risk_customer = {
        'monthly_usage': 20,
        'tenure_months': 3,
        'support_calls': 5,
        'bill_amount': 100
    }
    
    low_risk_customer = {
        'monthly_usage': 70,
        'tenure_months': 48,
        'support_calls': 0,
        'bill_amount': 60
    }
    
    for name, customer in [("High Risk", high_risk_customer), 
                           ("Low Risk", low_risk_customer)]:
        risk = predict_churn_risk(model, scaler, customer)
        print(f"\n{name} Customer:")
        print(f"  Features: {customer}")
        print(f"  Churn Risk: {risk:.1%}")
        print(f"  Action: {'URGENT - Contact immediately!' if risk > 0.7 else 'Monitor'}")
    
    print("\n" + "="*50)
    print("BUSINESS IMPACT")
    print("="*50)
    print("• Early identification of at-risk customers")
    print("• Targeted retention campaigns")
    print("• Expected: 25% reduction in churn")
    print("• Estimated savings: $2M annually")
    
    # Visualize business impact
    import matplotlib.pyplot as plt
    
    def visualize_business_impact():
        """Visualize churn prediction model's business value."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Risk Distribution
        customers = 1000
        high_risk = 150
        medium_risk = 300
        low_risk = 550
        
        ax1 = axes[0, 0]
        colors = ['#FF6B6B', '#FFA07A', '#90EE90']
        sizes = [high_risk, medium_risk, low_risk]
        labels = [f'High Risk\n({high_risk})', 
                 f'Medium Risk\n({medium_risk})', 
                 f'Low Risk\n({low_risk})']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=90, textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax1.set_title('Customer Risk Distribution', fontsize=12, fontweight='bold')
        
        # 2. Churn Reduction Impact
        ax2 = axes[0, 1]
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        before = [12, 13, 11, 14, 12, 13]
        after = [12, 11, 9, 8, 7, 6]
        
        x = np.arange(len(months))
        width = 0.35
        
        ax2.bar(x - width/2, before, width, label='Before AI', color='#FF6B6B', alpha=0.8)
        ax2.bar(x + width/2, after, width, label='After AI', color='#4ECDC4', alpha=0.8)
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Churn Rate (%)')
        ax2.set_title('Churn Reduction Over Time', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(months)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Cost Savings
        ax3 = axes[1, 0]
        quarters = ['Q1', 'Q2', 'Q3', 'Q4']
        savings = [400, 500, 550, 550]  # in thousands
        cumulative = np.cumsum(savings)
        
        ax3.bar(quarters, savings, color='#45B7D1', alpha=0.7, label='Quarterly Savings')
        ax3.plot(quarters, cumulative, 'ro-', linewidth=2, 
                markersize=8, label='Cumulative Savings')
        ax3.set_xlabel('Quarter')
        ax3.set_ylabel('Savings ($1000s)')
        ax3.set_title('Cost Savings from Churn Reduction', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value annotations
        for i, (q, s, c) in enumerate(zip(quarters, savings, cumulative)):
            ax3.text(i, s + 20, f'${s}K', ha='center', fontweight='bold')
            ax3.text(i, c + 30, f'${c}K', ha='center', 
                    color='red', fontweight='bold', fontsize=9)
        
        # 4. ROI Analysis
        ax4 = axes[1, 1]
        categories = ['Development\nCost', 'Annual\nMaintenance', 
                     'Annual\nSavings', 'Net\nBenefit']
        values = [-200, -40, 2000, 1760]
        colors_roi = ['#FF6B6B' if v < 0 else '#4ECDC4' for v in values]
        
        bars = ax4.bar(categories, values, color=colors_roi, alpha=0.8)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax4.set_ylabel('Amount ($1000s)')
        ax4.set_title('Annual ROI Analysis', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (50 if height > 0 else -50),
                    f'${abs(val)}K',
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontweight='bold', fontsize=10)
        
        # Add ROI percentage
        roi = (values[3] / abs(values[0] + values[1])) * 100
        ax4.text(0.5, 0.95, f'ROI: {roi:.0f}%', 
                transform=ax4.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('business_impact.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved business impact visualization to 'business_impact.png'")
        plt.show()
    
    visualize_business_impact()
```

---

## Slide 32: ROI of GenAI Projects

### Measuring Business Value

**Cost Components:**
1. **Development**: $50K - $500K
2. **Infrastructure**: $1K - $100K/month
3. **Maintenance**: 15-20% of development cost/year

**Value Calculation:**
```python
class ROICalculator:
    def calculate_roi(self, years=3):
        # Costs
        development_cost = 200000
        annual_infrastructure = 24000
        annual_maintenance = development_cost * 0.15
        
        total_cost = development_cost + (annual_infrastructure + annual_maintenance) * years
        
        # Benefits
        annual_benefit = 500000  # Example: automation savings
        total_benefit = annual_benefit * years
        
        # ROI
        roi = (total_benefit - total_cost) / total_cost * 100
        payback_period = total_cost / annual_benefit
        
        return {'roi_percent': roi, 'payback_years': payback_period}
```

---

## Slide 33: Ethical Considerations

### Responsible AI Development

**Key Concerns:**
1. Bias and Fairness
2. Privacy and Data Protection
3. Transparency and Explainability
4. Environmental Impact

---

## Slide 34: Model Interpretability

### Understanding AI Decisions

**Techniques:**
- SHAP values
- LIME
- Attention visualization

---

## Slide 35: Production Deployment

### From Research to Reality

**Deployment Steps:**
1. Model optimization (quantization)
2. API development
3. Monitoring and logging
4. A/B testing

---

## Slide 36: Week 4 Assignment

### Build and Train a Small Transformer

**Requirements:**
- Implement multi-head attention
- Build transformer blocks
- Train on text corpus
- Generate text samples

**Due Date:** February 18, 2026

---

## Slide 37: Recommended Resources

### Continue Learning

**Books:**
- Deep Learning by Goodfellow
- Dive into Deep Learning (d2l.ai)

**Courses:**
- Fast.ai
- Stanford CS231n, CS224n
- DeepLearning.AI

---

## Slide 38: Next Week Preview

### Week 5: Drawing, Image Generation, Music and Sounds

**Topics:**
- VAEs and GANs
- Diffusion Models
- Audio Generation
- Multimodal Models

---

## Slide 39: Week 4 Recap

### Key Takeaways

✅ Neural Networks and Deep Learning
✅ Attention Mechanisms
✅ Transformer Architecture
✅ GPT and Language Models
✅ Training and Evaluation

---

## Slide 40: Q&A and Office Hours

### Questions and Discussion

**Office Hours:**
- Tuesday 2-4 PM (Virtual)
- Thursday 3-5 PM (In-person)

**Contact:**
- Email: [instructor@email.com]
- Canvas Messages

---

**End of Week 4 - Deep Learning and the GenAI General Algorithm**

*Total Slides: 40*
*Batch 4: Slides 31-40*
