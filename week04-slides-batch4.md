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

**Understanding the True Cost of GenAI Implementation**

Many businesses underestimate the full cost of deploying GenAI solutions. A comprehensive ROI analysis must account for all cost components and realistic benefit projections.

**Detailed Cost Components:**

**1. Development Costs: $50K - $500K+**
- **Discovery & Planning** ($10K-$50K): Requirements gathering, feasibility studies, vendor selection
- **Model Development** ($20K-$200K): Custom model training, fine-tuning, or API integration
- **Integration** ($15K-$150K): Connecting to existing systems, databases, workflows
- **UI/UX Development** ($10K-$75K): Building user interfaces, dashboards
- **Testing & QA** ($5K-$50K): Quality assurance, security testing, compliance checks

**2. Infrastructure Costs: $1K - $100K/month**
- **Cloud Computing** ($500-$50K/month): GPU instances, API calls, storage
- **API Costs** ($100-$20K/month): OpenAI, Anthropic, or other provider fees
- **Data Storage** ($100-$5K/month): Training data, logs, model versions
- **Monitoring Tools** ($200-$2K/month): Performance tracking, error logging
- **Security** ($500-$10K/month): Encryption, access control, compliance tools

**3. Ongoing Costs:**
- **Maintenance** (15-20% of dev cost/year): Bug fixes, updates, optimizations
- **Personnel** ($50K-$200K/year): AI specialists, data scientists (full-time or consultants)
- **Training** ($5K-$25K/year): Staff education, change management
- **Compliance & Audits** ($10K-$50K/year): Regulatory compliance, bias audits

**Comprehensive ROI Calculator:**

```python
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

class ComprehensiveROICalculator:
    """
    Calculate detailed ROI for GenAI projects with sensitivity analysis.
    """
    
    def __init__(self):
        self.costs = {}
        self.benefits = {}
        self.risks = {}
    
    def calculate_total_cost(self, 
                            development_cost: float,
                            monthly_infrastructure: float,
                            annual_maintenance_pct: float = 0.15,
                            annual_personnel: float = 100000,
                            years: int = 3) -> Dict[str, float]:
        """
        Calculate total cost of ownership over project lifetime.
        
        Args:
            development_cost: One-time development cost
            monthly_infrastructure: Monthly cloud/API costs
            annual_maintenance_pct: Maintenance as % of dev cost
            annual_personnel: Annual personnel costs
            years: Project timeline
        
        Returns:
            Dictionary with cost breakdown
        """
        annual_infrastructure = monthly_infrastructure * 12
        annual_maintenance = development_cost * annual_maintenance_pct
        annual_recurring = annual_infrastructure + annual_maintenance + annual_personnel
        
        total_cost = development_cost + (annual_recurring * years)
        
        return {
            'development': development_cost,
            'infrastructure_total': annual_infrastructure * years,
            'maintenance_total': annual_maintenance * years,
            'personnel_total': annual_personnel * years,
            'annual_recurring': annual_recurring,
            'total_cost': total_cost
        }
    
    def calculate_benefits(self,
                          automation_savings: float = 0,
                          revenue_increase: float = 0,
                          efficiency_gains: float = 0,
                          customer_satisfaction_value: float = 0,
                          years: int = 3) -> Dict[str, float]:
        """
        Calculate tangible and intangible benefits.
        
        Args:
            automation_savings: Annual savings from automated tasks
            revenue_increase: Annual revenue increase from new capabilities
            efficiency_gains: Annual value of productivity improvements
            customer_satisfaction_value: Estimated value of improved customer experience
            years: Project timeline
        
        Returns:
            Dictionary with benefit breakdown
        """
        annual_benefit = (automation_savings + revenue_increase + 
                         efficiency_gains + customer_satisfaction_value)
        
        total_benefit = annual_benefit * years
        
        return {
            'automation_savings': automation_savings * years,
            'revenue_increase': revenue_increase * years,
            'efficiency_gains': efficiency_gains * years,
            'customer_value': customer_satisfaction_value * years,
            'annual_benefit': annual_benefit,
            'total_benefit': total_benefit
        }
    
    def calculate_roi_metrics(self,
                             total_cost: float,
                             total_benefit: float,
                             annual_benefit: float) -> Dict[str, float]:
        """
        Calculate key ROI metrics.
        
        Returns:
            Dictionary with ROI, payback period, NPV, etc.
        """
        net_benefit = total_benefit - total_cost
        roi_percent = (net_benefit / total_cost) * 100
        payback_period = total_cost / annual_benefit if annual_benefit > 0 else float('inf')
        
        # NPV calculation (assuming 10% discount rate)
        discount_rate = 0.10
        years = int(total_benefit / annual_benefit) if annual_benefit > 0 else 3
        
        npv = -total_cost
        for year in range(1, years + 1):
            npv += annual_benefit / ((1 + discount_rate) ** year)
        
        return {
            'roi_percent': roi_percent,
            'net_benefit': net_benefit,
            'payback_period_years': payback_period,
            'npv': npv,
            'irr_estimate': roi_percent / 100  # Simplified IRR estimate
        }
    
    def sensitivity_analysis(self,
                           base_costs: Dict,
                           base_benefits: Dict,
                           variation_pct: float = 0.20) -> Dict:
        """
        Perform sensitivity analysis on key variables.
        
        Args:
            base_costs: Base cost assumptions
            base_benefits: Base benefit assumptions
            variation_pct: Percentage variation to test (e.g., 0.20 = ±20%)
        """
        scenarios = {}
        
        # Best case: costs -20%, benefits +20%
        best_case_cost = base_costs['total_cost'] * (1 - variation_pct)
        best_case_benefit = base_benefits['total_benefit'] * (1 + variation_pct)
        scenarios['best_case'] = self.calculate_roi_metrics(
            best_case_cost, best_case_benefit, base_benefits['annual_benefit'] * (1 + variation_pct)
        )
        
        # Base case
        scenarios['base_case'] = self.calculate_roi_metrics(
            base_costs['total_cost'], base_benefits['total_benefit'], base_benefits['annual_benefit']
        )
        
        # Worst case: costs +20%, benefits -20%
        worst_case_cost = base_costs['total_cost'] * (1 + variation_pct)
        worst_case_benefit = base_benefits['total_benefit'] * (1 - variation_pct)
        scenarios['worst_case'] = self.calculate_roi_metrics(
            worst_case_cost, worst_case_benefit, base_benefits['annual_benefit'] * (1 - variation_pct)
        )
        
        return scenarios
    
    def visualize_roi(self, costs: Dict, benefits: Dict, years: int = 3):
        """
        Create comprehensive ROI visualization.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Cumulative Cash Flow
        ax1 = axes[0, 0]
        timeline = np.arange(0, years + 1)
        initial_cost = costs['development']
        annual_net = benefits['annual_benefit'] - costs['annual_recurring']
        
        cumulative = [-initial_cost]
        for year in range(1, years + 1):
            cumulative.append(cumulative[-1] + annual_net)
        
        ax1.plot(timeline, cumulative, 'b-', linewidth=3, marker='o', markersize=8)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2, alpha=0.7)
        ax1.fill_between(timeline, cumulative, 0, where=np.array(cumulative)>=0, 
                        alpha=0.3, color='green', label='Positive Cash Flow')
        ax1.fill_between(timeline, cumulative, 0, where=np.array(cumulative)<0, 
                        alpha=0.3, color='red', label='Negative Cash Flow')
        ax1.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Cash Flow ($)', fontsize=12, fontweight='bold')
        ax1.set_title('Cumulative Cash Flow Over Time', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        # 2. Cost Breakdown
        ax2 = axes[0, 1]
        cost_categories = ['Development', 'Infrastructure', 'Maintenance', 'Personnel']
        cost_values = [
            costs['development'],
            costs['infrastructure_total'],
            costs['maintenance_total'],
            costs['personnel_total']
        ]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        wedges, texts, autotexts = ax2.pie(cost_values, labels=cost_categories, colors=colors,
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax2.set_title(f'Total Cost Breakdown\nTotal: ${sum(cost_values)/1000:.0f}K', 
                     fontsize=14, fontweight='bold')
        
        # 3. Benefits Breakdown
        ax3 = axes[1, 0]
        benefit_categories = ['Automation\nSavings', 'Revenue\nIncrease', 
                             'Efficiency\nGains', 'Customer\nValue']
        benefit_values = [
            benefits['automation_savings'],
            benefits['revenue_increase'],
            benefits['efficiency_gains'],
            benefits['customer_value']
        ]
        
        x_pos = np.arange(len(benefit_categories))
        bars = ax3.bar(x_pos, benefit_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(benefit_categories, fontsize=10, fontweight='bold')
        ax3.set_ylabel('Value ($)', fontsize=12, fontweight='bold')
        ax3.set_title(f'Total Benefits Breakdown\nTotal: ${sum(benefit_values)/1000:.0f}K', 
                     fontsize=14, fontweight='bold')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax3.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height/1000:.0f}K',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # 4. ROI Summary Dashboard
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        roi_metrics = self.calculate_roi_metrics(
            costs['total_cost'],
            benefits['total_benefit'],
            benefits['annual_benefit']
        )
        
        summary_text = f"""
        ROI SUMMARY DASHBOARD
        {'='*50}
        
        💰 Total Investment: ${costs['total_cost']/1000:.0f}K
        
        📈 Total Return: ${benefits['total_benefit']/1000:.0f}K
        
        💵 Net Benefit: ${roi_metrics['net_benefit']/1000:.0f}K
        
        📊 ROI: {roi_metrics['roi_percent']:.1f}%
        
        ⏱️  Payback Period: {roi_metrics['payback_period_years']:.1f} years
        
        💹 NPV (10% discount): ${roi_metrics['npv']/1000:.0f}K
        
        {'='*50}
        
        VERDICT: {"✅ APPROVED" if roi_metrics['roi_percent'] > 100 else "⚠️ REVIEW NEEDED"}
        """
        
        ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', 
                         edgecolor='black', linewidth=2, alpha=0.9))
        
        plt.tight_layout()
        plt.savefig('roi_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Saved comprehensive ROI analysis to 'roi_comprehensive_analysis.png'")
        plt.show()


# Example Usage: Real-World Customer Service AI
if __name__ == "__main__":
    calculator = ComprehensiveROICalculator()
    
    print("="*60)
    print("GENAI PROJECT ROI ANALYSIS")
    print("Project: AI-Powered Customer Service Chatbot")
    print("="*60)
    
    # Calculate costs
    costs = calculator.calculate_total_cost(
        development_cost=200000,        # $200K for development
        monthly_infrastructure=5000,     # $5K/month for APIs & hosting
        annual_maintenance_pct=0.15,    # 15% annual maintenance
        annual_personnel=80000,          # $80K for AI specialist
        years=3
    )
    
    print("\n📊 COST ANALYSIS:")
    print(f"  Development: ${costs['development']:,.0f}")
    print(f"  Infrastructure (3yr): ${costs['infrastructure_total']:,.0f}")
    print(f"  Maintenance (3yr): ${costs['maintenance_total']:,.0f}")
    print(f"  Personnel (3yr): ${costs['personnel_total']:,.0f}")
    print(f"  ───────────────────────")
    print(f"  TOTAL 3-YEAR COST: ${costs['total_cost']:,.0f}")
    
    # Calculate benefits
    benefits = calculator.calculate_benefits(
        automation_savings=300000,      # $300K/yr saved on support staff
        revenue_increase=150000,         # $150K/yr from faster response → sales
        efficiency_gains=100000,        # $100K/yr from agent productivity
        customer_satisfaction_value=50000,  # $50K/yr estimated CSAT value
        years=3
    )
    
    print("\n💰 BENEFIT ANALYSIS:")
    print(f"  Automation Savings (3yr): ${benefits['automation_savings']:,.0f}")
    print(f"  Revenue Increase (3yr): ${benefits['revenue_increase']:,.0f}")
    print(f"  Efficiency Gains (3yr): ${benefits['efficiency_gains']:,.0f}")
    print(f"  Customer Value (3yr): ${benefits['customer_value']:,.0f}")
    print(f"  ───────────────────────")
    print(f"  TOTAL 3-YEAR BENEFIT: ${benefits['total_benefit']:,.0f}")
    
    # Calculate ROI metrics
    roi = calculator.calculate_roi_metrics(
        costs['total_cost'],
        benefits['total_benefit'],
        benefits['annual_benefit']
    )
    
    print("\n📈 ROI METRICS:")
    print(f"  Net Benefit: ${roi['net_benefit']:,.0f}")
    print(f"  ROI: {roi['roi_percent']:.1f}%")
    print(f"  Payback Period: {roi['payback_period_years']:.2f} years")
    print(f"  NPV (10% discount): ${roi['npv']:,.0f}")
    
    # Sensitivity analysis
    print("\n🔍 SENSITIVITY ANALYSIS:")
    scenarios = calculator.sensitivity_analysis(costs, benefits, variation_pct=0.20)
    
    for scenario_name, metrics in scenarios.items():
        print(f"\n  {scenario_name.upper().replace('_', ' ')}:")
        print(f"    ROI: {metrics['roi_percent']:.1f}%")
        print(f"    Payback: {metrics['payback_period_years']:.2f} years")
    
    # Visualize
    calculator.visualize_roi(costs, benefits, years=3)
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    if roi['roi_percent'] > 100 and roi['payback_period_years'] < 2:
        print("✅ STRONG APPROVAL - Excellent ROI and fast payback")
    elif roi['roi_percent'] > 50:
        print("✅ APPROVED - Positive ROI, reasonable payback")
    elif roi['roi_percent'] > 0:
        print("⚠️  CONDITIONAL - Positive but marginal ROI")
    else:
        print("❌ NOT RECOMMENDED - Negative ROI")
    print("="*60)
```

**Key Takeaways for Business Leaders:**

1. **Payback Period Matters**: Aim for <2 years in fast-moving industries
2. **Hidden Costs**: Budget 50% more than initial estimates for unexpected costs
3. **Benefit Realization**: Benefits often take 6-12 months to fully materialize
4. **Continuous Value**: ROI improves over time as models are optimized
5. **Risk Mitigation**: Always perform sensitivity analysis before approval

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
