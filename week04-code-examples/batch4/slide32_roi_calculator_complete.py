"""
Week 4: Slide 32 - Comprehensive ROI Calculator for GenAI Projects

Description: 
    Calculate detailed ROI for GenAI projects with sensitivity analysis.
    Helps businesses make informed investment decisions.

Note: This is a reference to the complete implementation in week04-slides-batch4.md
      The full implementation is 350+ lines with comprehensive analysis.
      
To get the complete code, see:
    week04-slides-batch4.md - Slide 32 (lines ~1350-1700)
    
This file contains the core calculator class.

Dependencies:
    - numpy
    - matplotlib

Business Value:
    - Informed investment decisions
    - Risk assessment
    - Scenario planning
    - Stakeholder communication
"""

import numpy as np
from typing import Dict

class ComprehensiveROICalculator:
    """Calculate detailed ROI for GenAI projects with sensitivity analysis"""
    
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
        """Calculate total cost of ownership over project lifetime"""
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
        """Calculate tangible and intangible benefits"""
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
        """Calculate key ROI metrics"""
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
            'irr_estimate': roi_percent / 100
        }

# For complete implementation with:
# - Detailed cost breakdowns
# - Benefit categories
# - Sensitivity analysis
# - Visualization suite (cash flow, cost breakdown, ROI dashboard)
# - Risk assessment
# - Best/worst case scenarios
# 
# See: week04-slides-batch4.md, Slide 32

if __name__ == "__main__":
    print("="*70)
    print("GENAI PROJECT ROI CALCULATOR")
    print("="*70)
    print("\n⚠️  This is a reference implementation.")
    print("\n📖 For the COMPLETE 350-line implementation with:")
    print("   • Detailed cost categories")
    print("   • Benefit analysis")
    print("   • Sensitivity analysis (best/worst case)")
    print("   • Comprehensive visualizations")
    print("   • NPV and payback calculations")
    print("   • Executive dashboard")
    print("\n👉 See: week04-slides-batch4.md, Slide 32")
    print("="*70)
    
    # Demo calculation
    calculator = ComprehensiveROICalculator()
    
    costs = calculator.calculate_total_cost(
        development_cost=200000,
        monthly_infrastructure=5000,
        annual_personnel=80000,
        years=3
    )
    
    benefits = calculator.calculate_benefits(
        automation_savings=300000,
        revenue_increase=150000,
        efficiency_gains=100000,
        years=3
    )
    
    roi = calculator.calculate_roi_metrics(
        costs['total_cost'],
        benefits['total_benefit'],
        benefits['annual_benefit']
    )
    
    print(f"\n📊 QUICK ROI ANALYSIS:")
    print(f"  Total 3-Year Cost: ${costs['total_cost']:,.0f}")
    print(f"  Total 3-Year Benefit: ${benefits['total_benefit']:,.0f}")
    print(f"  Net Benefit: ${roi['net_benefit']:,.0f}")
    print(f"  ROI: {roi['roi_percent']:.1f}%")
    print(f"  Payback Period: {roi['payback_period_years']:.1f} years")
    print(f"\n{'✅ APPROVED' if roi['roi_percent'] > 100 else '⚠️ REVIEW'}")
