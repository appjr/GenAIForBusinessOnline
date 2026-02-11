# Week 4 Code Examples

## Deep Learning and the GenAI General Algorithm

This directory contains standalone Python code examples extracted from Week 4 lecture slides. Each file corresponds to specific slides and can be run independently.

---

## 📁 File Organization

All files are named according to the slide number they come from:
- Format: `slideXX_descriptive_name.py`
- Each file is self-contained and runnable
- Full documentation included in each file

---

## 📋 Available Code Examples

### Neural Network Fundamentals

#### `slide04_neuron.py`
- **Slide:** 4 - What is a Neural Network?
- **Description:** Basic artificial neuron with sigmoid activation
- **Key Concepts:** Weighted sums, bias, activation functions
- **Dependencies:** numpy
- **Run:** `python slide04_neuron.py`

#### `slide05_activation_functions.py`
- **Slide:** 5 - Activation Functions
- **Description:** Implementation and visualization of Sigmoid, ReLU, Tanh, and Softmax
- **Key Concepts:** Non-linearity, output ranges, use cases
- **Dependencies:** numpy, matplotlib
- **Run:** `python slide05_activation_functions.py`
- **Output:** Creates `activation_functions.png`

#### `slide06_simple_neural_network.py`
- **Slide:** 6 - Building a Simple Neural Network
- **Description:** 2-layer neural network for classification
- **Architecture:** Input → Hidden (ReLU) → Output (Softmax)
- **Key Concepts:** Forward propagation, network architecture
- **Dependencies:** numpy
- **Run:** `python slide06_simple_neural_network.py`

#### `slide07_iris_classification.py`
- **Slide:** 7 - Complete Training Example
- **Description:** Full IRIS dataset classification with neural network
- **Key Concepts:** Data preprocessing, training loop, evaluation
- **Dependencies:** numpy, sklearn, matplotlib
- **Run:** `python slide07_iris_classification.py`
- **Note:** Simplified version. See slides for complete implementation with visualizations

---

### Business Applications

#### `slide31_churn_prediction.py`
- **Slide:** 31 - Customer Churn Prediction
- **Description:** Predict customer churn using neural networks
- **Business Value:** $2M annual savings, 25% churn reduction
- **Key Metrics:** >90% accuracy target
- **Dependencies:** pandas, numpy, torch, sklearn
- **Run:** `python slide31_churn_prediction.py`
- **Note:** Skeleton file - see slides for full implementation

#### `slide31b_fraud_detection.py`
- **Slide:** 31B - Fraud Detection
- **Description:** Real-time fraud detection system
- **Business Value:** $7.68M annual savings, 50% fraud reduction
- **Key Metrics:** <100ms inference, 70% fewer false positives
- **Dependencies:** pandas, numpy, torch, sklearn
- **Run:** `python slide31b_fraud_detection.py`
- **Note:** Skeleton file - see slides for full implementation

#### `slide31c_demand_forecasting.py`
- **Slide:** 31C - Demand Forecasting with LSTM
- **Description:** Time series prediction for inventory optimization
- **Business Value:** $2.2M annual savings, 20-30% inventory reduction
- **Key Metrics:** 85-95% forecast accuracy
- **Dependencies:** pandas, numpy, torch, sklearn
- **Run:** `python slide31c_demand_forecasting.py`
- **Note:** Skeleton file - see slides for full implementation

#### `slide32_roi_calculator.py`
- **Slide:** 32 - ROI of GenAI Projects
- **Description:** Comprehensive ROI calculator with sensitivity analysis
- **Features:** Cost breakdown, benefit analysis, NPV, payback period
- **Dependencies:** numpy, matplotlib
- **Run:** `python slide32_roi_calculator.py`
- **Note:** Skeleton file - see slides for full implementation

---

## 🚀 Quick Start

### Installation

1. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install numpy matplotlib scikit-learn torch pandas
```

### Running Examples

#### Basic Examples (No External Data Required):
```bash
# Run a simple neuron
python slide04_neuron.py

# Visualize activation functions
python slide05_activation_functions.py

# Test a simple neural network
python slide06_simple_neural_network.py
```

#### Complete Training Example:
```bash
# IRIS classification (downloads dataset automatically)
python slide07_iris_classification.py
```

#### Business Use Cases:
```bash
# Note: These are skeleton files
# For complete implementations, see the slide markdown files
python slide31_churn_prediction.py
python slide31b_fraud_detection.py
python slide31c_demand_forecasting.py
python slide32_roi_calculator.py
```

---

## 📚 Learning Path

**Recommended order for beginners:**

1. **Start Simple:**
   - `slide04_neuron.py` - Understand basic building blocks
   - `slide05_activation_functions.py` - Learn about activation functions

2. **Build Up:**
   - `slide06_simple_neural_network.py` - See how neurons combine into networks

3. **Complete Example:**
   - `slide07_iris_classification.py` - Full training pipeline

4. **Business Applications:**
   - Explore the business use case files to see real-world applications

---

## 🔍 Key Concepts by File

| File | Concepts Covered |
|------|-----------------|
| slide04 | Neurons, weights, bias, sigmoid activation |
| slide05 | Sigmoid, ReLU, Tanh, Softmax, activation visualization |
| slide06 | Network architecture, forward propagation, Xavier initialization |
| slide07 | Training loops, backpropagation, evaluation, data preprocessing |
| slide31 | Binary classification, churn prediction, business ROI |
| slide31b | Imbalanced data, precision/recall, real-time inference |
| slide31c | Time series, LSTM, sequential data, demand forecasting |
| slide32 | ROI calculation, cost-benefit analysis, NPV, payback period |

---

## 📊 Datasets Used

- **IRIS Dataset** (slide07): Built-in scikit-learn dataset
  - 150 samples, 4 features, 3 classes
  - Classic machine learning benchmark
  - No download required

- **Synthetic Data** (slide31, 31b, 31c): Generated programmatically
  - Churn: Customer behavior patterns
  - Fraud: Transaction patterns
  - Demand: Time series with seasonality

---

## 🛠️ Common Issues & Solutions

### Issue: Import Errors
```bash
# Solution: Install missing packages
pip install numpy matplotlib scikit-learn torch pandas
```

### Issue: ModuleNotFoundError for local imports
```bash
# Solution: Run from the week04-code-examples directory
cd week04-code-examples
python slide07_iris_classification.py
```

### Issue: matplotlib not showing plots
```bash
# Solution: Use plt.show() or save to file
# Most examples already include plt.savefig()
```

---

## 📖 Additional Resources

### For Complete Implementations:
- Full code with all features is in the markdown slide files:
  - `week04-slides-batch1.md` - Slides 1-10 (Neural network basics)
  - `week04-slides-batch2.md` - Slides 11-20 (CNNs, RNNs, Attention)
  - `week04-slides-batch3.md` - Slides 21-30 (GPT, Transformers, Training)
  - `week04-slides-batch4.md` - Slides 31-40 (Business applications, ROI)

### Documentation:
- Each `.py` file includes comprehensive docstrings
- Run with `--help` or read the header comments
- Check the slide files for theoretical background

---

## 💡 Tips for Students

1. **Start with the basics**: Don't skip slide04-slide06
2. **Experiment**: Modify hyperparameters and see what happens
3. **Read the code**: Every line is documented
4. **Compare to slides**: Code matches the lecture material
5. **Build your own**: Use these as templates for your projects

---

## 🎯 Assignment Reference

These code examples support the Week 4 assignment:
- **Due:** February 18, 2026
- **Task:** Build and train a small transformer
- **Resources:** Use slide06-slide07 as templates

---

## 📝 Notes

- **Slide References:** All code is extracted from official Week 4 lecture slides
- **Self-Contained:** Each file runs independently
- **Educational:** Heavily commented for learning
- **Production Ready:** Some examples (business cases) show production deployment patterns

---

## 🤝 Support

For questions about these code examples:
- **Office Hours:** Tuesday 2-4 PM (Virtual), Thursday 3-5 PM (In-person)
- **Canvas:** Post in the Week 4 discussion forum
- **Email:** See course syllabus

---

## 📅 Last Updated

February 11, 2026

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Week:** 4 - Deep Learning and the GenAI General Algorithm  
**Instructor:** See course syllabus
