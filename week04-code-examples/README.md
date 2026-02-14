# Week 4: Deep Learning and GenAI - Code Examples

Complete collection of runnable code examples from Week 4 slides, demonstrating neural networks, deep learning architectures, and business applications.

## 📁 Directory Structure

```
week04-code-examples/
├── batch1/    # Slides 1-10: Neural Network Fundamentals
├── batch2/    # Slides 11-20: Deep Learning Architectures  
├── batch3/    # Slides 21-30: GPT & Language Models
└── batch4/    # Slides 31-40: Business Applications
```

## 📊 Batch 1: Neural Network Fundamentals

**Slides 1-10: Building blocks of neural networks**

| File | Lines | Description |
|------|-------|-------------|
| `slide04_neuron.py` | 50 | Basic artificial neuron (perceptron) implementation |
| `slide05_activation_functions.py` | 200 | Sigmoid, ReLU, Tanh, Softmax with visualization |
| `slide06_simple_neural_network.py` | 120 | 2-layer neural network from scratch |
| `slide07_iris_classification.py` | 400 | Complete IRIS classification with training pipeline |

**Total: 4 files, ~770 lines**

### Running Examples

```bash
cd batch1

# Basic neuron
python slide04_neuron.py

# Activation functions with plots
python slide05_activation_functions.py

# Simple neural network
python slide06_simple_neural_network.py

# Complete IRIS classification
python slide07_iris_classification.py
```

**Dependencies:** numpy, scikit-learn, matplotlib

---

## 🖼️ Batch 2: Deep Learning Architectures

**Slides 11-20: CNNs, RNNs, Attention, Transformers**

| File | Lines | Description |
|------|-------|-------------|
| `slide12_cnn_mnist.py` | 450 | Complete CNN for MNIST digit classification |

**Total: 1 file, 450 lines**

### Running Examples

```bash
cd batch2

# CNN with MNIST (downloads dataset automatically)
python slide12_cnn_mnist.py
```

**Dependencies:** torch, torchvision, matplotlib, numpy

**Note:** Additional code for slides 13-19 (RNNs, LSTM, Attention, Transformers) is available in the markdown slide files.

---

## 🤖 Batch 3: GPT & Language Models

**Slides 21-30: Transformers, GPT, Training**

| File | Lines | Description |
|------|-------|-------------|
| `slide23_language_model_training.py` | 280 | Complete language model training with Shakespeare text |

**Total: 1 file, 280 lines**

### Running Examples

```bash
cd batch3

# Train mini GPT on Shakespeare (downloads text automatically)
python slide23_language_model_training.py
```

**Dependencies:** torch, urllib, matplotlib

**Note:** Code for GPT architecture, text generation, tokenization is available in the markdown slides.

---

## 💼 Batch 4: Business Applications

**Slides 31-40: Real-world use cases with ROI analysis**

| File | Lines | Description |
|------|-------|-------------|
| `slide31_churn_prediction.py` | 170 | Customer churn prediction with neural networks |

**Total: 1 file, 170 lines**

**Note:** Additional business applications (Fraud Detection, Demand Forecasting, ROI Calculator) with complete implementations are available in the markdown slide files (slides 31b, 31c, 32).

### Running Examples

```bash
cd batch4

# Customer churn prediction
python slide31_churn_prediction.py
```

**Dependencies:** torch, pandas, numpy, scikit-learn, matplotlib

---

## 📚 Complete File Inventory

### ✅ Extracted Files (6 files, ~1,670 lines)
- Neural network basics (4 files)
- CNN MNIST training (1 file)
- Language model training (1 file)
- Churn prediction (1 file)

### 📝 Available in Markdown Slides
Additional complete implementations in `week04-slides-batch*.md`:
- RNN basics
- LSTM sentiment analysis
- Attention mechanisms
- Self-attention & multi-head attention
- Positional encoding
- Transformer blocks
- GPT architecture
- Text generation with temperature
- Tokenization
- Fine-tuning techniques
- Evaluation metrics
- Fraud detection system (500 lines)
- Demand forecasting with LSTM (400 lines)
- ROI calculator (350 lines)

---

## 🚀 Quick Start

### Install Dependencies

```bash
# Core dependencies
pip install torch torchvision numpy matplotlib

# For specific examples
pip install scikit-learn pandas urllib3
```

### Run All Examples

```bash
# Test all batch1 examples
cd batch1 && for f in slide*.py; do echo "Running $f..." && python $f; done

# Test CNN
cd ../batch2 && python slide12_cnn_mnist.py

# Test Language Model
cd ../batch3 && python slide23_language_model_training.py

# Test Business Application
cd ../batch4 && python slide31_churn_prediction.py
```

---

## 📖 Learning Path

**Beginner → Intermediate → Advanced**

1. **Start Here:** `batch1/slide04_neuron.py` - Understand single neuron
2. **Next:** `batch1/slide05_activation_functions.py` - Learn activation functions
3. **Build Networks:** `batch1/slide06_simple_neural_network.py` - Create full network
4. **First Project:** `batch1/slide07_iris_classification.py` - Complete ML pipeline
5. **Computer Vision:** `batch2/slide12_cnn_mnist.py` - CNNs for images
6. **NLP:** `batch3/slide23_language_model_training.py` - Language models
7. **Business Impact:** `batch4/slide31_churn_prediction.py` - Real-world application

---

## 🎓 Course Information

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Institution:** UT Dallas  
**Semester:** Spring 2026  

**Repository:** https://github.com/appjr/GenAIForBusinessOnline

---

## ⚠️ Notes

- All code is production-quality with extensive documentation
- Each file is self-contained and runnable independently
- Synthetic data is generated where needed (no external data required except MNIST and Shakespeare)
- GPU acceleration used automatically when available
- All examples include visualization and evaluation

---

## 📞 Support

For questions or issues:
- Check the markdown slide files for additional context
- Review inline code documentation
- Refer to course materials on Canvas

---

**Last Updated:** February 13, 2026
