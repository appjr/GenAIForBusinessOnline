# Week 3: Python with ML and GenAI Toolkits - Slides Batch 2 (Slides 11-20)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 4, 2026

---

## Slide 11: Machine Learning Libraries Overview

### The Python ML Ecosystem

**Three Major ML Frameworks:**

1. **scikit-learn** 
   - Traditional ML algorithms
   - Best for: Classification, regression, clustering
   - Easy to learn, great documentation

2. **TensorFlow/Keras**
   - Deep learning framework by Google
   - Best for: Neural networks, production deployment
   - Industry standard for enterprise

3. **PyTorch**
   - Deep learning framework by Meta/Facebook
   - Best for: Research, flexibility, debugging
   - Popular in academia and cutting-edge AI

**When to Use Each:**
- **Simple ML tasks** → scikit-learn
- **Production deep learning** → TensorFlow
- **Research & experimentation** → PyTorch
- **GenAI applications** → All three + specialized tools

---

## Slide 12: scikit-learn Basics

### Traditional Machine Learning Made Easy

**Installation:**
```bash
pip install scikit-learn
```

**Common Algorithms:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Example: Classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
```

---

## Slide 13: scikit-learn Pipeline

### Building Complete ML Workflows

**ML Pipeline:**
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Train pipeline
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
```

**Cross-Validation:**
```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(pipeline, X, y, cv=5)
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

**Model Persistence:**
```python
import joblib

# Save model
joblib.dump(pipeline, 'model.pkl')

# Load model
loaded_model = joblib.load('model.pkl')
```

---

## Slide 14: TensorFlow & Keras Introduction

### Deep Learning Framework

**Installation:**
```bash
pip install tensorflow
```

**Basic Neural Network:**
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Create a simple neural network
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.3f}')
```

---

## Slide 15: PyTorch Fundamentals

### Research-Friendly Deep Learning

**Installation:**
```bash
pip install torch torchvision
```

**Basic Neural Network:**
```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model
model = SimpleNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
```

---

## Slide 16: GenAI Toolkits - OpenAI

### Working with OpenAI API

**Installation:**
```bash
pip install openai
```

**Basic Usage:**
```python
from openai import OpenAI
import os

# Initialize client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Text generation
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain machine learning in simple terms."}
    ],
    temperature=0.7,
    max_tokens=150
)

print(response.choices[0].message.content)
```

**Image Generation (DALL-E):**
```python
# Generate image
response = client.images.generate(
    model="dall-e-3",
    prompt="A futuristic AI classroom with students learning",
    size="1024x1024",
    quality="standard",
    n=1
)

image_url = response.data[0].url
print(f"Image URL: {image_url}")
```

---

## Slide 17: OpenAI Advanced Features

### Embeddings, Function Calling, and More

**Text Embeddings:**
```python
# Create embeddings for semantic search
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Machine learning is transforming business"
)

embedding = response.data[0].embedding
print(f"Embedding dimension: {len(embedding)}")
```

**Function Calling:**
```python
# Define functions for the model to use
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Dallas?"}],
    functions=functions,
    function_call="auto"
)
```

**Streaming Responses:**
```python
# Stream responses for real-time output
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a story about AI"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

---

## Slide 18: Hugging Face Transformers

### Access to Thousands of Pre-trained Models

**Installation:**
```bash
pip install transformers torch
```

**Text Generation:**
```python
from transformers import pipeline

# Load text generation pipeline
generator = pipeline('text-generation', model='gpt2')

# Generate text
result = generator(
    "The future of artificial intelligence is",
    max_length=50,
    num_return_sequences=1
)

print(result[0]['generated_text'])
```

**Sentiment Analysis:**
```python
# Sentiment analysis
sentiment = pipeline('sentiment-analysis')

result = sentiment("I love using AI tools for business!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]
```

**Question Answering:**
```python
# Question answering
qa = pipeline('question-answering')

context = "Python is a popular programming language for AI."
question = "What is Python used for?"

result = qa(question=question, context=context)
print(result['answer'])
```

---

## Slide 19: Hugging Face - Advanced Usage

### Fine-tuning and Custom Models

**Loading Specific Models:**
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Generate response
input_text = "Translate to French: Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
```

**Image Models:**
```python
from transformers import pipeline

# Image classification
classifier = pipeline("image-classification")
result = classifier("path/to/image.jpg")

# Image-to-text (image captioning)
captioner = pipeline("image-to-text")
caption = captioner("path/to/image.jpg")
```

**Model Hub:**
- Over 350,000+ models available
- Filter by task, language, license
- Easy download and deployment
- Visit: https://huggingface.co/models

---

## Slide 20: Comparing GenAI Toolkits

### Choosing the Right Tool

| Feature | OpenAI | Hugging Face | Google (Vertex AI) | Anthropic |
|---------|--------|--------------|-------------------|-----------|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Cost** | Pay-per-use | Free + Paid | Pay-per-use | Pay-per-use |
| **Customization** | Limited | High | Medium | Limited |
| **Model Variety** | GPT, DALL-E | 350K+ models | PaLM, Gemini | Claude |
| **Best For** | Quick prototyping | Research & custom | Enterprise | Safety-focused |

**Decision Framework:**
```python
# Pseudocode for choosing a toolkit
if need_latest_capabilities and budget_available:
    use = "OpenAI GPT-4"
elif need_customization or offline_deployment:
    use = "Hugging Face"
elif enterprise_integration:
    use = "Google Vertex AI or AWS Bedrock"
elif safety_and_reliability_critical:
    use = "Anthropic Claude"
```

**Key Considerations:**
- **Cost**: API calls vs. self-hosted
- **Privacy**: Data handling policies
- **Performance**: Speed and quality
- **Integration**: Existing infrastructure
- **Support**: Documentation and community

---

**End of Batch 2 (Slides 11-20)**

*Continue to Batch 3 for Hands-on Labs and Integration*
