# Week 3: Python with ML and GenAI Toolkits - Slides Batch 1 (Slides 1-10)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 4, 2026  
**Duration:** 2.5 hours

---

## Slide 1: Week 3 Title Slide

### Python with ML and GenAI Toolkits

**Learning Goals:**
- Master essential Python libraries for ML and GenAI
- Understand NumPy, Pandas, and data manipulation
- Explore ML frameworks (scikit-learn, TensorFlow, PyTorch)
- Get hands-on with GenAI toolkits and APIs

---

## Slide 2: Today's Agenda

### Class Overview

1. **Review Python Basics** (10 min)
2. **NumPy & Pandas Fundamentals** (30 min)
3. **Machine Learning Libraries** (30 min)
4. **Break** (10 min)
5. **GenAI Toolkits Overview** (30 min)
6. **Hands-on Lab: Building Your First AI App** (40 min)
7. **Assignment & Q&A** (10 min)

---

## Slide 3: Learning Objectives

### By the End of This Class, You Will:

✅ **Understand** core Python libraries for data science and AI  
✅ **Work with** NumPy arrays and Pandas DataFrames  
✅ **Explore** popular ML frameworks and their use cases  
✅ **Install and configure** GenAI toolkits (OpenAI, Hugging Face)  
✅ **Build** a simple GenAI application using Python  
✅ **Apply** best practices for library management and environments

---

## Slide 4: Quick Python Review

### Essential Python Concepts Recap

**Data Types:**
```python
# Numbers, strings, lists, dictionaries
age = 25
name = "AI Engineer"
skills = ["Python", "ML", "GenAI"]
config = {"model": "gpt-4", "temp": 0.7}
```

**Functions:**
```python
def greet_user(name):
    return f"Hello, {name}! Welcome to GenAI class."

message = greet_user("Student")
print(message)
```

**Key Concepts:**
- Variables and data structures
- Control flow (if/else, loops)
- Functions and modules
- Error handling with try/except

---

## Slide 5: Python Package Management

### Managing Libraries and Dependencies

**pip - Python Package Installer:**
```bash
# Install a package
pip install numpy

# Install specific version
pip install pandas==2.0.0

# Install from requirements file
pip install -r requirements.txt

# List installed packages
pip list
```

**Virtual Environments:**
```bash
# Create virtual environment
python -m venv genai_env

# Activate (macOS/Linux)
source genai_env/bin/activate

# Activate (Windows)
genai_env\Scripts\activate

# Deactivate
deactivate
```

**💡 Best Practice:** Always use virtual environments for projects!

---

## Slide 6: NumPy - Numerical Computing

### The Foundation of Scientific Python

**What is NumPy?**
- Fundamental package for numerical computing
- Powerful N-dimensional array object
- Broadcasting functions for fast operations
- Essential for ML and data science

**Installation:**
```bash
pip install numpy
```

**Basic Usage:**
```python
import numpy as np

# Create arrays
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Array operations
print(arr * 2)           # [2, 4, 6, 8, 10]
print(arr.mean())        # 3.0
print(arr.std())         # 1.41421356

# Generate data
random_data = np.random.randn(1000, 10)
zeros = np.zeros((3, 3))
ones = np.ones((2, 4))
```

---

## Slide 7: NumPy - Advanced Operations

### Powerful Array Manipulation

**Reshaping and Slicing:**
```python
import numpy as np

# Create and reshape
arr = np.arange(12)              # [0, 1, 2, ..., 11]
reshaped = arr.reshape(3, 4)     # 3x4 matrix

# Slicing
row = reshaped[0, :]             # First row
col = reshaped[:, 1]             # Second column
subset = reshaped[0:2, 1:3]      # Submatrix
```

**Broadcasting:**
```python
# Add scalar to array (broadcasting)
matrix = np.array([[1, 2], [3, 4]])
result = matrix + 10
# Result: [[11, 12], [13, 14]]

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
product = np.dot(A, B)           # Matrix multiplication
```

**Why NumPy Matters for AI:**
- Fast vectorized operations
- Memory efficient
- Foundation for TensorFlow, PyTorch, scikit-learn

---

## Slide 8: Pandas - Data Manipulation

### Working with Structured Data

**What is Pandas?**
- Data manipulation and analysis library
- DataFrame: 2D labeled data structure
- Built on top of NumPy
- Essential for data preprocessing

**Installation:**
```bash
pip install pandas
```

**Creating DataFrames:**
```python
import pandas as pd

# From dictionary
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'salary': [50000, 60000, 75000]
}
df = pd.DataFrame(data)

# From CSV file
df = pd.read_csv('data.csv')

# From Excel
df = pd.read_excel('data.xlsx')

# View data
print(df.head())        # First 5 rows
print(df.describe())    # Statistical summary
print(df.info())        # Data types and info
```

---

## Slide 9: Pandas - Data Operations

### Cleaning, Filtering, and Transforming Data

**Data Selection:**
```python
import pandas as pd

# Load sample data
df = pd.read_csv('employees.csv')

# Select columns
names = df['name']
subset = df[['name', 'age']]

# Filter rows
seniors = df[df['age'] > 30]
high_earners = df[df['salary'] >= 60000]
combined = df[(df['age'] > 25) & (df['salary'] > 50000)]
```

**Data Cleaning:**
```python
# Handle missing values
df.dropna()                    # Remove rows with NaN
df.fillna(0)                   # Fill NaN with 0
df['age'].fillna(df['age'].mean())  # Fill with mean

# Remove duplicates
df.drop_duplicates()

# Rename columns
df.rename(columns={'old_name': 'new_name'})
```

**Aggregation:**
```python
# Group by and aggregate
avg_salary_by_dept = df.groupby('department')['salary'].mean()
counts = df['category'].value_counts()
```

---

## Slide 10: Pandas for AI/ML

### Preparing Data for Machine Learning

**Feature Engineering:**
```python
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('customer_data.csv')

# Create new features
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 50, 100], 
                          labels=['Young', 'Middle', 'Senior'])
df['log_income'] = np.log(df['income'] + 1)

# Encode categorical variables
df_encoded = pd.get_dummies(df, columns=['category'])

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['income', 'age']] = scaler.fit_transform(df[['income', 'age']])
```

**Train/Test Split:**
```python
from sklearn.model_selection import train_test_split

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Create train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

**💡 Key Point:** Clean data = Better AI models!

---

**End of Batch 1 (Slides 1-10)**

*Continue to Batch 2 for ML Frameworks and GenAI Toolkits*
