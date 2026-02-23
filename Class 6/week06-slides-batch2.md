# Week 6: Coding with AI - Slides Batch 2 (Slides 6-10)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Topic:** Code Generation & GitHub Copilot Deep Dive

---

## Slide 6: GitHub Copilot - Getting Started

### Installation and Setup

**Step 1: Prerequisites**
```bash
# Required:
- GitHub account
- Visual Studio Code (or supported IDE)
- Active Copilot subscription ($10/month individual)

# Optional but recommended:
- Git installed
- Python 3.8+ installed
- Node.js for JavaScript projects
```

**Step 2: Install GitHub Copilot Extension**

1. Open VS Code
2. Go to Extensions (Cmd+Shift+X on Mac, Ctrl+Shift+X on Windows)
3. Search for "GitHub Copilot"
4. Click "Install"
5. Sign in with your GitHub account
6. Authorize the extension

**Step 3: Verify Installation**

```python
# Type this comment in a new Python file:
# Function to calculate the factorial of a number

# Copilot should suggest:
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

# If you see suggestions appearing, you're ready to go!
```

**Configuration Options:**

```json
// In VS Code settings.json
{
  // Enable/disable Copilot
  "github.copilot.enable": {
    "*": true,
    "yaml": false,
    "plaintext": false
  },
  
  // Inline suggestions
  "editor.inlineSuggest.enabled": true,
  
  // Auto-trigger
  "github.copilot.editor.enableAutoCompletions": true
}
```

**Keyboard Shortcuts:**

| Action | Mac | Windows/Linux |
|--------|-----|---------------|
| **Accept suggestion** | Tab | Tab |
| **Reject suggestion** | Esc | Esc |
| **Next suggestion** | Option+] | Alt+] |
| **Previous suggestion** | Option+[ | Alt+[ |
| **Open Copilot** | Option+\ | Alt+\ |
| **Trigger inline** | Option+Space | Alt+Space |

**Best Practices for Setup:**

1. **Enable Copilot for specific languages:**
   - Enable for: Python, JavaScript, Java, etc.
   - Disable for: Config files, secrets, sensitive data

2. **Configure your workspace:**
   ```bash
   # Add .gitignore to exclude sensitive files
   echo "*.env" >> .gitignore
   echo "secrets.json" >> .gitignore
   echo ".copilot-ignore" >> .gitignore
   ```

3. **Security settings:**
   - Never commit API keys or passwords
   - Review Copilot suggestions before accepting
   - Use code scanning tools alongside Copilot

---

## Slide 7: Effective Prompting for Code Generation

### Writing Comments That Get Great Code

**The Art of Copilot Prompting:**

Just like with ChatGPT, the quality of your comments determines the quality of generated code.

**Principle 1: Be Specific and Descriptive**

❌ **Bad Comment:**
```python
# Make a function
```

✅ **Good Comment:**
```python
# Function to validate email address using regex pattern
# Returns True if valid, False otherwise
# Handles common email formats including subdomains

def validate_email(email):
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
```

**Principle 2: Include Input/Output Specifications**

❌ **Bad Comment:**
```python
# Process data
```

✅ **Good Comment:**
```python
# Function to process sales data
# Input: List of dictionaries with keys: 'date', 'product', 'quantity', 'price'
# Output: Dictionary with total revenue, average order value, and top products
# Example: [{'date': '2024-01-01', 'product': 'Widget', 'quantity': 5, 'price': 10.99}]

def process_sales_data(sales_records):
    from collections import defaultdict
    
    total_revenue = sum(record['quantity'] * record['price'] for record in sales_records)
    avg_order_value = total_revenue / len(sales_records) if sales_records else 0
    
    product_sales = defaultdict(float)
    for record in sales_records:
        product_sales[record['product']] += record['quantity'] * record['price']
    
    top_products = sorted(product_sales.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'total_revenue': total_revenue,
        'average_order_value': avg_order_value,
        'top_products': top_products
    }
```

**Principle 3: Specify Error Handling**

✅ **Good Comment:**
```python
# Function to read JSON file and return parsed data
# Handles FileNotFoundError and JSONDecodeError
# Returns None if file doesn't exist or is invalid
# Logs errors to console

def read_json_file(filepath):
    import json
    import logging
    
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"File not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {filepath}: {e}")
        return None
```

**Principle 4: Include Examples in Comments**

✅ **Great Comment with Examples:**
```python
# Function to convert snake_case to camelCase
# Examples:
#   convert_to_camel_case('hello_world') -> 'helloWorld'
#   convert_to_camel_case('user_first_name') -> 'userFirstName'
#   convert_to_camel_case('api_key_value') -> 'apiKeyValue'

def convert_to_camel_case(snake_str):
    components = snake_str.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])
```

**Principle 5: Context from Existing Code**

Copilot learns from your codebase. Start a pattern and it follows:

```python
# Establish a pattern first
class UserRepository:
    def get_by_id(self, user_id):
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_by_email(self, email):
        return self.db.query(User).filter(User.email == email).first()

# Now Copilot understands your pattern - just start typing:
class ProductRepository:
    def get_by_id(self, product_id):
        # Copilot will suggest similar pattern
        return self.db.query(Product).filter(Product.id == product_id).first()
    
    def get_by_sku(self, sku):
        # Copilot continues the pattern
        return self.db.query(Product).filter(Product.sku == sku).first()
```

**Prompting Strategy: The 4-Part Comment**

```python
# 1. What: Clear function purpose
# 2. Input: Expected parameters and types
# 3. Output: Return value and type
# 4. Special: Edge cases, errors, or important notes

# Calculate monthly payment for a loan (amortization formula)
# Input: principal (float), annual_rate (float, as decimal), years (int)
# Output: monthly_payment (float, rounded to 2 decimals)
# Handles zero interest rate case, validates inputs

def calculate_monthly_payment(principal, annual_rate, years):
    if principal <= 0 or years <= 0:
        raise ValueError("Principal and years must be positive")
    
    if annual_rate == 0:
        return round(principal / (years * 12), 2)
    
    monthly_rate = annual_rate / 12
    num_payments = years * 12
    
    monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                     ((1 + monthly_rate)**num_payments - 1)
    
    return round(monthly_payment, 2)
```

**Advanced: Multi-Function Prompting**

```python
# Create a complete data validation module for user registration
# Include functions for: email validation, password strength, phone number format
# Use regex patterns and return detailed error messages

# Email validation function
def validate_email(email):
    import re
    if not email or not isinstance(email, str):
        return False, "Email must be a non-empty string"
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True, "Valid email"
    return False, "Invalid email format"

# Password strength validation (min 8 chars, 1 upper, 1 lower, 1 digit, 1 special)
def validate_password(password):
    import re
    if not password or len(password) < 8:
        return False, "Password must be at least 8 characters"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain digit"
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain special character"
    
    return True, "Valid password"

# Phone number validation (US format)
def validate_phone(phone):
    import re
    # Clean the input
    cleaned = re.sub(r'\D', '', phone)
    
    if len(cleaned) != 10:
        return False, "Phone number must be 10 digits"
    
    pattern = r'^\d{3}\d{3}\d{4}$'
    if re.match(pattern, cleaned):
        return True, "Valid phone number"
    return False, "Invalid phone number format"
```

**Prompting Patterns Cheat Sheet:**

```python
# Pattern 1: Algorithm Implementation
# "Implement [algorithm name] using [approach]"
# Implement binary search using recursive approach

# Pattern 2: Data Processing
# "Function to transform [input format] to [output format]"
# Function to transform CSV data to JSON format

# Pattern 3: API Integration
# "Function to call [API name] and [specific operation]"
# Function to call Stripe API and create a payment intent

# Pattern 4: Database Operations
# "Function to [CRUD operation] [entity] with [conditions]"
# Function to query users with pagination and filtering

# Pattern 5: Error Handling
# "Function to [operation] with error handling for [specific errors]"
# Function to parse XML with error handling for malformed data
```

---

## Slide 8: Real-World Code Generation Examples

### Practical Business Use Cases

**Example 1: Data Analysis Function**

**Scenario:** CFO needs sales trend analysis

```python
# Function to analyze monthly sales trends
# Input: DataFrame with columns: date, product_id, quantity, revenue
# Output: Dictionary with growth rates, seasonal patterns, top products
# Include month-over-month and year-over-year comparisons

import pandas as pd
import numpy as np

def analyze_sales_trends(df):
    # Ensure date column is datetime
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.to_period('M')
    
    # Monthly aggregation
    monthly_sales = df.groupby('month').agg({
        'revenue': 'sum',
        'quantity': 'sum'
    }).reset_index()
    
    # Month-over-month growth
    monthly_sales['mom_growth'] = monthly_sales['revenue'].pct_change() * 100
    
    # Year-over-year growth (if data spans multiple years)
    monthly_sales['year'] = monthly_sales['month'].dt.year
    monthly_sales['month_num'] = monthly_sales['month'].dt.month
    
    # Top products
    product_revenue = df.groupby('product_id')['revenue'].sum().sort_values(ascending=False)
    top_products = product_revenue.head(10).to_dict()
    
    # Seasonal analysis (average revenue by month across all years)
    seasonal = df.copy()
    seasonal['month_num'] = seasonal['date'].dt.month
    seasonal_pattern = seasonal.groupby('month_num')['revenue'].mean().to_dict()
    
    return {
        'monthly_sales': monthly_sales.to_dict('records'),
        'average_mom_growth': monthly_sales['mom_growth'].mean(),
        'top_products': top_products,
        'seasonal_pattern': seasonal_pattern,
        'total_revenue': df['revenue'].sum(),
        'total_quantity': df['quantity'].sum()
    }

# Example usage:
# df = pd.read_csv('sales_data.csv')
# results = analyze_sales_trends(df)
# print(f"Average MoM Growth: {results['average_mom_growth']:.2f}%")
```

**Example 2: API Client with Error Handling**

**Scenario:** Need to integrate with CRM API

```python
# Create a CRM API client class
# Features: authentication, rate limiting, retry logic, error handling
# Methods: get_customer, update_customer, create_order
# Use requests library with exponential backoff

import requests
import time
from typing import Dict, Optional
from functools import wraps

class CRMAPIClient:
    def __init__(self, api_key: str, base_url: str, max_retries: int = 3):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
    
    def _retry_with_backoff(self, func):
        """Decorator for exponential backoff retry logic"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(self.max_retries):
                try:
                    response = func(*args, **kwargs)
                    response.raise_for_status()
                    return response.json()
                except requests.exceptions.RequestException as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait_time = 2 ** attempt  # Exponential backoff
                    time.sleep(wait_time)
            return None
        return wrapper
    
    @_retry_with_backoff
    def get_customer(self, customer_id: str) -> Optional[Dict]:
        """Retrieve customer details by ID"""
        url = f"{self.base_url}/customers/{customer_id}"
        return self.session.get(url)
    
    @_retry_with_backoff
    def update_customer(self, customer_id: str, data: Dict) -> Optional[Dict]:
        """Update customer information"""
        url = f"{self.base_url}/customers/{customer_id}"
        return self.session.put(url, json=data)
    
    @_retry_with_backoff
    def create_order(self, customer_id: str, order_data: Dict) -> Optional[Dict]:
        """Create new order for customer"""
        url = f"{self.base_url}/orders"
        order_data['customer_id'] = customer_id
        return self.session.post(url, json=order_data)
    
    def close(self):
        """Close the session"""
        self.session.close()

# Usage example:
# client = CRMAPIClient(api_key="your_key", base_url="https://api.crm.com/v1")
# customer = client.get_customer("CUST123")
# print(customer)
```

**Example 3: Automated Report Generator**

**Scenario:** Weekly executive dashboard

```python
# Function to generate executive sales report
# Input: sales data, date range, format (PDF or HTML)
# Output: formatted report with charts and key metrics
# Include: revenue, growth, top customers, regional breakdown

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import base64

def generate_executive_report(sales_df, start_date, end_date, format='html'):
    """
    Generate executive sales report with visualizations
    """
    # Filter data for date range
    mask = (sales_df['date'] >= start_date) & (sales_df['date'] <= end_date)
    df = sales_df[mask].copy()
    
    # Calculate key metrics
    total_revenue = df['revenue'].sum()
    total_orders = len(df)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    
    # Previous period comparison
    period_days = (end_date - start_date).days
    prev_start = start_date - timedelta(days=period_days)
    prev_mask = (sales_df['date'] >= prev_start) & (sales_df['date'] < start_date)
    prev_revenue = sales_df[prev_mask]['revenue'].sum()
    growth_rate = ((total_revenue - prev_revenue) / prev_revenue * 100) if prev_revenue > 0 else 0
    
    # Top customers
    top_customers = df.groupby('customer_id')['revenue'].sum().sort_values(ascending=False).head(5)
    
    # Regional breakdown
    regional_sales = df.groupby('region')['revenue'].sum().sort_values(ascending=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Executive Sales Report: {start_date} to {end_date}', fontsize=16)
    
    # Daily revenue trend
    daily_revenue = df.groupby('date')['revenue'].sum()
    axes[0, 0].plot(daily_revenue.index, daily_revenue.values)
    axes[0, 0].set_title('Daily Revenue Trend')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Revenue ($)')
    
    # Top customers
    axes[0, 1].barh(range(len(top_customers)), top_customers.values)
    axes[0, 1].set_yticks(range(len(top_customers)))
    axes[0, 1].set_yticklabels(top_customers.index)
    axes[0, 1].set_title('Top 5 Customers')
    axes[0, 1].set_xlabel('Revenue ($)')
    
    # Regional breakdown
    axes[1, 0].pie(regional_sales.values, labels=regional_sales.index, autopct='%1.1f%%')
    axes[1, 0].set_title('Revenue by Region')
    
    # Key metrics table
    axes[1, 1].axis('off')
    metrics_text = f"""
    KEY METRICS
    
    Total Revenue: ${total_revenue:,.2f}
    Total Orders: {total_orders:,}
    Avg Order Value: ${avg_order_value:,.2f}
    
    Growth vs Previous Period: {growth_rate:+.1f}%
    
    Top Region: {regional_sales.index[0]}
    ({regional_sales.values[0]/total_revenue*100:.1f}% of revenue)
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                     family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Return based on format
    if format == 'html':
        # Convert plot to base64 image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=100, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode()
        
        html = f"""
        <html>
        <head><title>Executive Sales Report</title></head>
        <body>
            <h1>Executive Sales Report</h1>
            <p>Period: {start_date} to {end_date}</p>
            <img src="data:image/png;base64,{img_base64}" />
        </body>
        </html>
        """
        return html
    else:
        # Save as file
        filename = f"executive_report_{start_date}_{end_date}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        return filename

# Usage:
# df = pd.read_csv('sales_data.csv')
# df['date'] = pd.to_datetime(df['date'])
# report = generate_executive_report(df, datetime(2024, 1, 1), datetime(2024, 1, 31))
```

**Example 4: Automated Testing**

**Scenario:** Need unit tests for validation functions

```python
# Write unit tests for the email validation function
# Include tests for: valid emails, invalid formats, edge cases
# Use pytest framework with parametrize for multiple test cases

import pytest
from validation import validate_email

class TestEmailValidation:
    @pytest.mark.parametrize("email,expected", [
        ("user@example.com", (True, "Valid email")),
        ("john.doe@company.co.uk", (True, "Valid email")),
        ("test+tag@domain.org", (True, "Valid email")),
        ("invalid.email", (False, "Invalid email format")),
        ("@example.com", (False, "Invalid email format")),
        ("user@", (False, "Invalid email format")),
        ("", (False, "Email must be a non-empty string")),
        (None, (False, "Email must be a non-empty string")),
        ("user@.com", (False, "Invalid email format")),
        ("user name@example.com", (False, "Invalid email format")),
    ])
    def test_email_validation(self, email, expected):
        """Test email validation with various inputs"""
        assert validate_email(email) == expected
    
    def test_email_with_subdomain(self):
        """Test email with multiple subdomains"""
        result = validate_email("user@mail.company.example.com")
        assert result[0] is True
    
    def test_email_case_insensitive(self):
        """Email validation should work regardless of case"""
        assert validate_email("User@Example.COM")[0] is True
    
    def test_email_max_length(self):
        """Test very long email addresses"""
        long_email = "a" * 64 + "@" + "b" * 63 + ".com"
        assert validate_email(long_email)[0] is True

# Run tests with: pytest test_validation.py -v
```

**Time Saved: Copilot generated these 4 complete examples in under 2 minutes!**

---

## Slide 9: Code Completion and Refactoring

### Making Existing Code Better

**Use Case 1: Refactoring Legacy Code**

**Before (Legacy Code):**
```python
def process_data(data):
    result = []
    for item in data:
        if item['status'] == 'active':
            if item['value'] > 100:
                if item['category'] == 'A':
                    result.append(item)
    return result
```

**Prompt for Copilot:**
```python
# Refactor the above function to be more pythonic
# Use list comprehension and reduce nesting
# Add type hints and docstring

def process_data(data: list[dict]) -> list[dict]:
    """
    Filter active items with value > 100 in category A.
    
    Args:
        data: List of dictionaries with keys: status, value, category
    
    Returns:
        Filtered list of items meeting all criteria
    """
    return [
        item for item in data
        if item.get('status') == 'active'
        and item.get('value', 0) > 100
        and item.get('category') == 'A'
    ]
```

**Use Case 2: Adding Error Handling**

**Original:**
```python
def read_config(filename):
    with open(filename) as f:
        return json.load(f)
```

**Copilot Enhancement (just add comment):**
```python
# Add comprehensive error handling for file operations and JSON parsing
def read_config(filename: str) -> dict:
    """
    Read and parse JSON configuration file with error handling.
    
    Args:
        filename: Path to JSON config file
    
    Returns:
        Parsed configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    import json
    import logging
    
    try:
        with open(filename, 'r') as f:
            config = json.load(f)
            logging.info(f"Successfully loaded config from {filename}")
            return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {filename}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {filename}: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error reading {filename}: {e}")
        raise
```

**Use Case 3: Performance Optimization**

**Original (Slow):**
```python
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates
```

**Copilot Optimization:**
```python
# Optimize the above function using set operations for O(n) complexity
def find_duplicates(items: list) -> list:
    """
    Find duplicate items efficiently using set operations.
    
    Time complexity: O(n) vs O(n²) in original
    
    Args:
        items: List of items to check for duplicates
    
    Returns:
        List of items that appear more than once
    """
    from collections import Counter
    
    counts = Counter(items)
    return [item for item, count in counts.items() if count > 1]
```

**Use Case 4: Adding Documentation**

**Prompt:** Select your function and type:

```python
# Add comprehensive docstring with examples and type hints

# Copilot generates:
def calculate_discount(price: float, discount_percent: float, is_member: bool = False) -> float:
    """
    Calculate final price after applying discount and membership benefits.
    
    Args:
        price: Original price of the item
        discount_percent: Discount percentage (0-100)
        is_member: Whether customer is a loyalty member (gets additional 5% off)
    
    Returns:
        Final price after all discounts applied
    
    Examples:
        >>> calculate_discount(100, 10)
        90.0
        >>> calculate_discount(100, 10, is_member=True)
        85.5
        >>> calculate_discount(50, 20, is_member=True)
        38.0
    
    Raises:
        ValueError: If price is negative or discount_percent not in range 0-100
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount percent must be between 0 and 100")
    
    discounted_price = price * (1 - discount_percent / 100)
    
    if is_member:
        discounted_price *= 0.95  # Additional 5% member discount
    
    return round(discounted_price, 2)
```

**Refactoring Patterns Copilot Excels At:**

1. **Converting loops to comprehensions**
2. **Adding type hints**
3. **Error handling improvements**
4. **Documentation generation**
5. **Breaking large functions into smaller ones**
6. **Modernizing syntax (e.g., f-strings, walrus operator)**
7. **Performance optimizations**

---

## Slide 10: GitHub Copilot Chat & Interactive Coding

### Conversational AI Inside Your IDE

**New Feature: GitHub Copilot Chat (2023)**

Brings ChatGPT-like experience directly into VS Code!

**Accessing Copilot Chat:**

1. Click the chat icon in VS Code sidebar
2. Or use keyboard shortcut: `Cmd+I` (Mac) or `Ctrl+I` (Windows)
3. Select code and right-click → "Copilot" → "Explain this"

**Key Features:**

✅ **Explain Code:** Understand complex code snippets  
✅ **Fix Bugs:** Identify and fix issues  
✅ **Generate Tests:** Create unit tests automatically  
✅ **Refactor:** Improve code structure  
✅ **Document:** Add docstrings and comments  

**Example 1: Explain Complex Code**

```python
# Select this code and ask Copilot Chat: "Explain what this does"

@lru_cache(maxsize=None)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
```

**Copilot Chat Response:**
```
This function implements the Fibonacci sequence with memoization:

1. @lru_cache decorator caches results to avoid redundant calculations
2. Base case: returns n for n < 2 (0 and 1)
3. Recursive case: sum of two previous Fibonacci numbers
4. maxsize=None means unlimited cache size

Without caching, this would be O(2^n). With caching, it's O(n).
Time saved increases exponentially for larger n values.
```

**Example 2: Fix Bugs**

```python
# Code with bug
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

# Ask Copilot: "What's wrong with this code and how can I fix it?"
```

**Copilot Chat Response:**
```
Bug identified: Division by zero error if numbers list is empty.

Fixed version:

def calculate_average(numbers):
    """Calculate average of numbers, returns 0 for empty list"""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

Also improved: Using sum() is more Pythonic than manual loop.
```

**Example 3: Generate Tests**

**Prompt in Copilot Chat:**
"Generate pytest tests for this function with edge cases"

```python
def validate_password(password):
    if len(password) < 8:
        return False
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    return has_upper and has_lower and has_digit
```

**Copilot Generates:**
```python
import pytest

class TestPasswordValidation:
    def test_valid_password(self):
        assert validate_password("Password123") is True
    
    def test_too_short(self):
        assert validate_password("Pass1") is False
    
    def test_no_uppercase(self):
        assert validate_password("password123") is False
    
    def test_no_lowercase(self):
        assert validate_password("PASSWORD123") is False
    
    def test_no_digit(self):
        assert validate_password("Password") is False
    
    def test_exactly_8_chars(self):
        assert validate_password("Pass123!") is True
    
    def test_empty_string(self):
        assert validate_password("") is False
    
    def test_special_characters_allowed(self):
        assert validate_password("Pass@123") is True
```

**Example 4: Architectural Questions**

**Prompt:**
"I need to build a REST API for managing todo items. What's the best project structure using FastAPI?"

**Copilot Response:**
```
Recommended FastAPI project structure:

project/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app instance
│   ├── models/              # Database models
│   │   ├── __init__.py
│   │   └── todo.py
│   ├── schemas/             # Pydantic schemas
│   │   ├── __init__.py
│   │   └── todo.py
│   ├── routers/             # API routes
│   │   ├── __init__.py
│   │   └── todos.py
│   ├── database.py          # Database connection
│   └── dependencies.py      # Shared dependencies
├── tests/
│   ├── __init__.py
│   └── test_todos.py
├── requirements.txt
└── README.md

Would you like me to generate the code for any of these files?
```

**Copilot Chat Commands:**

| Command | Purpose | Example |
|---------|---------|---------|
| `/explain` | Explain selected code | `/explain` |
| `/fix` | Find and fix bugs | `/fix` |
| `/tests` | Generate tests | `/tests` |
| `/doc` | Add documentation | `/doc` |
| `/help` | Show available commands | `/help` |

**Best Practices for Copilot Chat:**

1. **Be specific in your questions**
   - ❌ "Fix this"
   - ✅ "This function throws a TypeError. Can you identify why and suggest a fix?"

2. **Provide context**
   - ❌ "Write a function"
   - ✅ "Write a function to parse CSV files for our sales data, which has columns: date, product_id, quantity, revenue"

3. **Iterate on responses**
   - Ask follow-up questions
   - Request alternatives
   - Ask for explanations

4. **Use for learning**
   - "Why is this approach better?"
   - "What are the trade-offs?"
   - "Can you explain the time complexity?"

**Comparison: Copilot Inline vs. Copilot Chat**

| Feature | Inline Suggestions | Copilot Chat |
|---------|-------------------|--------------|
| **Speed** | Instant | 2-5 seconds |
| **Context** | Current file | Entire codebase |
| **Interaction** | Accept/reject | Conversational |
| **Best for** | Code completion | Problem solving |
| **Explanation** | No | Yes |
| **Alternatives** | Limited | Multiple options |

**When to use each:**

- **Inline:** Writing code, autocomplete, known patterns
- **Chat:** Debugging, learning, architecture, complex problems

---

**End of Batch 2 (Slides 6-10)**

*Continue to Batch 3 for Conversational Coding & Debugging with ChatGPT/Claude (Slides 11-15)*
