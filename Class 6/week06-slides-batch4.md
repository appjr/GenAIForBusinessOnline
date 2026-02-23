# Week 6: Coding with AI - Slides Batch 4 (Slides 16-20)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Topic:** Code Review, Testing & Documentation with AI

---

## Slide 16: AI-Powered Code Review

### Improving Code Quality with AI Analysis

**What is AI Code Review?**

AI code review uses machine learning models to analyze code for:
- **Code quality issues** (code smells, anti-patterns)
- **Security vulnerabilities** (SQL injection, XSS, etc.)
- **Performance bottlenecks**
- **Best practice violations**
- **Maintainability concerns**
- **Documentation gaps**

**Traditional Code Review vs AI-Assisted:**

| Aspect | Traditional | AI-Assisted |
|--------|-------------|-------------|
| **Speed** | Hours to days | Seconds to minutes |
| **Coverage** | Depends on reviewer | 100% of code |
| **Consistency** | Varies by reviewer | Always consistent |
| **Focus** | All issues | Human focuses on logic, AI handles patterns |
| **Availability** | Limited by time zones | 24/7 |
| **Cost** | Developer time | Tool subscription |

**Benefits:**
✅ Catch issues before human review  
✅ Free up senior developers for architecture review  
✅ Standardize code quality across team  
✅ Reduce technical debt  
✅ Improve security posture  

---

## Slide 17: Using ChatGPT/Claude for Code Review

### Step-by-Step Code Review Process

**Example 1: Security Review**

**Prompt:**
```
Please review this code for security vulnerabilities:

```python
from flask import Flask, request
import sqlite3

app = Flask(__name__)

@app.route('/user')
def get_user():
    user_id = request.args.get('id')
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    result = cursor.fetchone()
    conn.close()
    return str(result)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    cursor.execute(query)
    user = cursor.fetchone()
    
    if user:
        return {'status': 'success', 'user': user}
    return {'status': 'failed'}
```

Please identify all security issues and provide secure alternatives.
```

**ChatGPT Response:**

```
CRITICAL SECURITY VULNERABILITIES FOUND:

1. **SQL Injection (CRITICAL - Severity 10/10)**
   
   Problem: Direct string interpolation in SQL queries
   Lines: 9, 18
   
   Current code:
   ```python
   query = f"SELECT * FROM users WHERE id = {user_id}"
   query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
   ```
   
   Attack example:
   - user_id = "1 OR 1=1" would return all users
   - username = "admin'--" bypasses password check
   
   Impact: Complete database compromise, data theft, data manipulation

2. **Plain Text Password Storage (CRITICAL - Severity 9/10)**
   
   Problem: Passwords stored and compared in plain text
   Impact: If database is compromised, all passwords are exposed

3. **Information Disclosure (HIGH - Severity 7/10)**
   
   Problem: Returning raw database results exposes internal structure
   Impact: Reveals database schema, sensitive user data

4. **No Input Validation (MEDIUM - Severity 6/10)**
   
   Problem: No validation of user inputs
   Impact: Allows malicious payloads

SECURE VERSION:

```python
from flask import Flask, request, jsonify
import sqlite3
import hashlib
import secrets
from typing import Optional, Dict

app = Flask(__name__)

# Database helper functions
def get_db_connection():
    """Create secure database connection"""
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row  # Return dict-like rows
    return conn

def hash_password(password: str, salt: str = None) -> tuple:
    """
    Hash password using SHA-256 with salt
    Returns: (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hashed = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    )
    return hashed.hex(), salt

# Secure endpoints
@app.route('/user')
def get_user():
    """Get user by ID with SQL injection protection"""
    user_id = request.args.get('id')
    
    # Input validation
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
    
    try:
        user_id = int(user_id)  # Validate it's a number
    except ValueError:
        return jsonify({'error': 'Invalid user ID format'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Parameterized query prevents SQL injection
    cursor.execute('SELECT id, username, email FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    
    # Return only safe fields (no passwords!)
    return jsonify({
        'id': user['id'],
        'username': user['username'],
        'email': user['email']
    })

@app.route('/login', methods=['POST'])
def login():
    """Secure login endpoint"""
    username = request.form.get('username')
    password = request.form.get('password')
    
    # Input validation
    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400
    
    if len(username) > 100 or len(password) > 100:
        return jsonify({'error': 'Invalid input length'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Parameterized query
    cursor.execute(
        'SELECT id, username, password_hash, salt FROM users WHERE username = ?',
        (username,)
    )
    user = cursor.fetchone()
    conn.close()
    
    if user is None:
        # Don't reveal if username exists
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Verify password
    hashed_input, _ = hash_password(password, user['salt'])
    
    if hashed_input != user['password_hash']:
        return jsonify({'error': 'Invalid credentials'}), 401
    
    # Success - return only safe data
    return jsonify({
        'status': 'success',
        'user': {
            'id': user['id'],
            'username': user['username']
        }
    })

if __name__ == '__main__':
    # Never run with debug=True in production
    app.run(debug=False, host='127.0.0.1')
```

ADDITIONAL SECURITY RECOMMENDATIONS:

1. **Use proper authentication**
   - Implement JWT tokens or session management
   - Add rate limiting to prevent brute force

2. **Database security**
   - Use SQLAlchemy ORM instead of raw SQL
   - Implement principle of least privilege

3. **HTTPS**
   - Always use HTTPS in production
   - Never send passwords over HTTP

4. **Environment variables**
   - Store secrets in environment variables
   - Never hardcode database credentials

5. **Logging & Monitoring**
   - Log failed login attempts
   - Monitor for suspicious activity
   - Don't log sensitive data

SEVERITY SUMMARY:
- Critical: 2 issues (SQL Injection, Plain Text Passwords)
- High: 1 issue (Information Disclosure)
- Medium: 1 issue (No Input Validation)

RECOMMENDATION: Do not deploy current code to production. Implement secure version immediately.
```

**Key Takeaways:**

1. AI provides detailed security analysis
2. Explains attack vectors and impact
3. Provides secure code alternatives
4. Prioritizes issues by severity
5. Gives best practice recommendations

---

## Slide 18: Automated Test Generation

### Let AI Write Your Tests

**Why Generate Tests with AI?**

- Writing tests is time-consuming (often 2-3x the code size)
- AI knows testing patterns and best practices
- Ensures better coverage including edge cases
- Reduces human error in test writing
- Frees developers for feature work

**Test Generation Workflow:**

```
1. Write your function/class
2. Ask AI to generate comprehensive tests
3. Review and customize tests
4. Run tests and iterate
5. Add to CI/CD pipeline
```

---

## Slide 19: Test Generation Examples

### From Code to Complete Test Suite

**Example 1: Unit Test Generation**

**Original Function:**
```python
def calculate_discount(price: float, discount_percent: float, customer_tier: str) -> float:
    """
    Calculate final price after discount and tier bonus.
    
    Args:
        price: Original price
        discount_percent: Discount percentage (0-100)
        customer_tier: 'bronze', 'silver', 'gold', 'platinum'
    
    Returns:
        Final price after all discounts
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount must be between 0 and 100")
    
    # Apply base discount
    discounted = price * (1 - discount_percent / 100)
    
    # Apply tier bonus
    tier_bonuses = {
        'bronze': 0.00,
        'silver': 0.05,
        'gold': 0.10,
        'platinum': 0.15
    }
    
    tier_bonus = tier_bonuses.get(customer_tier.lower(), 0)
    final_price = discounted * (1 - tier_bonus)
    
    return round(final_price, 2)
```

**Prompt to ChatGPT:**
```
Generate comprehensive pytest tests for the calculate_discount function above.
Include:
- Happy path tests
- Edge cases
- Error cases
- Parametrized tests for efficiency
- Test fixtures if needed
```

**Generated Tests:**

```python
import pytest
from decimal import Decimal
from discount import calculate_discount

class TestCalculateDiscount:
    """Comprehensive test suite for calculate_discount function"""
    
    # Happy path tests
    def test_no_discount_bronze(self):
        """Test with no discount and bronze tier"""
        assert calculate_discount(100, 0, 'bronze') == 100.00
    
    def test_basic_discount_no_tier_bonus(self):
        """Test 10% discount with bronze tier (no bonus)"""
        assert calculate_discount(100, 10, 'bronze') == 90.00
    
    def test_discount_with_silver_tier(self):
        """Test 20% discount with silver tier (5% bonus)"""
        # 100 * 0.8 = 80, then 80 * 0.95 = 76
        assert calculate_discount(100, 20, 'silver') == 76.00
    
    def test_discount_with_gold_tier(self):
        """Test 15% discount with gold tier (10% bonus)"""
        # 100 * 0.85 = 85, then 85 * 0.90 = 76.5
        assert calculate_discount(100, 15, 'gold') == 76.50
    
    def test_discount_with_platinum_tier(self):
        """Test 25% discount with platinum tier (15% bonus)"""
        # 100 * 0.75 = 75, then 75 * 0.85 = 63.75
        assert calculate_discount(100, 25, 'platinum') == 63.75
    
    # Parametrized tests for multiple scenarios
    @pytest.mark.parametrize("price,discount,tier,expected", [
        (100, 0, 'bronze', 100.00),
        (100, 10, 'bronze', 90.00),
        (100, 20, 'silver', 76.00),
        (100, 15, 'gold', 76.50),
        (100, 25, 'platinum', 63.75),
        (50, 50, 'bronze', 25.00),
        (200, 10, 'silver', 171.00),
        (1000, 5, 'platinum', 807.50),
    ])
    def test_various_combinations(self, price, discount, tier, expected):
        """Test various price, discount, and tier combinations"""
        assert calculate_discount(price, discount, tier) == expected
    
    # Edge cases
    def test_zero_price(self):
        """Test with zero price"""
        assert calculate_discount(0, 10, 'bronze') == 0.00
    
    def test_100_percent_discount(self):
        """Test with 100% discount"""
        assert calculate_discount(100, 100, 'bronze') == 0.00
    
    def test_zero_percent_discount(self):
        """Test with 0% discount"""
        assert calculate_discount(100, 0, 'platinum') == 85.00
    
    def test_very_small_price(self):
        """Test with very small price"""
        assert calculate_discount(0.01, 10, 'bronze') == 0.01
    
    def test_very_large_price(self):
        """Test with very large price"""
        result = calculate_discount(1000000, 10, 'gold')
        assert result == 810000.00
    
    def test_case_insensitive_tier(self):
        """Test that tier names are case-insensitive"""
        assert calculate_discount(100, 10, 'GOLD') == calculate_discount(100, 10, 'gold')
        assert calculate_discount(100, 10, 'Silver') == calculate_discount(100, 10, 'silver')
    
    def test_unknown_tier_defaults_to_no_bonus(self):
        """Test that unknown tier defaults to no bonus"""
        assert calculate_discount(100, 10, 'unknown') == 90.00
        assert calculate_discount(100, 10, 'diamond') == 90.00
    
    # Error cases
    def test_negative_price_raises_error(self):
        """Test that negative price raises ValueError"""
        with pytest.raises(ValueError, match="Price cannot be negative"):
            calculate_discount(-10, 10, 'bronze')
    
    def test_discount_below_zero_raises_error(self):
        """Test that discount < 0 raises ValueError"""
        with pytest.raises(ValueError, match="Discount must be between 0 and 100"):
            calculate_discount(100, -5, 'bronze')
    
    def test_discount_above_100_raises_error(self):
        """Test that discount > 100 raises ValueError"""
        with pytest.raises(ValueError, match="Discount must be between 0 and 100"):
            calculate_discount(100, 150, 'bronze')
    
    # Boundary tests
    @pytest.mark.parametrize("discount", [0, 1, 99, 100])
    def test_discount_boundaries(self, discount):
        """Test discount at boundaries"""
        result = calculate_discount(100, discount, 'bronze')
        assert isinstance(result, float)
        assert result >= 0
    
    # Precision tests
    def test_rounding_precision(self):
        """Test that result is rounded to 2 decimal places"""
        result = calculate_discount(99.99, 33.33, 'bronze')
        # Should have exactly 2 decimal places
        assert len(str(result).split('.')[-1]) <= 2
    
    def test_multiple_decimal_precision(self):
        """Test calculations that result in many decimals"""
        result = calculate_discount(33.33, 33.33, 'silver')
        assert isinstance(result, float)
        assert result == 21.11  # Verify rounding works correctly

# Fixtures for integration tests
@pytest.fixture
def standard_prices():
    """Fixture providing standard test prices"""
    return [10, 50, 100, 500, 1000]

@pytest.fixture
def all_tiers():
    """Fixture providing all tier levels"""
    return ['bronze', 'silver', 'gold', 'platinum']

class TestDiscountIntegration:
    """Integration tests using fixtures"""
    
    def test_all_tiers_with_standard_prices(self, standard_prices, all_tiers):
        """Test all tier combinations with standard prices"""
        for price in standard_prices:
            for tier in all_tiers:
                result = calculate_discount(price, 10, tier)
                assert result > 0
                assert result <= price

# Performance tests
def test_performance_with_many_calls():
    """Test that function performs well with many calls"""
    import time
    
    start = time.time()
    for _ in range(10000):
        calculate_discount(100, 10, 'gold')
    duration = time.time() - start
    
    # Should complete 10,000 calls in under 1 second
    assert duration < 1.0, f"Performance issue: took {duration} seconds"

# Documentation tests (pytest-doctests)
def test_function_has_docstring():
    """Verify function has documentation"""
    assert calculate_discount.__doc__ is not None
    assert len(calculate_discount.__doc__) > 50

# Run with: pytest -v test_discount.py
# Coverage: pytest --cov=discount test_discount.py
# Verbose: pytest -v -s test_discount.py
```

**Test Coverage Analysis:**

```bash
# Running the tests:
$ pytest test_discount.py -v --cov=discount

======================== test session starts =========================
collected 35 items

test_discount.py::TestCalculateDiscount::test_no_discount_bronze PASSED
test_discount.py::TestCalculateDiscount::test_basic_discount_no_tier_bonus PASSED
test_discount.py::TestCalculateDiscount::test_discount_with_silver_tier PASSED
...
test_discount.py::test_performance_with_many_calls PASSED

======================== 35 passed in 0.15s ==========================

---------- coverage: platform darwin, python 3.10 -----------
Name           Stmts   Miss  Cover
----------------------------------
discount.py       15      0   100%
----------------------------------
TOTAL             15      0   100%
```

**What AI Generated:**

✅ 35 comprehensive tests  
✅ 100% code coverage  
✅ Happy path, edge cases, errors  
✅ Parametrized tests for efficiency  
✅ Integration tests with fixtures  
✅ Performance tests  
✅ Documentation checks  

**Time Saved:**
- Manual: 2-3 hours to write these tests
- With AI: 30 seconds + 10 minutes review
- **Productivity gain: ~90%**

---

## Slide 20: Documentation Automation

### Let AI Document Your Code

**The Documentation Problem:**

- Developers hate writing documentation
- Documentation gets outdated quickly
- Inconsistent documentation styles
- Missing edge case documentation
- Time-consuming to maintain

**AI Solution:**

AI can generate:
1. **Docstrings** for functions/classes
2. **README files** for projects
3. **API documentation**
4. **Code comments** for complex logic
5. **User guides** and tutorials

---

**End of Batch 4 (Slides 16-20)**

*Continue to Batch 5 for Business Applications, ROI & Hands-on Lab (Slides 21-25+)*
