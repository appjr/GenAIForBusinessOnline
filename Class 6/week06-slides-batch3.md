# Week 6: Coding with AI - Slides Batch 3 (Slides 11-15)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Topic:** Conversational Coding & Debugging with ChatGPT/Claude

---

## Slide 11: ChatGPT vs Claude for Coding

### Choosing the Right Conversational AI

**Break Time: 10 Minutes** ☕

*When we return, we'll dive into using ChatGPT and Claude for complex coding tasks*

---

## Slide 12: ChatGPT/Claude for Complex Problems

### When to Use Conversational AI vs IDE Tools

**The Workflow Difference:**

**GitHub Copilot:**
- Real-time, in-IDE suggestions
- Best for: writing code as you think
- Context: current file and nearby files
- Speed: instant
- Workflow: type → see suggestion → accept/reject

**ChatGPT/Claude:**
- Conversational, web-based (or API)
- Best for: complex problems, learning, architecture
- Context: large codebases (Claude: 100K+ tokens)
- Speed: 2-10 seconds per response
- Workflow: describe problem → get solution → copy/paste → iterate

**When to Use ChatGPT/Claude:**

1. **Architectural Decisions**
   - "What's the best database for a high-traffic analytics dashboard?"
   - "Should I use microservices or monolith for this application?"

2. **Learning New Technologies**
   - "Explain how React hooks work with examples"
   - "Teach me async/await in Python step by step"

3. **Debugging Complex Issues**
   - Paste error stack trace and get explanation
   - "This code works in dev but fails in production. Why?"

4. **Code Review & Optimization**
   - Paste entire function/class for analysis
   - "Review this code for security vulnerabilities"

5. **Writing Complete Features**
   - "Build a complete authentication system with JWT"
   - "Create a data pipeline for processing CSV files"

6. **Algorithm Design**
   - "Design an algorithm to find the shortest path considering traffic"
   - "How do I implement rate limiting efficiently?"

---

## Slide 13: Effective Prompting for ChatGPT/Claude

### Getting Better Code Answers

**Prompt Engineering Principles:**

**Principle 1: Provide Context**

❌ **Bad Prompt:**
```
Write a function to process data
```

✅ **Good Prompt:**
```
I'm building a sales analytics dashboard using Python and Pandas.
I need a function that:
- Takes a CSV file of sales transactions
- Columns: date, customer_id, product_id, quantity, price, region
- Calculates monthly revenue by region
- Returns a DataFrame with columns: month, region, revenue, growth_rate
- Growth rate should be month-over-month for each region

Please include error handling for missing files and malformed data.
```

**Principle 2: Specify Requirements**

✅ **Detailed Requirements:**
```
Create a REST API endpoint for user registration with these requirements:

Technology Stack:
- FastAPI framework
- PostgreSQL database
- SQLAlchemy ORM
- Pydantic for validation

Requirements:
1. Endpoint: POST /api/v1/users/register
2. Input: email, password, first_name, last_name
3. Validation:
   - Email must be valid format and unique
   - Password min 8 chars, 1 uppercase, 1 number, 1 special char
   - Names must be non-empty strings
4. Security:
   - Hash password with bcrypt
   - Return JWT token on success
5. Error Handling:
   - 400 for validation errors
   - 409 for duplicate email
   - 500 for server errors
6. Response format:
   - Success: {user_id, email, token}
   - Error: {error, details, status_code}

Please include:
- Complete endpoint code
- Database model
- Pydantic schemas
- Unit tests
```

**Principle 3: Show Examples**

✅ **With Examples:**
```
I need a function to parse log files and extract errors.

Example log format:
```
2024-01-15 10:30:45 INFO User login successful: user123
2024-01-15 10:31:12 ERROR Database connection failed: timeout after 30s
2024-01-15 10:31:45 WARNING High memory usage: 85%
2024-01-15 10:32:01 ERROR API request failed: 500 Internal Server Error
```

Expected output:
```python
[
    {
        'timestamp': '2024-01-15 10:31:12',
        'level': 'ERROR',
        'message': 'Database connection failed: timeout after 30s'
    },
    {
        'timestamp': '2024-01-15 10:32:01',
        'level': 'ERROR',
        'message': 'API request failed: 500 Internal Server Error'
    }
]
```

Please handle different date formats and malformed log lines.
```

**Principle 4: Iterative Refinement**

First prompt:
```
Create a Python class for managing a cache
```

Response: *Gets basic cache implementation*

Second prompt (refinement):
```
Great! Can you enhance this to:
1. Add TTL (time-to-live) for each cache entry
2. Implement LRU eviction when max size reached
3. Add statistics tracking (hits, misses, evictions)
4. Make it thread-safe
5. Add optional persistence to Redis
```

Response: *Gets enhanced version*

Third prompt (further refinement):
```
Perfect! Now add:
1. Type hints for all methods
2. Comprehensive docstrings
3. Unit tests with pytest
4. Example usage in comments
```

**Principle 5: Ask for Explanations**

```
Please explain:
1. Why you chose this approach
2. Time and space complexity
3. Potential edge cases
4. Alternative solutions and trade-offs
```

---

## Slide 14: Real-World ChatGPT/Claude Examples

### Complete Solutions to Complex Problems

**Example 1: Data Pipeline Architecture**

**Prompt:**
```
I need to design a data pipeline for processing e-commerce transactions.

Current Situation:
- Receiving 10,000 transactions per minute from our API
- Need to: validate, enrich with customer data, calculate metrics, store in database
- Must handle failures gracefully and retry
- Need real-time dashboard showing metrics

Tech Stack: Python, PostgreSQL, Redis, Docker
Budget: Can use AWS services if needed

Please provide:
1. Architecture diagram (in text/ASCII)
2. Technology recommendations
3. Python code for key components
4. Scaling considerations
```

**Claude Response:**

```
I'll design a robust, scalable data pipeline for your e-commerce transactions.

ARCHITECTURE:
```
┌─────────────┐
│   API       │
│ (FastAPI)   │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────┐
│   Message Queue (Redis Streams/RabbitMQ)│
└──────┬──────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│      Worker Pool (Celery Workers)        │
│  ┌─────────┬─────────┬─────────┐        │
│  │Worker 1 │Worker 2 │Worker 3 │  ...   │
│  └────┬────┴────┬────┴────┬────┘        │
│       │         │         │              │
│       ▼         ▼         ▼              │
│  ┌──────────────────────────┐           │
│  │  Transaction Processor   │           │
│  │  1. Validate             │           │
│  │  2. Enrich               │           │
│  │  3. Calculate Metrics    │           │
│  └─────────┬────────────────┘           │
└────────────┼──────────────────────────────┘
             │
    ┌────────┴────────┐
    ▼                 ▼
┌──────────┐    ┌──────────┐
│PostgreSQL│    │  Redis   │
│(Storage) │    │ (Cache)  │
└────┬─────┘    └────┬─────┘
     │               │
     └───────┬───────┘
             ▼
      ┌────────────┐
      │  Dashboard │
      │ (Streamlit)│
      └────────────┘
```

RECOMMENDATIONS:

1. Message Queue: Redis Streams
   - Pros: Fast, persistent, built-in consumer groups
   - Handles 10K/min easily
   - Lower cost than Kafka for this scale

2. Workers: Celery with Redis backend
   - Auto-scaling based on queue size
   - Retry logic built-in
   - Easy monitoring

3. Database: PostgreSQL with TimescaleDB extension
   - Time-series optimization for metrics
   - Supports analytics queries
   - ACID compliance for transactions

4. Caching: Redis
   - Cache customer data (reduce DB hits)
   - Store real-time metrics
   - TTL for automatic cleanup

IMPLEMENTATION:

# 1. Transaction Model
from sqlalchemy import Column, Integer, String, Float, DateTime, Index
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    transaction_id = Column(String(50), unique=True, index=True)
    customer_id = Column(String(50), index=True)
    product_id = Column(String(50))
    quantity = Column(Integer)
    price = Column(Float)
    total_amount = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    status = Column(String(20))  # pending, processed, failed
    region = Column(String(50))
    
    # Composite index for common queries
    __table_args__ = (
        Index('idx_customer_timestamp', 'customer_id', 'timestamp'),
        Index('idx_region_timestamp', 'region', 'timestamp'),
    )

# 2. Celery Task for Processing
from celery import Celery
from redis import Redis
import logging

app = Celery('transaction_processor', broker='redis://localhost:6379/0')
redis_client = Redis(host='localhost', port=6379, db=1)
logger = logging.getLogger(__name__)

@app.task(bind=True, max_retries=3)
def process_transaction(self, transaction_data):
    """
    Process a single transaction with retry logic
    """
    try:
        # Step 1: Validate
        validated = validate_transaction(transaction_data)
        
        # Step 2: Enrich with customer data
        enriched = enrich_customer_data(validated)
        
        # Step 3: Calculate metrics
        update_metrics(enriched)
        
        # Step 4: Store in database
        save_transaction(enriched)
        
        logger.info(f"Successfully processed transaction: {transaction_data['transaction_id']}")
        return {'status': 'success', 'transaction_id': transaction_data['transaction_id']}
        
    except Exception as e:
        logger.error(f"Error processing transaction {transaction_data['transaction_id']}: {e}")
        
        # Retry with exponential backoff
        try:
            raise self.retry(exc=e, countdown=2 ** self.request.retries)
        except self.MaxRetriesExceededError:
            # After max retries, save to dead letter queue
            save_to_dlq(transaction_data, str(e))
            return {'status': 'failed', 'error': str(e)}

def validate_transaction(data):
    """Validate transaction data"""
    required_fields = ['transaction_id', 'customer_id', 'product_id', 'quantity', 'price']
    
    for field in required_fields:
        if field not in data or data[field] is None:
            raise ValueError(f"Missing required field: {field}")
    
    if data['quantity'] <= 0:
        raise ValueError("Quantity must be positive")
    
    if data['price'] < 0:
        raise ValueError("Price cannot be negative")
    
    data['total_amount'] = data['quantity'] * data['price']
    data['status'] = 'validated'
    
    return data

def enrich_customer_data(transaction):
    """Enrich with customer data from cache or database"""
    customer_id = transaction['customer_id']
    
    # Try cache first
    cached = redis_client.get(f"customer:{customer_id}")
    if cached:
        import json
        customer_data = json.loads(cached)
    else:
        # Fetch from database
        customer_data = fetch_customer_from_db(customer_id)
        # Cache for 1 hour
        redis_client.setex(
            f"customer:{customer_id}",
            3600,
            json.dumps(customer_data)
        )
    
    transaction['region'] = customer_data.get('region', 'unknown')
    transaction['customer_tier'] = customer_data.get('tier', 'standard')
    
    return transaction

def update_metrics(transaction):
    """Update real-time metrics in Redis"""
    from datetime import datetime
    import json
    
    current_hour = datetime.utcnow().strftime('%Y-%m-%d-%H')
    
    # Increment transaction count
    redis_client.hincrby(f"metrics:{current_hour}", 'transaction_count', 1)
    
    # Add to revenue
    redis_client.hincrbyfloat(f"metrics:{current_hour}", 'total_revenue', transaction['total_amount'])
    
    # Update region metrics
    redis_client.hincrby(f"metrics:region:{transaction['region']}:{current_hour}", 'count', 1)
    redis_client.hincrbyfloat(f"metrics:region:{transaction['region']}:{current_hour}", 'revenue', transaction['total_amount'])
    
    # Set expiry for 7 days
    redis_client.expire(f"metrics:{current_hour}", 7 * 24 * 3600)

def save_transaction(transaction):
    """Save to PostgreSQL"""
    from sqlalchemy.orm import sessionmaker
    from database import engine
    
    Session = sessionmaker(bind=engine)
    session = Session()
    
    try:
        db_transaction = Transaction(
            transaction_id=transaction['transaction_id'],
            customer_id=transaction['customer_id'],
            product_id=transaction['product_id'],
            quantity=transaction['quantity'],
            price=transaction['price'],
            total_amount=transaction['total_amount'],
            status='processed',
            region=transaction['region']
        )
        session.add(db_transaction)
        session.commit()
    except Exception as e:
        session.rollback()
        raise
    finally:
        session.close()

def save_to_dlq(transaction, error):
    """Save failed transaction to dead letter queue"""
    import json
    redis_client.lpush('dlq:transactions', json.dumps({
        'transaction': transaction,
        'error': error,
        'timestamp': datetime.utcnow().isoformat()
    }))

# 3. FastAPI Endpoint
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import uuid

app = FastAPI()

class TransactionCreate(BaseModel):
    customer_id: str
    product_id: str
    quantity: int
    price: float
    
    @validator('quantity')
    def quantity_positive(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
    
    @validator('price')
    def price_non_negative(cls, v):
        if v < 0:
            raise ValueError('Price cannot be negative')
        return v

@app.post("/api/v1/transactions")
async def create_transaction(transaction: TransactionCreate):
    """
    Accept transaction and queue for processing
    """
    # Generate unique transaction ID
    transaction_id = str(uuid.uuid4())
    
    transaction_data = {
        'transaction_id': transaction_id,
        **transaction.dict()
    }
    
    # Queue for processing
    process_transaction.delay(transaction_data)
    
    return {
        'status': 'accepted',
        'transaction_id': transaction_id,
        'message': 'Transaction queued for processing'
    }

@app.get("/api/v1/metrics/current")
async def get_current_metrics():
    """
    Get real-time metrics for current hour
    """
    from datetime import datetime
    current_hour = datetime.utcnow().strftime('%Y-%m-%d-%H')
    
    metrics = redis_client.hgetall(f"metrics:{current_hour}")
    
    return {
        'hour': current_hour,
        'transaction_count': int(metrics.get(b'transaction_count', 0)),
        'total_revenue': float(metrics.get(b'total_revenue', 0)),
        'avg_transaction_value': float(metrics.get(b'total_revenue', 0)) / max(int(metrics.get(b'transaction_count', 1)), 1)
    }

# 4. Docker Compose Configuration
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: transactions
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  api:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis
      - postgres
    environment:
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgresql://admin:password@postgres/transactions

  worker:
    build: .
    command: celery -A tasks worker --loglevel=info --concurrency=4
    depends_on:
      - redis
      - postgres
    environment:
      REDIS_URL: redis://redis:6379
      DATABASE_URL: postgresql://admin:password@postgres/transactions
    deploy:
      replicas: 3  # 3 workers to start

volumes:
  redis_data:
  postgres_data:
```

SCALING CONSIDERATIONS:

1. Horizontal Scaling:
   - API: Add more containers behind load balancer (ALB)
   - Workers: Auto-scale based on queue length
   - Database: Read replicas for analytics queries

2. Performance Optimizations:
   - Batch insert transactions (bulk operations)
   - Connection pooling for database
   - Redis pipelining for metrics updates

3. Monitoring:
   - Celery Flower for worker monitoring
   - Prometheus + Grafana for metrics
   - CloudWatch for AWS resources

4. Cost Optimization:
   - Start with 3 workers, scale up to 10 based on queue
   - Use reserved instances for database
   - Redis ElastiCache instead of self-hosted

Expected Capacity:
- 10,000 transactions/min = 167/second
- With 3 workers @ 50 tx/sec each = handles load with buffer
- Total monthly cost (AWS): ~$500-700
```

**This complete architecture took Claude about 30 seconds to generate!**

---

## Slide 15: Debugging with AI

### Finding and Fixing Bugs Faster

**Use Case 1: Understanding Error Messages**

**Scenario:** You get a cryptic error

**Error:**
```python
Traceback (most recent call last):
  File "app.py", line 45, in process_data
    result = data['users'][user_id]['transactions']
KeyError: 'transactions'
```

**Prompt to ChatGPT:**
```
I'm getting this error in my Python application:

[paste error]

Here's the relevant code:

```python
def process_data(data, user_id):
    user = data['users'][user_id]
    transactions = user['transactions']
    return sum(t['amount'] for t in transactions)
```

The data structure should be:
{
  'users': {
    'user123': {
      'name': 'John',
      'transactions': [...]
    }
  }
}

Why might I be getting this KeyError, and how can I fix it defensively?
```

**ChatGPT Response:**
```
The KeyError indicates that the 'transactions' key doesn't exist for this user.

Possible causes:
1. User exists but has no 'transactions' key
2. Data structure differs from expected
3. Transactions key is sometimes None instead of missing

Defensive solution:

```python
def process_data(data, user_id):
    """
    Process user transactions with defensive error handling
    """
    # Check if user exists
    if 'users' not in data:
        raise ValueError("Data missing 'users' key")
    
    if user_id not in data['users']:
        raise ValueError(f"User {user_id} not found")
    
    user = data['users'][user_id]
    
    # Safely get transactions with default empty list
    transactions = user.get('transactions', [])
    
    # Validate transactions is a list
    if not isinstance(transactions, list):
        logger.warning(f"User {user_id} has invalid transactions format")
        return 0
    
    # Calculate total with validation
    total = 0
    for transaction in transactions:
        if isinstance(transaction, dict) and 'amount' in transaction:
            try:
                total += float(transaction['amount'])
            except (ValueError, TypeError):
                logger.warning(f"Invalid amount in transaction: {transaction}")
                continue
    
    return total
```

Additional debugging steps:
1. Add logging to see actual data structure
2. Validate data schema before processing
3. Use Pydantic for data validation
```

**Use Case 2: Performance Debugging**

**Prompt:**
```
This function is very slow with large datasets:

```python
def find_duplicates(data):
    duplicates = []
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if data[i] == data[j] and data[i] not in duplicates:
                duplicates.append(data[i])
    return duplicates

# Takes 45 seconds with 10,000 items!
```

Can you:
1. Explain why it's slow
2. Provide an optimized version
3. Show performance comparison
```

**ChatGPT Response:**
```
PROBLEM ANALYSIS:

Your function has O(n³) time complexity:
- Outer loop: O(n)
- Inner loop: O(n)
- `data[i] not in duplicates`: O(n) for list lookup

Total: O(n) * O(n) * O(n) = O(n³)

For 10,000 items: ~1,000,000,000,000 operations!

OPTIMIZED VERSION:

```python
from collections import Counter

def find_duplicates_optimized(data):
    """
    Find duplicates using Counter - O(n) time complexity
    """
    counts = Counter(data)
    return [item for item, count in counts.items() if count > 1]
```

PERFORMANCE COMPARISON:

```python
import time
import random

# Generate test data
test_data = [random.randint(1, 1000) for _ in range(10000)]

# Original version
start = time.time()
result1 = find_duplicates(test_data[:1000])  # Only 1000 items!
time1 = time.time() - start

# Optimized version
start = time.time()
result2 = find_duplicates_optimized(test_data)  # All 10,000 items
time2 = time.time() - start

print(f"Original (1,000 items): {time1:.2f} seconds")
print(f"Optimized (10,000 items): {time2:.4f} seconds")
print(f"Speedup factor: {time1/time2:.0f}x")

# Output:
# Original (1,000 items): 2.45 seconds
# Optimized (10,000 items): 0.0023 seconds
# Speedup factor: 1065x
```

EXPLANATION:

Counter approach:
1. Single pass through data: O(n)
2. Build hash map of counts: O(n) space
3. Filter items with count > 1: O(n)

Total: O(n) time, O(n) space

For 10,000 items: ~10,000 operations vs ~1 trillion!
```

**Use Case 3: Logic Bugs**

**Scenario:** Code runs but gives wrong results

**Prompt:**
```
This function should calculate customer loyalty discount, but customers are complaining:

```python
def calculate_loyalty_discount(total, years_member):
    discount = 0
    if years_member > 1:
        discount = 0.05
    if years_member > 3:
        discount = 0.10
    if years_member > 5:
        discount = 0.15
    
    return total * (1 - discount)
```

A customer with 6 years membership and $100 purchase is getting charged $95 instead of $85.
What's wrong?
```

**ChatGPT Response:**
```
BUG IDENTIFIED:

The issue is with multiple IF statements instead of IF-ELIF.

Current behavior:
- For 6 years: all three conditions are TRUE
- But each IF assigns discount, so only last assignment (0.15) matters
- However, the customer expects cumulative discount

Wait - if only 0.15 is applied, they should pay $85, not $95...

Let me recalculate:
- $100 * (1 - 0.15) = $85 ✓

Actually, checking the logic more carefully - they ARE getting $95, which means:
- $100 * (1 - 0.05) = $95

This suggests the discount variable is being set to 0.05, which means the later conditions aren't being met!

ACTUAL BUG: The comparison operators!

Looking again at years_member > 5:
- If years_member is exactly 6, this should be TRUE
- But customer is getting 5% discount (years_member > 1)

OH! I see it now - the conditions are all being evaluated, but perhaps years_member is a string "6" not integer 6!

FIXED VERSION:

```python
def calculate_loyalty_discount(total, years_member):
    """
    Calculate price after loyalty discount
    
    Discount tiers:
    - 1-3 years: 5%
    - 3-5 years: 10%
    - 5+ years: 15%
    """
    # Ensure years_member is an integer
    years_member = int(years_member)
    
    # Use elif to ensure only one discount applies
    if years_member >= 5:
        discount = 0.15
    elif years_member >= 3:
        discount = 0.10
    elif years_member >= 1:
        discount = 0.05
    else:
        discount = 0.0
    
    discounted_total = total * (1 - discount)
    
    return round(discounted_total, 2)

# Test cases
print(calculate_loyalty_discount(100, 6))    # Should be $85.00
print(calculate_loyalty_discount(100, "6"))  # Should handle string input
print(calculate_loyalty_discount(100, 3))    # Should be $90.00
print(calculate_loyalty_discount(100, 1))    # Should be $95.00
```

KEY FIXES:
1. Convert years_member to int (handles string inputs)
2. Use >= instead of > (5 years should get 15% discount)
3. Use elif to ensure only one discount applies (though with proper ordering, if statements would work)
4. Round result to 2 decimal places for currency
5. Order conditions from highest to lowest for clarity
```

---

**Best Practices for AI-Assisted Debugging:**

1. **Provide complete context:**
   - Full error message
   - Relevant code
   - Expected vs actual behavior
   - Sample data

2. **Be specific about the problem:**
   - "Returns wrong value" → "Returns 95 instead of expected 85"
   - "It's slow" → "Takes 45 seconds for 10,000 items"

3. **Ask for explanations:**
   - "Why is this happening?"
   - "What's the root cause?"
   - "What are the implications?"

4. **Request multiple solutions:**
   - "Show me 3 different approaches"
   - "What are the trade-offs?"

5. **Verify with tests:**
   - Always ask for test cases
   - Run the suggested fix
   - Compare with original

---

**End of Batch 3 (Slides 11-15)**

*Continue to Batch 4 for Code Review, Testing & Documentation (Slides 16-20)*
