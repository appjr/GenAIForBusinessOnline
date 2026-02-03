# Week 3: Python with ML and GenAI Toolkits - Slides Batch 4 (Slides 31-40)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 4, 2026

---

## Slide 31: Best Practices for GenAI Development

### Building Robust AI Applications

**1. Prompt Engineering**
```python
# Bad prompt
prompt = "Analyze this"

# Good prompt
prompt = """
You are an expert business analyst with 10 years of experience.
Analyze the following business metrics and provide:
1. Key trends
2. Areas of concern
3. Three specific recommendations

Data: {data}

Format your response in markdown with clear sections.
"""
```

**2. Error Handling & Fallbacks**
```python
def safe_ai_call(prompt, fallback_response="Unable to process"):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"AI call failed: {e}")
        return fallback_response
```

**3. Response Validation**
```python
def validate_ai_response(response, expected_format="json"):
    if not response:
        return False
    
    if expected_format == "json":
        try:
            json.loads(response)
            return True
        except:
            return False
    
    return len(response) > 10  # Basic validation
```

---

## Slide 32: Ethics and Responsible AI

### Building Trustworthy AI Systems

**Ethical Considerations:**

1. **Bias and Fairness**
   - Test AI outputs across diverse scenarios
   - Monitor for discriminatory patterns
   - Use diverse training data

2. **Privacy and Data Protection**
   - Never send sensitive customer data to external APIs
   - Anonymize data before processing
   - Follow GDPR, CCPA regulations

3. **Transparency**
   - Disclose AI usage to users
   - Explain AI decisions when possible
   - Provide human oversight options

**Implementation Example:**
```python
def analyze_with_privacy(text):
    # Remove PII before sending to AI
    import re
    
    # Remove emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '[EMAIL]', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    
    # Remove SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
    
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": text}]
    )
```

**AI Ethics Checklist:**
- ✅ Obtain user consent for AI processing
- ✅ Provide opt-out mechanisms
- ✅ Regular bias audits
- ✅ Human review for critical decisions
- ✅ Clear documentation and logging

---

## Slide 33: Performance Optimization

### Making Your AI Apps Faster

**1. Parallel Processing**
```python
from concurrent.futures import ThreadPoolExecutor
import time

def process_batch(texts):
    """Process multiple texts in parallel"""
    
    def single_analysis(text):
        return client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": f"Analyze: {text}"}]
        )
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(single_analysis, texts))
    
    return results

# Usage
texts = ["text1", "text2", "text3", "text4", "text5"]
results = process_batch(texts)
```

**2. Caching Results**
```python
from functools import lru_cache
import hashlib

class CachedAIClient:
    def __init__(self):
        self.cache = {}
    
    def generate(self, prompt):
        # Create cache key
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            print("Cache hit!")
            return self.cache[cache_key]
        
        # Generate and cache
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        self.cache[cache_key] = response
        return response
```

**3. Batch Processing**
```python
def batch_embeddings(texts, batch_size=100):
    """Process embeddings in batches"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

---

## Slide 34: Monitoring and Observability

### Tracking AI Application Performance

**Key Metrics to Monitor:**

1. **Response Time**
2. **Token Usage**
3. **Error Rates**
4. **Cost per Request**
5. **User Satisfaction**

**Implementation:**
```python
import time
from dataclasses import dataclass
from datetime import datetime
import uuid

@dataclass
class AIMetrics:
    request_id: str
    timestamp: datetime
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error: str = None

class MonitoredAIClient:
    def __init__(self):
        self.client = OpenAI()
        self.metrics = []
    
    def generate(self, prompt, model="gpt-4"):
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            latency = (time.time() - start_time) * 1000
            usage = response.usage
            
            # Calculate cost (example rates)
            cost = (usage.prompt_tokens * 0.00003 + 
                   usage.completion_tokens * 0.00006)
            
            metric = AIMetrics(
                request_id=request_id,
                timestamp=datetime.now(),
                model=model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                latency_ms=latency,
                cost_usd=cost,
                success=True
            )
            
            self.metrics.append(metric)
            return response
            
        except Exception as e:
            metric = AIMetrics(
                request_id=request_id,
                timestamp=datetime.now(),
                model=model,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=(time.time() - start_time) * 1000,
                cost_usd=0,
                success=False,
                error=str(e)
            )
            self.metrics.append(metric)
            raise
    
    def get_summary(self):
        total_requests = len(self.metrics)
        successful = sum(1 for m in self.metrics if m.success)
        total_cost = sum(m.cost_usd for m in self.metrics)
        avg_latency = sum(m.latency_ms for m in self.metrics) / total_requests
        
        return {
            "total_requests": total_requests,
            "success_rate": successful / total_requests,
            "total_cost_usd": total_cost,
            "avg_latency_ms": avg_latency
        }
```

---

## Slide 35: Advanced Integration Patterns

### Combining Multiple AI Services

**Multi-Model Strategy:**
```python
class HybridAISystem:
    def __init__(self):
        self.openai_client = OpenAI()
        # Could add Anthropic, Cohere, etc.
    
    def route_request(self, task_type, content):
        """Route to best model for task"""
        
        routes = {
            "simple_qa": ("gpt-3.5-turbo", 0.5),
            "complex_analysis": ("gpt-4", 0.7),
            "creative": ("gpt-4", 0.9),
            "factual": ("gpt-3.5-turbo", 0.3)
        }
        
        model, temperature = routes.get(task_type, ("gpt-4", 0.7))
        
        return self.openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=temperature
        )
    
    def ensemble_response(self, prompt):
        """Get responses from multiple models and combine"""
        
        models = ["gpt-3.5-turbo", "gpt-4"]
        responses = []
        
        for model in models:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            responses.append(response.choices[0].message.content)
        
        # Use AI to synthesize responses
        synthesis_prompt = f"""
        Here are multiple AI responses to the same question.
        Synthesize them into one comprehensive answer:
        
        Response 1: {responses[0]}
        Response 2: {responses[1]}
        """
        
        final = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": synthesis_prompt}]
        )
        
        return final.choices[0].message.content
```

---

## Slide 36: Troubleshooting Common Issues

### Debugging AI Applications

**Common Problems & Solutions:**

**1. Inconsistent Outputs**
```python
# Problem: Different results each time
# Solution: Control temperature and use seed (when available)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": prompt}],
    temperature=0,  # More deterministic
    seed=42  # Reproducible results
)
```

**2. Rate Limiting**
```python
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def resilient_api_call(prompt):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
```

**3. Context Length Exceeded**
```python
def chunk_and_process(long_text, chunk_size=3000):
    """Break long text into chunks"""
    
    chunks = [long_text[i:i+chunk_size] 
              for i in range(0, len(long_text), chunk_size)]
    
    results = []
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": chunk}]
        )
        results.append(response.choices[0].message.content)
    
    # Combine results
    combined = "\n\n".join(results)
    return combined
```

**4. Slow Responses**
```python
# Use streaming for better UX
def stream_response(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Slide 37: Week 3 Assignment

### Build Your Own AI-Powered Business Tool

**Assignment Overview:**
Create a GenAI application that solves a real business problem.

**Requirements:**

1. **Core Functionality** (40 points)
   - Use at least one GenAI API (OpenAI, Hugging Face, etc.)
   - Implement proper error handling
   - Include input validation

2. **Technical Implementation** (30 points)
   - Clean, well-documented code
   - Virtual environment with requirements.txt
   - Proper API key management (.env file)

3. **User Interface** (20 points)
   - Command-line OR web interface (Streamlit recommended)
   - Clear instructions for users
   - Professional appearance

4. **Documentation** (10 points)
   - README.md with setup instructions
   - Code comments
   - Example usage

**Project Ideas:**
- Email response generator
- Meeting notes summarizer
- Product description writer
- Customer feedback analyzer
- Business report generator
- Social media content creator

**Deliverables:**
- GitHub repository link
- 2-3 minute video demo
- Brief write-up (500 words)

**Due Date:** February 11, 2026 (before next class)

---

## Slide 38: Recommended Resources

### Continue Your Learning Journey

**Documentation:**
- OpenAI API Docs: https://platform.openai.com/docs
- Hugging Face: https://huggingface.co/docs
- LangChain: https://python.langchain.com/docs
- Streamlit: https://docs.streamlit.io

**Online Courses:**
- DeepLearning.AI - ChatGPT Prompt Engineering
- Coursera - Generative AI for Everyone
- Fast.ai - Practical Deep Learning

**Books:**
- "Hands-On Large Language Models" by Jay Alammar
- "Building LLM Apps" by Valentine Boyev
- "AI and Machine Learning for Business" by Steven Finlay

**Communities:**
- r/MachineLearning
- Hugging Face Forums
- OpenAI Community
- AI Stack Exchange

**Newsletters:**
- The Batch (DeepLearning.AI)
- Import AI
- AI Breakfast
- The Gradient

**YouTube Channels:**
- Andrej Karpathy
- Two Minute Papers
- AI Explained
- Yannic Kilcher

---

## Slide 39: Next Week Preview

### Week 4: Deep Learning and GenAI Algorithms

**What We'll Cover:**

1. **Neural Network Fundamentals**
   - Architecture basics
   - Forward and backward propagation
   - Training process

2. **Transformer Architecture**
   - Attention mechanisms
   - Self-attention
   - How GPT works under the hood

3. **Training Large Models**
   - Data requirements
   - Computational needs
   - Fine-tuning vs. training from scratch

4. **Model Evaluation**
   - Metrics for GenAI
   - Human evaluation
   - Benchmarks

5. **Hands-on Lab**
   - Fine-tune a small model
   - Build custom training pipeline

**Preparation:**
- Review linear algebra basics
- Install PyTorch
- Read: "Attention Is All You Need" paper (optional)

---

## Slide 40: Week 3 Recap & Q&A

### What We Accomplished Today

**Key Takeaways:**

✅ **Python Libraries Mastered**
   - NumPy for numerical computing
   - Pandas for data manipulation
   - scikit-learn, TensorFlow, PyTorch basics

✅ **GenAI Toolkits Explored**
   - OpenAI API integration
   - Hugging Face Transformers
   - LangChain for complex workflows

✅ **Built Complete Application**
   - AI-powered business analyzer
   - Web interface with Streamlit
   - Production-ready code

✅ **Best Practices Learned**
   - Error handling and monitoring
   - Cost optimization
   - Ethics and responsible AI

**Action Items:**
1. Complete the assignment
2. Experiment with different AI models
3. Join online communities
4. Start building your portfolio project

**Questions?**

**Office Hours:**
- Tuesday 2-4 PM (Virtual)
- Thursday 3-5 PM (In-person)
- By appointment

**Contact:**
- Email: [instructor email]
- Discord: [course channel]
- Canvas: [course page]

---

**Class Dismissed! See you next week! 🚀**

**Remember:** The best way to learn GenAI is by building. Start your project today!

---

**End of Week 3 - Python with ML and GenAI Toolkits**

*Total Slides: 40*
*Batch 4: Slides 31-40*
