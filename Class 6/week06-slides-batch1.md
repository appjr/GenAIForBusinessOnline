# Week 6: Coding with AI - Slides Batch 1 (Slides 1-5)

**Course:** BUAN 6v99.SW2 - Generative AI for Business  
**Date:** February 24, 2026  
**Duration:** 2.5 hours

---

## Slide 1: Week 6 Title Slide

### Coding with AI: The Future of Software Development

**Today's Focus:**
- AI-powered coding assistants and tools
- Code generation, completion, and refactoring
- Debugging and code review with AI
- Automated testing and documentation
- Measuring productivity gains and ROI
- Best practices for AI-assisted development

**Prerequisites:**
- Basic programming knowledge (Python preferred)
- Understanding of software development lifecycle
- Familiarity with version control (Git)

**Tools We'll Explore:**
- GitHub Copilot
- ChatGPT/Claude for coding
- Amazon CodeWhisperer
- AI debugging and testing tools

---

## Slide 2: Today's Agenda

### Class Overview

1. **AI Coding Tools Landscape** (30 min)
   - Evolution from autocomplete to AI assistants
   - Major tools comparison
   - Choosing the right tool

2. **Code Generation & Completion** (30 min)
   - GitHub Copilot deep dive
   - Prompt engineering for code
   - Real-world examples

3. **Break** (10 min)

4. **Conversational Coding & Debugging** (35 min)
   - ChatGPT/Claude for complex problems
   - AI-powered debugging
   - Error analysis and fixes

5. **Code Review, Testing & Documentation** (25 min)
   - Automated code review
   - Test generation
   - Documentation automation

6. **Business Applications & ROI** (20 min)
   - Productivity measurements
   - Cost-benefit analysis
   - Enterprise implementation

7. **Hands-on Lab & Q&A** (20 min)
   - Live coding with AI assistance
   - Best practices workshop

---

## Slide 3: Learning Objectives

### By the End of This Class, You Will:

✅ **Understand** the landscape of AI coding assistants  
✅ **Use** GitHub Copilot and ChatGPT/Claude for coding tasks  
✅ **Apply** AI to code generation, debugging, and refactoring  
✅ **Generate** tests and documentation automatically  
✅ **Evaluate** productivity gains and ROI for your organization  
✅ **Implement** best practices for AI-assisted development  
✅ **Recognize** limitations and potential pitfalls

**Practical Skills:**
- Write better code faster with AI assistance
- Debug complex issues using AI analysis
- Automate repetitive coding tasks
- Measure and improve development velocity

**Business Skills:**
- Calculate ROI of AI coding tools
- Build business case for adoption
- Implement AI tools across development teams
- Measure productivity improvements

---

## Slide 4: The Evolution of Coding Assistance

### From Autocomplete to AI Pair Programming

**Historical Timeline:**

**1. Early IDEs (1990s-2000s)**
- Syntax highlighting
- Basic autocomplete (variable names, keywords)
- Manual code templates
- No intelligence, just text matching
- Example: Eclipse, Visual Studio

**2. IntelliSense & Smart Completion (2000s-2010s)**
- Context-aware suggestions
- API documentation integration
- Type inference
- Static analysis
- Still rule-based, not learning

**3. Language Servers & Advanced Analysis (2010s)**
- Deep code understanding
- Refactoring suggestions
- Error detection before compilation
- Jump to definition, find references
- Example: Language Server Protocol (LSP)

**4. Machine Learning-Based Tools (2015-2020)**
- Pattern recognition from large codebases
- Smarter autocomplete
- Code smell detection
- Limited to simple suggestions
- Example: TabNine (early versions), Kite

**5. AI Pair Programming (2020-Present)**
- **GitHub Copilot (2021):** Revolutionary change
- **ChatGPT for Code (2022):** Natural language → Code
- **GPT-4 & Claude (2023):** Complex reasoning
- **Specialized Tools (2024+):** Domain-specific AI

**The Paradigm Shift:**

```
Traditional Coding:
Developer → Writes every line → Code → Debug → Test

AI-Assisted Coding:
Developer → Describes intent → AI generates code → Developer reviews → Test
         ↓
     50-80% faster for many tasks
```

**Key Statistics:**

| Metric | Without AI | With AI | Improvement |
|--------|-----------|---------|-------------|
| **Code completion time** | 100% | 46% | 54% faster |
| **New API learning** | 2-3 days | 2-3 hours | 90% faster |
| **Boilerplate code** | Manual | Automated | 95% faster |
| **Documentation** | Often skipped | Auto-generated | ∞ improvement |
| **Bug fix time** | Varies | 30% faster | Average |

**Source:** GitHub Copilot research (2023), McKinsey Developer Survey

**Business Impact:**
```python
# Calculate productivity gain
developers = 50
avg_salary = 120000
hours_per_year = 2000
time_saved_percent = 0.30  # Conservative estimate

annual_value = developers * avg_salary * time_saved_percent
print(f"Annual value: ${annual_value:,}")
# Output: Annual value: $1,800,000

# Cost of tools
copilot_cost = 10 * 12 * developers  # $10/month per developer
roi = (annual_value - copilot_cost) / copilot_cost * 100
print(f"ROI: {roi:.0f}%")
# Output: ROI: 2,900%
```

---

## Slide 5: The AI Coding Tools Landscape

### Major Players and Their Capabilities

**1. GitHub Copilot (Microsoft/OpenAI)**

**Overview:**
- Powered by OpenAI Codex (GPT-3.5/GPT-4 based)
- Trained on billions of lines of public code
- IDE integration: VS Code, JetBrains, Neovim, Visual Studio
- Launched: June 2021

**Key Features:**
- ✅ Real-time code suggestions as you type
- ✅ Multi-language support (40+ languages)
- ✅ Context-aware (understands surrounding code)
- ✅ Generates functions, classes, tests
- ✅ Learns from your coding style

**Pricing:**
- Individual: $10/month or $100/year
- Business: $19/user/month
- Enterprise: Custom pricing

**Strengths:**
- Best IDE integration
- Fast, real-time suggestions
- Great for common patterns
- Active development community

**Limitations:**
- Sometimes suggests outdated patterns
- May suggest insecure code
- Limited context window
- Requires internet connection

**Example Use Case:**
```python
# Type a comment, Copilot generates the code
# Function to calculate compound interest

# Copilot suggestion:
def calculate_compound_interest(principal, rate, time, frequency=12):
    """
    Calculate compound interest.
    
    Args:
        principal: Initial investment amount
        rate: Annual interest rate (as decimal)
        time: Time period in years
        frequency: Compounding frequency per year
    
    Returns:
        Final amount after compound interest
    """
    amount = principal * (1 + rate/frequency) ** (frequency * time)
    interest = amount - principal
    return amount, interest

# That entire function was generated from just the comment!
```

---

**2. Amazon CodeWhisperer**

**Overview:**
- Amazon's AI coding assistant
- Integrated with AWS services
- Free tier available
- Launched: April 2022, Generally Available: April 2023

**Key Features:**
- ✅ AWS SDK optimization
- ✅ Security scanning built-in
- ✅ Reference tracking (shows source of suggestions)
- ✅ Custom model training on private repos
- ✅ IDE support: VS Code, JetBrains, AWS Cloud9

**Pricing:**
- Individual: FREE
- Professional: $19/user/month

**Strengths:**
- Free for individuals
- Excellent AWS integration
- Security scanning included
- Reference tracking for licensing compliance

**Limitations:**
- Less mature than Copilot
- Smaller training dataset
- Fewer IDE integrations

**Example Use Case:**
```python
# CodeWhisperer excels at AWS-specific code
# Type: "Upload file to S3 bucket with encryption"

import boto3
from botocore.exceptions import ClientError

def upload_file_to_s3(file_path, bucket_name, object_name=None, encryption='AES256'):
    """
    Upload file to S3 bucket with server-side encryption.
    
    Args:
        file_path: Path to file to upload
        bucket_name: Target S3 bucket
        object_name: S3 object name (defaults to file_path)
        encryption: Encryption type ('AES256' or 'aws:kms')
    
    Returns:
        True if successful, False otherwise
    """
    if object_name is None:
        object_name = file_path.split('/')[-1]
    
    s3_client = boto3.client('s3')
    
    try:
        s3_client.upload_file(
            file_path,
            bucket_name,
            object_name,
            ExtraArgs={
                'ServerSideEncryption': encryption
            }
        )
        return True
    except ClientError as e:
        print(f"Error uploading file: {e}")
        return False

# CodeWhisperer also shows: "Reference: boto3 documentation"
```

---

**3. ChatGPT/Claude (Conversational AI)**

**Overview:**
- General-purpose AI that excels at coding
- Web interface + API
- Not IDE-integrated (copy/paste workflow)
- Best for: Architecture, learning, complex problems

**ChatGPT (OpenAI):**
- GPT-4: Advanced reasoning, large context
- GPT-3.5: Fast, cost-effective
- Web: $20/month for GPT-4
- API: Pay per token

**Claude (Anthropic):**
- Claude 3 Opus: Best for complex reasoning
- Claude 3 Sonnet: Balanced performance
- 100K+ token context window (huge advantage!)
- Web: Free tier + Pro at $20/month
- API: Pay per token

**Strengths:**
- ✅ Explains concepts deeply
- ✅ Helps with architecture decisions
- ✅ Great for learning new technologies
- ✅ Debugging complex issues
- ✅ Can analyze entire codebases (Claude)

**Limitations:**
- ❌ No IDE integration (manual copy/paste)
- ❌ Not real-time
- ❌ May generate plausible but incorrect code
- ❌ Needs careful prompt engineering

**Example Use Case:**
```
User: "I need to build a REST API for a e-commerce product catalog 
with search, filtering, and caching. Using Python FastAPI. What's 
the best architecture?"

ChatGPT/Claude Response:
"""
Here's a recommended architecture for your e-commerce product catalog API:

1. Architecture Layers:
   - API Layer: FastAPI with Pydantic models
   - Business Logic: Service layer for catalog operations
   - Data Access: Repository pattern with SQLAlchemy
   - Caching: Redis for frequently accessed products
   - Search: Elasticsearch for full-text search
   - Database: PostgreSQL for product data

2. Key Components:

[Provides complete code structure with:]
- FastAPI app setup
- Database models
- Pydantic schemas
- Repository layer
- Service layer with caching
- Search integration
- API endpoints
- Error handling
- Testing approach
- Deployment considerations

3. Why this architecture:
   - Scalability: Caching reduces DB load
   - Performance: Elasticsearch for fast search
   - Maintainability: Clear separation of concerns
   - Testing: Easy to mock dependencies
"""

[Followed by complete implementation code]
```

---

**4. Specialized Tools**

**Tabnine:**
- AI code completion
- Trains on your team's codebase
- Privacy-focused (local models available)
- Free tier + Pro at $12/month

**Cursor:**
- AI-first code editor (fork of VS Code)
- Built-in ChatGPT integration
- Codebase-aware chat
- $20/month

**Codeium:**
- Free Copilot alternative
- Fast, accurate suggestions
- 70+ language support
- Free for individuals

**Replit Ghostwriter:**
- Integrated with Replit IDE
- Great for learning/prototyping
- $10/month

**Sourcegraph Cody:**
- Codebase-aware AI assistant
- Understands your entire repository
- Free tier + Pro at $9/month

---

**Comparison Matrix:**

| Tool | Best For | Pricing | IDE Integration | Context Size | Unique Feature |
|------|----------|---------|-----------------|--------------|----------------|
| **GitHub Copilot** | General coding | $10-19/mo | Excellent | Medium | Best autocomplete |
| **CodeWhisperer** | AWS development | Free-$19/mo | Good | Medium | Security scanning |
| **ChatGPT** | Complex problems | $20/mo | None | Large | Best explanations |
| **Claude** | Large codebases | $20/mo | None | Huge (100K+) | Largest context |
| **Tabnine** | Team collaboration | Free-$12/mo | Good | Small-Large | Team training |
| **Cursor** | Integrated workflow | $20/mo | Built-in | Large | AI-first editor |
| **Codeium** | Budget-conscious | Free | Good | Medium | Free alternative |

**Choosing the Right Tool:**

```python
def recommend_tool(use_case):
    """
    Recommend AI coding tool based on use case.
    """
    recommendations = {
        'general_coding': 'GitHub Copilot - best all-around',
        'aws_development': 'Amazon CodeWhisperer - AWS optimized',
        'learning': 'ChatGPT/Claude - best explanations',
        'architecture': 'Claude - huge context window',
        'budget': 'Codeium - free alternative',
        'team_collaboration': 'Tabnine - trains on team code',
        'privacy_sensitive': 'Tabnine (local) - on-premise models',
        'integrated_workflow': 'Cursor - AI-first editor'
    }
    
    return recommendations.get(use_case, 'Try GitHub Copilot first')

# Examples:
print(recommend_tool('general_coding'))
# Output: GitHub Copilot - best all-around

print(recommend_tool('learning'))
# Output: ChatGPT/Claude - best explanations
```

**Best Practice: Use Multiple Tools**

Most effective developers use a combination:
1. **GitHub Copilot** for day-to-day coding
2. **ChatGPT/Claude** for complex problems and learning
3. **CodeWhisperer** for AWS-specific tasks (if applicable)

**ROI Comparison:**

```python
# Calculate ROI for different tools
import pandas as pd

tools = {
    'Tool': ['Copilot', 'CodeWhisperer', 'ChatGPT', 'Claude', 'Cursor'],
    'Monthly Cost': [10, 19, 20, 20, 20],
    'Time Saved (hrs/month)': [20, 15, 10, 12, 18],
    'Developer Rate ($/hr)': [75, 75, 75, 75, 75]
}

df = pd.DataFrame(tools)
df['Monthly Value'] = df['Time Saved (hrs/month)'] * df['Developer Rate ($/hr)']
df['Monthly ROI'] = ((df['Monthly Value'] - df['Monthly Cost']) / df['Monthly Cost'] * 100).round(0)
df['Annual ROI'] = df['Monthly ROI']  # Same for annual

print(df[['Tool', 'Monthly Cost', 'Time Saved (hrs/month)', 'Monthly Value', 'Monthly ROI']])

"""
Output:
         Tool  Monthly Cost  Time Saved  Monthly Value  Monthly ROI
0     Copilot            10          20           1500       14900%
1  CodeWhisperer         19          15           1125        5821%
2     ChatGPT            20          10            750        3650%
3      Claude            20          12            900        4400%
4      Cursor            20          18           1350        6650%

Conclusion: Even conservative estimates show massive ROI!
"""
```

---

**End of Batch 1 (Slides 1-5)**

*Continue to Batch 2 for Code Generation & GitHub Copilot Deep Dive (Slides 6-10)*
