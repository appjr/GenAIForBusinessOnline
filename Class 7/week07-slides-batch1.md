# Week 7: Text Generation, Chatbots - Slides Batch 1 (Slides 1-6)

**Course:** BUAN 6v99.SW2 - Generative AI for Business
**Date:** March 4, 2026
**Duration:** 2.5 hours

---

## Slide 1: Week 7 Title Slide

### Text Generation & Chatbots: Building Intelligent Conversational AI

**Today's Focus:**
- How large language models generate text
- OpenAI and Anthropic APIs for text generation
- Prompt engineering for conversational AI
- Building chatbots from scratch with Python
- Retrieval-Augmented Generation (RAG)
- Deploying chatbots for real business use cases

**Prerequisites:**
- Python basics (from earlier classes)
- API concepts (calling external services)
- Familiarity with ChatGPT / Claude as users

**What We'll Build Today:**
- A customer service chatbot using the OpenAI API
- A knowledge-base chatbot using RAG
- A multi-turn conversational agent

---

## Slide 2: Today's Agenda

### Class Overview

1. **Text Generation Fundamentals** (25 min)
   - How LLMs generate text
   - Key parameters: temperature, top-p, max tokens
   - The chat completion format

2. **OpenAI & Anthropic APIs** (30 min)
   - Authentication and setup
   - Making your first API call
   - System prompts, roles, messages

3. **Break** (10 min)

4. **Prompt Engineering for Chatbots** (30 min)
   - Designing system prompts
   - Few-shot examples in chat
   - Chain-of-thought prompting

5. **Building a Chatbot** (35 min)
   - Conversation memory and history
   - Multi-turn dialogue management
   - Streaming responses

6. **RAG & Advanced Techniques** (20 min)
   - Retrieval-Augmented Generation
   - Grounding chatbots in your data

7. **Hands-on Lab & Q&A** (20 min)
   - Build your own chatbot
   - Best practices workshop

---

## Slide 3: Environment Setup — Ollama (Free Local LLMs)

### Run AI Models 100% Locally — No API Key, No Cost

**Why Ollama?**
- Runs entirely on your laptop — no internet required after setup
- No API key or billing account needed
- OpenAI-compatible API: code that works with Ollama also works with GPT-4

---

**Step 1 — Install Ollama**

```bash
# Mac / Linux (one command):
curl -fsSL https://ollama.com/install.sh | sh

# Mac (Homebrew alternative):
brew install ollama

# Windows:
# Download the installer from https://ollama.com/download
```

---

**Step 2 — Start the Local Server**

```bash
# Run this in a terminal and keep it open (like a local API server)
ollama serve
# Listens on http://localhost:11434
```

---

**Step 3 — Download Models (one-time)**

```bash
ollama pull llama3.2    # ~2 GB — fast, good for most tasks
ollama pull mistral     # ~4 GB — higher quality reasoning
```

| Model | Size | Speed | Best For |
|-------|------|-------|----------|
| `llama3.2` | ~2 GB | Fast | Chatbots, Q&A, everyday tasks |
| `mistral` | ~4 GB | Medium | Analysis, reasoning, longer text |

---

**Step 4 — Install Python Packages**

```bash
pip install openai sentence-transformers gradio
```

---

**Step 5 — Verify Everything Works**

```python
from openai import OpenAI

client = OpenAI(
    api_key="ollama",                        # No real key needed
    base_url="http://localhost:11434/v1"     # Points to local server
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Say hello!"}]
)
print(response.choices[0].message.content)
# Output: "Hello! How can I help you today?"
```

**All code examples in this class use this pattern — free, local, no billing.**

---

## Slide 4: Learning Objectives

### By the End of This Class, You Will:

✅ **Understand** how large language models generate text token by token
✅ **Use** the OpenAI and Anthropic APIs to generate text programmatically
✅ **Design** effective system prompts for specific chatbot personas
✅ **Build** a multi-turn chatbot with conversation memory in Python
✅ **Implement** Retrieval-Augmented Generation (RAG) to ground answers in data
✅ **Evaluate** chatbot quality and measure business impact
✅ **Deploy** a chatbot via Gradio or a simple REST API

**Practical Skills:**
- Make API calls to OpenAI (GPT-4) and Anthropic (Claude)
- Manage conversation history as a list of messages
- Stream responses for real-time user experience
- Connect a chatbot to a knowledge base using embeddings

**Business Skills:**
- Identify the right use cases for chatbots vs. human agents
- Calculate cost per conversation and ROI
- Navigate safety and compliance requirements

---

## Slide 5: How LLMs Generate Text

### The Token-by-Token Process

**What is a Token?**

LLMs don't read words — they read *tokens*. A token is roughly 4 characters or ¾ of a word.

```python
# Approximate token counting
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # Works for all modern LLMs

text = "Hello, I am a large language model generating text."
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Token count: {len(tokens)}")
print(f"Tokens: {tokens}")
# Output:
# Text: Hello, I am a large language model generating text.
# Token count: 12
# Tokens: [9906, 11, 358, 1097, 264, 3544, 4221, 1646, 24cayendo, 1495, 13]

# Rule of thumb: 1000 tokens ≈ 750 words
words = 750
approx_tokens = words * (4/3)
print(f"{words} words ≈ {approx_tokens:.0f} tokens")
```

**How Generation Works: Next-Token Prediction**

```
Input:  "The weather today is"
Model predicts probabilities for next token:
  "sunny"  → 32%
  "rainy"  → 28%
  "cloudy" → 18%
  "cold"   → 12%
  "great"  →  6%
  ...

Model samples "sunny" → appends → new input:
"The weather today is sunny"

Model predicts again:
  "and"    → 41%
  "."      → 25%
  "with"   → 18%
  ","      → 11%
  ...
```

This process repeats until:
- A stop token is reached
- `max_tokens` limit is hit
- A stop sequence is matched

**Key Parameters That Control Generation:**

| Parameter | Range | Effect | Business Use |
|-----------|-------|--------|--------------|
| **temperature** | 0.0 – 2.0 | Controls randomness | 0 = factual/consistent, 1+ = creative |
| **top_p** | 0.0 – 1.0 | Nucleus sampling | Limits token pool to top-p probability mass |
| **max_tokens** | 1 – 128K | Output length limit | Control costs and response length |
| **frequency_penalty** | -2 to 2 | Discourages repetition | +0.5 helps avoid repetitive text |
| **presence_penalty** | -2 to 2 | Encourages new topics | Useful for creative tasks |

```python
from openai import OpenAI

# Free: Ollama runs locally — install at ollama.com, then: ollama pull llama3.2
client = OpenAI(
    api_key="ollama",               # Ollama doesn't need a real key
    base_url="http://localhost:11434/v1"
)

# Low temperature = consistent, factual
response_factual = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    temperature=0.0,   # deterministic
    max_tokens=50
)

# High temperature = creative, varied
response_creative = client.chat.completions.create(
    model="llama3.2",
    messages=[{"role": "user", "content": "Write a tagline for a coffee shop."}],
    temperature=1.2,   # more creative
    max_tokens=50
)

print("Factual:", response_factual.choices[0].message.content)
print("Creative:", response_creative.choices[0].message.content)
```

**Visual: Temperature Effect**

```
Temperature = 0.1 (Deterministic):
"The capital of France is Paris."
"The capital of France is Paris."
"The capital of France is Paris."
(Same answer every time)

Temperature = 1.0 (Balanced):
"Paris is the capital of France."
"France's capital city is Paris."
"The capital of France is Paris, known for the Eiffel Tower."
(Similar but varied phrasing)

Temperature = 1.8 (Creative/Risky):
"Paris, jewel of Europe!"
"France? Oh, Paris of course!"
"The romantic city of Paris crowns France as its capital."
(More varied, sometimes less accurate)
```

---

## Slide 6: The Chat Completion Format

### How Modern LLMs Structure Conversations

**The Message Object**

Every message in a conversation has two fields:
- **role**: `"system"`, `"user"`, or `"assistant"`
- **content**: The text of the message

```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful customer service agent for Acme Corp."
    },
    {
        "role": "user",
        "content": "I ordered a product 2 weeks ago and it hasn't arrived."
    },
    {
        "role": "assistant",
        "content": "I'm sorry to hear your order hasn't arrived! Let me help..."
    },
    {
        "role": "user",
        "content": "My order number is AC-12345."
    }
    # The model will generate the next assistant message
]
```

**The Three Roles Explained:**

```
┌─────────────────────────────────────────────────────┐
│  SYSTEM (invisible to user)                         │
│  "You are a helpful assistant for Acme Corp.        │
│   Only answer questions about our products.         │
│   Be concise and professional."                     │
└───────────────────┬─────────────────────────────────┘
                    │ sets persona & rules
                    ▼
┌─────────────────────────────────────────────────────┐
│  USER                  ASSISTANT                    │
│  "Hi, what products  → "We offer widgets,           │
│   do you sell?"         gadgets, and gizmos."       │
│                                                     │
│  "How much is a      → "Our standard widget is      │
│   widget?"              $29.99. Would you like      │
│                         to place an order?"         │
└─────────────────────────────────────────────────────┘
```

**Complete API Call Example:**

```python
from openai import OpenAI

# Free: Ollama runs locally — install at ollama.com, then: ollama pull llama3.2
client = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1"
)

response = client.chat.completions.create(
    model="llama3.2",
    messages=[
        {
            "role": "system",
            "content": "You are a knowledgeable financial analyst assistant. "
                       "Provide concise, accurate information about financial topics. "
                       "Always remind users to consult a licensed professional for advice."
        },
        {
            "role": "user",
            "content": "What is the difference between a stock and a bond?"
        }
    ],
    temperature=0.3,
    max_tokens=300
)

# Accessing the response
message = response.choices[0].message
print("Role:", message.role)       # "assistant"
print("Content:", message.content)

# Usage statistics
usage = response.usage
print(f"\nTokens used:")
print(f"  Prompt tokens:     {usage.prompt_tokens}")
print(f"  Completion tokens: {usage.completion_tokens}")
print(f"  Total tokens:      {usage.total_tokens}")

# Cost: $0.00 — Ollama runs locally, no API costs!
print(f"\nEstimated cost: $0.00 (Ollama is free/local)")
```

**Response Structure:**

```python
# Full response object breakdown
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1709500000,
  "model": "llama3.2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "A stock represents ownership..."
      },
      "finish_reason": "stop"  # "stop", "length", or "content_filter"
    }
  ],
  "usage": {
    "prompt_tokens": 85,
    "completion_tokens": 142,
    "total_tokens": 227
  }
}
```

**finish_reason Values:**

| Value | Meaning | Action |
|-------|---------|--------|
| `"stop"` | Model finished naturally | Normal — response is complete |
| `"length"` | Hit `max_tokens` limit | Increase `max_tokens` if needed |
| `"content_filter"` | Safety filter triggered | Review prompt; may need to rephrase |
| `"tool_calls"` | Model wants to call a function | Handle tool call (advanced) |

---

**End of Batch 1 (Slides 1-6)**

*Continue to Batch 2 for OpenAI API, Anthropic API, and Prompt Engineering (Slides 7-11)*
