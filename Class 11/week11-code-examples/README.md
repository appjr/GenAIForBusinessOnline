# Week 11 — Code Examples: New AI Tools and Trends
## BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

This directory contains three production-patterned Python examples that demonstrate the key technical concepts from Class 11.

---

## Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install anthropic chromadb sentence-transformers python-dotenv

# Optional for web scraping examples
pip install requests beautifulsoup4

# Create your .env file (never commit this!)
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

Get a free Anthropic API key at: https://console.anthropic.com

---

## Examples

### Batch 1: Model Comparison (`batch1/slide06_model_comparison.py`)

**Topic:** Slide 6 — Model Selection Framework

Demonstrates how to call multiple LLM providers from Python with a unified interface. Measures and compares latency, token usage, and estimated cost across models.

**What you'll learn:**
- How to structure model API calls for comparison
- Measuring latency and token usage programmatically
- Cost estimation for different model tiers

**Run it:**
```bash
python batch1/slide06_model_comparison.py
```

**Expected output:** A side-by-side comparison table showing response quality, latency, and cost for Claude Haiku vs. Claude Sonnet on a business analysis prompt.

---

### Batch 2: Business Intelligence Agent (`batch2/slide13_business_agent.py`)

**Topic:** Slide 13 — Building Your First Agent

A complete, production-patterned AI agent with 4 tools: URL fetching, market data lookup, business calculations, and report saving. Demonstrates the full agent loop.

**What you'll learn:**
- How to define tools using the Anthropic tool schema
- The agent loop: plan → call tool → observe → continue
- Tool dispatch pattern for production agents
- How the model decides which tools to call

**Run it:**
```bash
python batch2/slide13_business_agent.py
```

**Expected output:** The agent performs market research, calculates CAGR, and saves a formatted report to `genai_market_brief.md`.

---

### Batch 3: RAG Pipeline (`batch3/slide26_rag_pipeline.py`)

**Topic:** Slide 26 — RAG Improvements & Knowledge Management

A complete RAG pipeline: document ingestion → chunking → embedding → vector storage → retrieval → LLM-grounded response generation.

**What you'll learn:**
- Document chunking with overlap
- Local embeddings with sentence-transformers (no API needed)
- ChromaDB vector storage
- Retrieval-augmented prompt construction
- How to cite sources in RAG responses

**Run it:**
```bash
pip install chromadb sentence-transformers  # additional deps
python batch3/slide26_rag_pipeline.py
```

**Expected output:** Q&A session demonstrating that the model answers questions from the provided documents, with source citations, and refuses to answer questions outside the document corpus.

---

## Key Concepts Demonstrated

| Concept | File | Slide |
|---------|------|-------|
| Multi-model comparison | batch1/slide06_model_comparison.py | 6 |
| Tool definition (JSON schema) | batch2/slide13_business_agent.py | 8–9, 13 |
| Agent loop (tool use + iteration) | batch2/slide13_business_agent.py | 7, 13 |
| Document chunking | batch3/slide26_rag_pipeline.py | 26 |
| Vector embeddings | batch3/slide26_rag_pipeline.py | 26 |
| RAG retrieval + generation | batch3/slide26_rag_pipeline.py | 26 |
| Source attribution in RAG | batch3/slide26_rag_pipeline.py | 26 |

---

## Extending These Examples

**Add more tools to the agent:**
```python
# In slide13_business_agent.py, add to TOOLS list:
{
    "name": "send_slack_message",
    "description": "Send a message to a Slack channel",
    "input_schema": {
        "type": "object",
        "properties": {
            "channel": {"type": "string"},
            "message": {"type": "string"}
        },
        "required": ["channel", "message"]
    }
}
# Then add a handler in execute_tool()
```

**Add your own documents to RAG:**
```python
# In slide26_rag_pipeline.py:
from pathlib import Path

# Add any .txt files to the ./documents/ folder, then:
for filepath in Path("./documents").glob("*.txt"):
    content = filepath.read_text()
    n = store.add_document(filepath.name, content)
    print(f"Indexed {filepath.name}: {n} chunks")
```

**Switch to a different LLM:**
```python
# Using LiteLLM for provider-agnostic calls:
pip install litellm

import litellm
response = litellm.completion(
    model="gpt-4o-mini",  # or "gemini/gemini-2.0-flash" or "claude-haiku-4-5-20251001"
    messages=[{"role": "user", "content": "Your prompt here"}]
)
```

---

*Week 11 Code Examples — BUAN 6v99 Generative AI for Business — Spring 2026*
*University of Texas at Dallas | Professor Antonio de Pádua Paes Jr.*
