# Class 11 Homework — Build an AI Automation Pipeline
## BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026
### Professor Antonio de Pádua Paes Jr.

---

## Assignment Overview

You will act as an **AI automation consultant** hired to automate a real business process using the tools covered in Class 11. You will choose a business scenario, design an AI-powered automation pipeline, implement at least one working component using Python or a no-code tool, and present your solution to the class.

**Deliverable:** A 5–7 minute presentation showing your working pipeline (screen share required).

**Due:** Next class session.

---

## Step 1 — Choose Your Business Scenario

Pick **one** scenario below, or define your own with professor approval:

| # | Scenario | Industry |
|---|----------|----------|
| A | Automated competitive intelligence briefing — monitor 3–5 competitor websites and generate a weekly summary | Any |
| B | AI-powered customer support ticket triage — classify, summarize, and draft initial replies | SaaS / Service |
| C | Document Q&A system — build a RAG chatbot that answers questions from a set of company documents | Any |
| D | AI content pipeline — take a topic, generate a blog post, social posts, and a summary email | Marketing |
| E | Automated meeting notes processor — take a transcript and output action items, decisions, and follow-ups | Operations |
| F | Job posting analyzer — analyze 10+ job descriptions to extract required skills, trends, and salary data | HR / Analytics |
| G | Invoice/receipt data extractor — use vision AI to extract structured data from document images | Finance / Ops |
| H | Research synthesizer — given a business question, search and synthesize information into a structured report | Consulting |

> **Custom scenario:** If your career interests point elsewhere, define your own. Requirements: concrete inputs, clear AI-powered processing step, measurable output. Propose via email before starting.

---

## Step 2 — Design Your Pipeline

Before writing any code or building any workflow, map out your pipeline on paper or in a diagram tool.

**Your pipeline design must specify:**

1. **Trigger:** What starts the process?
   - Manual: user types a query / uploads a file
   - Scheduled: runs daily/weekly automatically
   - Event-driven: new email, new form submission, new file

2. **Input:** What data enters the pipeline?
   - Type: text, file, URL, API data, form submission
   - Source: where does it come from?

3. **AI processing step(s):** What does the LLM do?
   - One step or multiple? Sequential or parallel?
   - What model? Why?
   - What's the prompt strategy?

4. **Output:** What does the pipeline produce?
   - Format: structured JSON, natural language text, email, Slack message, saved file
   - Destination: where does it go?

**Template to complete:**
```
Pipeline: [your pipeline name]

TRIGGER:    [what starts it]
INPUT:      [data type and source]
AI STEP 1:  [model + task description]
AI STEP 2:  [if applicable]
OUTPUT:     [format + destination]

Tools used: [Python SDK / n8n / Make / Zapier / other]
Model(s):   [Claude / GPT / Gemini / other]
```

---

## Step 3 — Implement Your Pipeline

Build at least **one working component** of your pipeline. You have two implementation paths:

### Path A: Python (Recommended for technical depth)

Build your pipeline in Python using one of the SDKs from class:

**Minimum requirements:**
- Working Python script that runs end-to-end
- Uses at least one LLM API call (Claude, OpenAI, or Gemini)
- Takes real input and produces useful output
- Has basic error handling (try/except around API calls)

**Starter template:**
```python
"""
Class 11 Homework: AI Automation Pipeline
[Your name] | BUAN 6v99 | April 2026

Pipeline: [your pipeline name]
Scenario: [your chosen scenario]
"""

import anthropic
import os
from typing import Optional

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def process_input(raw_input: str) -> str:
    """
    Step 1: Process raw input with AI.

    Args:
        raw_input: The incoming data to process

    Returns:
        AI-processed result as a string
    """
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # Use Haiku for cost efficiency
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": f"""[Your prompt here]

Input: {raw_input}

Output:"""
            }
        ]
    )
    return response.content[0].text


def format_output(processed_text: str) -> dict:
    """
    Step 2: Structure the AI output into the desired format.

    Args:
        processed_text: Raw AI output

    Returns:
        Structured output dictionary
    """
    # Your formatting logic here
    return {"result": processed_text}


def run_pipeline(input_data: str) -> Optional[dict]:
    """
    Main pipeline orchestrator.
    """
    try:
        print(f"Processing: {input_data[:100]}...")
        processed = process_input(input_data)
        output = format_output(processed)
        return output
    except anthropic.APIError as e:
        print(f"API error: {e}")
        return None


if __name__ == "__main__":
    # Test with a sample input
    test_input = "[your test input here]"
    result = run_pipeline(test_input)
    if result:
        print("\n=== PIPELINE OUTPUT ===")
        print(result)
```

**For Scenario C (RAG) — additional requirements:**
```python
# You'll also need:
pip install llama-index chromadb

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# Index at least 3 real documents
# Demonstrate at least 3 queries
# Show that answers are grounded in the documents
```

---

### Path B: No-Code Tool (n8n, Make, or Zapier)

Build your pipeline using one of the automation tools from class:

**Minimum requirements:**
- Complete, working workflow with at least 3 nodes/steps
- Includes at least one AI step (any LLM)
- Runs successfully on real input data
- Screenshot documentation of the entire workflow

**Tool-specific guidance:**

*n8n (self-hosted or n8n cloud):*
- Use the AI Agent node or OpenAI/Anthropic nodes
- Aim for: Trigger → AI Processing → Output (send email, save file, or post to Slack)

*Make (make.com):*
- Use the OpenAI or HTTP module for AI calls
- Build a scenario with at least 3 modules
- Use the scenario inspector to capture a test run

*Zapier (zapier.com):*
- Use the AI Actions step or ChatGPT integration
- Create a multi-step Zap with at least 3 steps
- Run and document a real test execution

---

## Step 4 — Document and Test

**For Python pipelines:**

1. Test with at least 3 different inputs — document all 3 results
2. Note any errors or unexpected outputs you encountered
3. Measure actual API cost (Anthropic/OpenAI dashboards show usage)
4. Write a short analysis:
   - What did the pipeline do well?
   - What did it fail to handle?
   - What would you improve with more time?

**For no-code pipelines:**

1. Take screenshots of every node/step configuration
2. Run the pipeline 3 times and document the outputs
3. Note the cost (if applicable — n8n and Make have free tiers)
4. Write a short analysis (same questions as above)

**Quality evaluation — complete this table:**

| Criterion | Score (1–5) | Evidence |
|-----------|-------------|----------|
| Output accuracy — does it correctly accomplish the task? | /5 | |
| Output completeness — does it cover all the requirements? | /5 | |
| Consistency — does it produce similar quality on different inputs? | /5 | |
| Speed — how long does it take to run? (fast / medium / slow) | /5 | |
| Cost efficiency — is the cost per run acceptable for business use? | /5 | |
| **Total** | /25 | |

---

## Step 5 — Extend Your Pipeline (Bonus)

For additional depth, implement **one or more** of the following extensions:

**Extension A — Multi-step agent (+15 pts):**
Upgrade your Python pipeline to use the full agent loop (from Slide 13). Add at least 2 tools the agent can call.

**Extension B — RAG addition (+15 pts):**
Add a retrieval step to your pipeline — when the AI needs specific information, it retrieves it from a document store rather than relying on training data alone.

**Extension C — Structured output (+10 pts):**
Make your pipeline output structured JSON using Pydantic or response format enforcement. Validate the output schema on every run.

**Extension D — Evaluation harness (+10 pts):**
Build a simple eval script that runs your pipeline on 10 test cases and scores the outputs automatically. Even a simple keyword-check evaluation demonstrates understanding of production AI practices.

---

## Step 6 — Prepare Your Presentation

**Format:** 5–7 minutes, live screen share. No slides required.

| Section | Duration | Content |
|---------|----------|---------|
| **1. Context** | 1 min | What business problem you solved, what tools you used, and why |
| **2. Live demo** | 2–3 min | Run your pipeline live. Show an input, walk through the processing, show the output. |
| **3. Evaluation** | 1 min | Your quality scorecard results. What worked, what didn't. |
| **4. Business case** | 1 min | If deployed in a real company: what would this save? How would you measure it? |

**Strong presentations will:**
- Show the pipeline running, not just describe it
- Be honest about limitations — what does it fail on?
- Make a concrete business case: "This would save X hours per Y period at Z cost"
- Discuss one thing you'd build next if you had more time

---

## Grading Rubric

| Category | Points | What We're Looking For |
|----------|--------|----------------------|
| **Pipeline design** | 15 pts | Clear, logical pipeline diagram with trigger, input, processing, and output specified |
| **Implementation** | 35 pts | Working code or workflow; at least 3 test runs documented; handles errors gracefully |
| **Evaluation quality** | 20 pts | Scorecard completed with evidence; honest analysis of strengths and weaknesses |
| **Presentation** | 20 pts | Live demo works; stays within time; concrete business case; confident delivery |
| **Critical thinking** | 10 pts | Identifies edge cases; discusses cost/quality trade-offs; proposes meaningful improvements |
| **Bonus extensions** | up to +40 pts | See Step 5 |
| **TOTAL** | **100 pts** (+40 bonus) | |

---

## Technical Setup Guide

**Environment setup (do this before you start coding):**

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# Install core packages
pip install anthropic openai python-dotenv

# For RAG (Scenario C)
pip install llama-index chromadb

# For web scraping (Scenario A, H)
pip install requests beautifulsoup4

# Store your API key in a .env file (NEVER commit this to git)
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

**Load your API key safely:**
```python
from dotenv import load_dotenv
import os

load_dotenv()  # Reads .env file
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
```

**Get API keys (all have free tiers):**
- Anthropic: console.anthropic.com → API Keys → Create Key
- OpenAI: platform.openai.com → API Keys
- Google: aistudio.google.com → Get API key

---

## Frequently Asked Questions

**Q: Do I need to spend money on API calls?**
All major providers offer free tiers sufficient for this assignment. Anthropic gives new accounts $5 free credit. OpenAI gives new accounts $5 credit. Google AI Studio is free for experimentation. Using smaller models (Claude Haiku, GPT-4o-mini, Gemini Flash) keeps costs minimal — typically < $0.10 for the entire assignment.

**Q: My pipeline doesn't work perfectly — is that okay?**
Yes. Document what doesn't work and why. Honest analysis of limitations is worth more than a polished demo that hides problems. Real engineering includes knowing what breaks.

**Q: Can I use an LLM API I haven't seen in class?**
Yes — Groq, Mistral, Cohere, Perplexity all have APIs. Document your choice and explain why you chose it.

**Q: Can I use my own employer's real business problem?**
Yes, but do not include confidential, proprietary, or personally identifiable data in API calls to external services. Use realistic but hypothetical/anonymized data.

**Q: How do I handle secrets safely?**
Use a `.env` file and `python-dotenv`. Never hardcode API keys. Never commit `.env` to git (add it to `.gitignore`).

---

*Class 11 Homework — BUAN 6v99 Generative AI for Business — Spring 2026*
*University of Texas at Dallas | Professor Antonio de Pádua Paes Jr.*
