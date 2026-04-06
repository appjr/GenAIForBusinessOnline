# Week 11 — New AI Tools and Trends
# Batch 5 of 5: Key Trends & Strategic Outlook (Slides 26–31)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 26: RAG in 2026 — Beyond Basic Retrieval

### From Prototype to Production-Grade Knowledge Systems

**What RAG does:** Connects an LLM to your proprietary data so it answers questions grounded in *your* knowledge — policies, products, client history, research — not just its training data.

**Why RAG beats fine-tuning for most knowledge use cases:**

| Criterion | RAG | Fine-tuning |
|-----------|-----|------------|
| **Updates when data changes** | Yes — re-index the new document | No — requires retraining |
| **Cites sources** | Yes — provenance is built in | No — model "knows" it but can't cite |
| **Cost to implement** | Low to medium | High |
| **Time to production** | Days to weeks | Weeks to months |
| **Handles proprietary facts** | Yes | Yes |
| **Requires ML expertise** | No | Yes |

**The 2025–2026 RAG improvements that matter for production:**

| Advancement | What It Does | When You Need It |
|-------------|-------------|-----------------|
| **Hybrid search** | Combines keyword + semantic similarity | When your documents use technical jargon or product names |
| **Reranking** | Second model re-scores retrieved chunks for relevance | When retrieval quality is inconsistent |
| **GraphRAG** | Understands entity relationships across documents | Complex, interconnected knowledge (org charts, product dependencies) |
| **Multimodal RAG** | Retrieves from PDFs with charts, tables, images | Real business documents — most have mixed content |
| **Agentic RAG** | Model decides what to retrieve, and can do multiple retrieval passes | Multi-hop questions requiring information from many places |

**Minimal production RAG stack (2026):**
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
# Index once
docs = SimpleDirectoryReader("./company_docs/").load_data()
index = VectorStoreIndex.from_documents(docs)
# Query many times
response = index.as_query_engine().query(
    "What is our enterprise refund policy and how does it differ from SMB?"
)
```

**Cost benchmark:** Indexing 1,000 documents ≈ $0.50. Answering a query ≈ $0.02–$0.10.

---

## Slide 27: Fine-Tuning Accessibility in 2026

### From Research Lab to Business Tool

**Fine-tuning definition:** Continue training a pre-trained model on your specific data — producing a model that reflects your domain language, output style, and task requirements.

**When fine-tuning is the right answer (and when it isn't):**

| Scenario | Best Approach | Why |
|----------|--------------|-----|
| Need factual knowledge from your documents | RAG | Fine-tuning doesn't reliably memorize facts |
| Need consistent output format or writing style | **Fine-tuning** | Style is baked in — no per-call prompting needed |
| Domain-specific terminology and reasoning | **Fine-tuning** | Model learns your language, not just your content |
| Running 10M+ calls/month, cost is critical | **Fine-tuning on a small model** | Fine-tuned small model outperforms large model on your task at 1/10th the cost |
| Data is fresh / changes frequently | RAG | Re-indexing is instant; retraining is slow |

**What's accessible in 2026 — no PhD required:**

**OpenAI Fine-Tuning API (cloud):**
```python
# Format: JSONL file with {"messages": [...]} examples
# Minimum: ~50 high-quality examples; optimal: 200–500+
job = client.fine_tuning.jobs.create(
    training_file="file-abc123",
    model="gpt-4o-mini"     # Fine-tune the cheap model to match GPT-4o quality on your task
)
# Cost: ~$3–8 per 1M training tokens; inference 2–4× cheaper than base model
```

**LoRA / QLoRA (local, open source):**
- **LoRA:** Trains a small adapter on top of frozen base weights — 90% fewer trainable parameters, same quality improvement
- **QLoRA:** LoRA + 4-bit quantization — fine-tune a 7B model on a single consumer GPU in 1–3 hours
- **Unsloth:** Library that makes QLoRA training 2–5× faster; runs free on Google Colab

**Real business fine-tuning ROI examples:**
- Law firm: Fine-tuned on 2,000 past contracts → model drafts in firm's exact clause style → partner review time −60%
- E-commerce: Fine-tuned on brand voice guide + 500 product descriptions → 10,000 on-brand SKU descriptions/day
- Call center: Fine-tuned on 3 years of resolved tickets → model drafts replies matching senior agent quality → handle time −35%

---

## Slide 28: Building Your AI Stack — A Tiered Roadmap

### From Individual Tools to Enterprise AI in 5 Steps

**The three-tier maturity model:**

| Tier | What You Build | Tools | Monthly Cost | Timeline |
|------|---------------|-------|-------------|---------|
| **Tier 1: Individual Productivity** | Personal AI assistants for daily work | Claude/ChatGPT/Gemini, Cursor/Copilot, Perplexity | $0–$50/person | Start now |
| **Tier 2: Team Workflows** | Automated processes shared across the team | Zapier/Make/n8n, NotebookLM, RAG chatbot on your docs | $100–$1,000/month | Months 1–6 |
| **Tier 3: Enterprise AI** | AI-native products and internal platforms | Fine-tuned models, multi-agent systems, AI-embedded products | $10,000+/month | Months 6–18 |

**The five-step adoption playbook:**

1. **Map:** List your 10 most time-consuming, low-judgment repetitive tasks. These are automation candidates.
2. **Pilot:** Pick 2–3. Use existing AI tools ($0 incremental cost) to assist with them for 2 weeks.
3. **Measure:** Quantify time saved per task. Calculate weekly/monthly hours. Document quality differences.
4. **Scale:** Once value is proven, formalize the workflow, automate with n8n/Zapier, roll out to the team.
5. **Invest:** With a track record, now consider custom fine-tuning, agents, or product integration.

**The five mistakes that derail AI adoption:**

| Mistake | What Goes Wrong | The Fix |
|---------|----------------|---------|
| Starting at Tier 3 | "We need an enterprise AI platform" before proving any use case | Tier 1 costs $20/month and proves value in week 1 |
| Vendor lock-in | Deep dependency on one LLM provider; painful to migrate | Use LiteLLM or LangChain as provider abstraction from day one |
| No evaluation framework | "It seems good" is not a quality standard | Define what good looks like, build a test set, measure it |
| Ignoring data privacy | Sensitive data in cloud AI APIs before legal review | Audit data classification; self-host for regulated data |
| Automating broken processes | Hoping AI will fix a workflow that was already chaotic | Redesign the process first, then automate the clean version |

---

## Slide 29: Six Trends Reshaping Business AI — 2026 and Beyond

### What to Watch, What It Means for You

**Trend 1: Intelligence cost collapses toward zero**
- GPT-4 level in 2023: $60/million tokens → Today: $0.50–$2.00 → Projection 2027: $0.10 or less
- Implication: Cost is no longer the barrier to AI deployment — **trust, integration, and governance** are

**Trend 2: Agentic AI moves from demo to production**
- 2024: Impressive demos. 2025: Pilots. 2026: Production at scale at companies like Klarna, JPMorgan, Salesforce
- Agents completing 4-hour knowledge work tasks in 15 minutes is now observable, not aspirational

**Trend 3: Domain specialization creates competitive moats**
- General models → domain models: Harvey (legal), Abridge (clinical), BloombergGPT (finance)
- Companies that fine-tune on proprietary data create models competitors can't replicate
- Your domain expertise × AI = a differentiated capability, not just efficiency

**Trend 4: Multimodal becomes the baseline**
- By 2027, the distinction between "text AI" and "vision AI" will be meaningless — all models are multimodal
- Every business process touching documents, images, audio, or video becomes automatable
- 85%+ of business information is currently unstructured — this unlocks most of it

**Trend 5: AI governance transitions from voluntary to mandatory**
- EU AI Act enforcement: companies using AI in hiring, credit, healthcare, law enforcement face compliance requirements
- Enterprise buyers now require: audit logs, explainability reports, data lineage, bias testing
- New job titles appearing: Chief AI Ethics Officer, AI Risk Manager, Model Governance Lead

**Trend 6: The human role shifts from executor to director**
> The highest-value professionals in 2027 will be those who combine **deep domain expertise** with **AI fluency** — who know what to ask AI to do, can validate what it produces, and can connect AI outputs to business strategy. Neither pure AI skill nor pure domain knowledge alone will be enough.

---

## Slide 30: Career Opportunities in the AI Era

### Where Business Analytics Students Fit in 2026

**Roles that barely existed in 2022 — and are now in high demand:**

| Role | What They Do Day-to-Day | Salary Range (US) | Key Skill Stack |
|------|------------------------|-------------------|----------------|
| **AI Product Manager** | Define AI features, manage LLM-based products, write evals | $140–$210K | Product instinct + LLM behavior understanding |
| **AI Automation Engineer** | Build agent workflows, connect LLMs to business systems | $100–$165K | Python, LangChain/n8n, REST APIs, cloud |
| **Prompt Engineer / LLM Engineer** | Design prompts, build evals, optimize model behavior | $90–$155K | LLM internals, eval frameworks, domain expertise |
| **AI Data Curator** | Build training and evaluation datasets for fine-tuning | $80–$135K | Domain expertise + annotation tooling + quality metrics |
| **ML Ops / AI Ops Engineer** | Deploy, monitor, and maintain production AI systems | $130–$185K | Cloud, MLflow, monitoring, cost optimization |
| **AI Governance Analyst** | Ensure AI compliance, conduct risk assessments | $90–$145K | Policy + risk + AI literacy |
| **Business AI Analyst** | Find automation opportunities, measure ROI, build business cases | $80–$135K | Analytics + business acumen + AI tools fluency |

**The five skills that compound most in this market:**

1. **Python + API integration** — the LLM API call is the fundamental unit of AI product development; everything builds on this
2. **RAG and vector databases** — the backbone of every enterprise AI deployment; LlamaIndex, LangChain, ChromaDB, Pinecone
3. **Prompt design + evaluation** — systematic, measurable approach to getting consistent AI outputs; not art, it's engineering
4. **Agent orchestration** — building workflows where AI takes actions, not just answers; Claude SDK, LangGraph, n8n
5. **Domain expertise × AI** — knowing your industry deeply means you can direct AI better than a generalist; this is your moat

**Your portfolio action plan:**
- Build 2–3 AI projects: one RAG app, one agent, one automation pipeline — put them on GitHub
- Write about what you built and what you learned — LinkedIn posts, a blog, anything public
- Get certified: Anthropic, OpenAI, Google Cloud AI, AWS Machine Learning — all have free or cheap certifications

---

## Slide 31: Class Wrap-Up — Your AI Landscape Map

### Six Things to Walk Away Knowing

**What today's class gave you:**

**1. The model landscape is rich, competitive, and fast-moving**
GPT-4.1, Claude 3.7, Gemini 2.5, and open-source models like Llama 4 and DeepSeek all compete at the frontier. Choose by use case, data requirements, and cost — not by brand loyalty. The best model changes every 3–6 months.

**2. Agents are the most important architectural shift since ChatGPT**
Claude SDK, OpenAI Agents SDK, LangGraph, and CrewAI let you build AI that takes actions. MCP standardizes how agents connect to every tool in your stack. This is the category where the most value is being created right now.

**3. AI coding tools are no longer optional for anyone who writes code**
Claude Code, Cursor, Windsurf, and GitHub Copilot have moved from productivity boosters to competitive necessities. Writing code without AI assistance is increasingly a disadvantage, not a preference.

**4. Business automation + AI is transforming every operational role**
n8n, Make, and Zapier now orchestrate AI judgment — not just rule-based routing. Any workflow involving document reading, content generation, or classification is automatable today.

**5. RAG and fine-tuning are your tools for customization — not re-training**
You don't build models from scratch. You adapt existing frontier models to your domain. RAG for knowledge. Fine-tuning for style, format, and task-specific performance.

**6. The opportunity is specific, not general**
> *"The opportunity isn't to use AI for everything. It's to identify the 20% of your work where AI creates 80% of the value — and go deep there."*

**Your homework:** Choose a real business process. Automate at least one step of it. Present it next class.

**Next class:** AI in Finance and Business Decision-Making — applying these tools to financial modeling, forecasting, and quantitative analysis.

---
