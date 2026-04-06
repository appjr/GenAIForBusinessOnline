# Week 11 — New AI Tools and Trends
# Batch 1 of 5: Opening + The AI Model Landscape (Slides 1–6)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 1: New AI Tools and Trends

### BUAN 6v99 — Generative AI for Business
### Class 11 | University of Texas at Dallas | Spring 2026

**Professor:** Antonio de Pádua Paes Jr.

---

> *"We are not in a period of gradual improvement. We are in a period of compounding transformation — where every month rewrites what's possible."*

**Today we cover five sections:**
- The AI model landscape in early 2026 — who's leading and why
- AI agent frameworks — how AI now takes actions, not just answers questions
- AI coding tools — Claude Code, Cursor, Copilot, Windsurf compared
- Business automation and multimodal AI — n8n, Make, Zapier, video generation
- Key trends — RAG, fine-tuning, building your AI stack, career outlook

**The goal:** Leave with a complete, actionable map of the 2026 AI ecosystem

---

## Slide 2: Today's Agenda

### Class Overview — 2 Hours

| # | Section | Topics | Time |
|---|---------|--------|------|
| 1 | AI Model Landscape | GPT, Claude, Gemini, DeepSeek, Llama — what's new and how to choose | 25 min |
| 2 | AI Agent Frameworks | Claude SDK, OpenAI Agents, LangGraph, CrewAI, MCP protocol | 25 min |
| 3 | AI Coding Tools | Claude Code, Cursor, Windsurf, GitHub Copilot — live comparison | 20 min |
| 4 | Business Automation | n8n, Make, Zapier AI, vision/audio/video — real workflows | 20 min |
| 5 | Key Trends | RAG 2026, fine-tuning access, your AI stack, career paths | 20 min |

**Wrap-up:** Building your personal AI stack + homework preview

**What's different about this class:** We're not just surveying tools — we're building judgment about when to use each one, at what cost, and for what business outcome.

---

## Slide 3: The Five-Layer AI Ecosystem

### How the Industry Organizes Itself

| Layer | What It Is | Key Players | Where Value Lives |
|-------|-----------|-------------|-------------------|
| **Layer 5: Applications** | End-user products people interact with daily | ChatGPT, Claude.ai, Gemini, Perplexity, Copilot | User experience + workflow integration |
| **Layer 4: Agent Frameworks** | Orchestration of multi-step AI workflows | Claude Agent SDK, OpenAI Agents, LangGraph, CrewAI | Business process automation |
| **Layer 3: Integration Tools** | Connecting models to data and external systems | MCP, LangChain, LlamaIndex, RAG Pipelines | Enterprise data access |
| **Layer 2: Foundation Models** | The raw intelligence — trained on massive data | GPT-4.1, Claude 3.7, Gemini 2.5, Llama 4, DeepSeek | Capability benchmarks |
| **Layer 1: Infrastructure** | Compute, cloud, and specialized AI chips | NVIDIA H100, AWS, Azure, GCP, Groq | Speed and cost |

**The critical insight for business strategists:**
- Layers 1 and 2 are rapidly commoditizing — multiple providers offer similar capability at falling prices
- Competitive advantage has moved to Layers 3, 4, and 5 — how you connect AI to your data and workflows
- The companies winning in AI are not the ones with the best models — they're the ones with the best integration

---

## Slide 4: Frontier Models — The Big Three in 2026

### OpenAI, Anthropic, Google: What's New and Why It Matters

**OpenAI — Reasoning at Scale**

| Model | Capability | Best For | Cost |
|-------|-----------|---------|------|
| GPT-4.1 | Best general-purpose, fast | Everyday tasks, broad applications | ~$2–8/M tokens |
| o3 | Deep reasoning, thinks before answering | Math, complex logic, multi-step analysis | ~$15/M tokens |
| o4-mini | Near-o3 quality, 10× cheaper | Reasoning tasks at scale | ~$1–3/M tokens |

**Anthropic — Coding and Long Context**

| Model | Capability | Best For | Cost |
|-------|-----------|---------|------|
| Claude 3.7 Sonnet | #1 on coding benchmarks, extended thinking | Software development, document analysis | ~$3–15/M tokens |
| Claude 3.7 + Computer Use | Controls browser, clicks, fills forms | GUI automation, legacy system access | API add-on |
| Claude Haiku | Very fast and cheap | High-volume simple tasks | ~$0.25–1.25/M tokens |

**Google — Multimodal and Massive Context**

| Model | Capability | Best For | Cost |
|-------|-----------|---------|------|
| Gemini 2.5 Pro | 1 million token context, native multimodal | Analyzing entire codebases, long documents | ~$3–15/M tokens |
| Gemini 2.0 Flash | Extremely fast, cheap, good quality | Real-time applications, high volume | ~$0.10–0.40/M tokens |

> **Key 2026 shift:** The capability gap between providers has narrowed dramatically. The decision framework is no longer "which is best?" — it's "which fits my specific task, data requirements, and budget?"

---

## Slide 5: The Open Source Revolution

### DeepSeek, Llama, Mistral — Why This Changes Everything

**The January 2025 Inflection Point: The DeepSeek Moment**

DeepSeek (a Chinese AI lab) released two open-source models in early 2025 that matched the best proprietary models — trained for a reported **$6 million**, compared to hundreds of millions spent by OpenAI and Google. The market reacted: NVIDIA's stock dropped 10% in one day.

**Leading open-source models as of April 2026:**

| Model | Creator | Size | Benchmark vs. Proprietary | Can Run Locally? |
|-------|---------|------|--------------------------|-----------------|
| **Llama 4 Scout** | Meta | 17B active (109B total MoE) | Matches GPT-4o on most tasks | Yes (requires 24GB+ GPU) |
| **DeepSeek V3 / R1** | DeepSeek | 671B (MoE) | V3 ≈ GPT-4o; R1 ≈ o1 | Via API or self-hosted |
| **Mistral Large 2** | Mistral AI | 123B | Strong multilingual, coding | Via API or self-hosted |
| **Qwen 2.5 / QwQ-32B** | Alibaba | 7B–72B | QwQ-32B strong reasoning | Yes — runs on consumer GPU |
| **Phi-4** | Microsoft | 14B | Outperforms models 5× larger | Yes — runs on a laptop |

**What open source unlocks for business:**
- **Privacy:** Entire pipeline stays inside your infrastructure — no data sent to any cloud provider
- **Cost at scale:** Zero per-token fees once deployed; fixed infrastructure cost only
- **Customization:** Fine-tune on proprietary data with no sharing or licensing restrictions
- **Audit and compliance:** Full visibility into model behavior — required in regulated industries
- **No vendor dependency:** Not subject to pricing changes, API outages, or terms-of-service shifts

---

## Slide 6: Choosing the Right Model — Decision Framework

### A Systematic Approach for Business Use Cases

**Step 1: Match capability to task type**

| Task Category | Top Choice | Runner-Up | Why |
|---------------|-----------|-----------|-----|
| Complex reasoning & analysis | Claude 3.7 / o3 | Gemini 2.5 Pro | Step-by-step thinking, high accuracy |
| High-volume text processing | Gemini 2.0 Flash | GPT-4.1-mini | Cheap + fast at scale |
| Software development & coding | Claude 3.7 Sonnet | GPT-4.1 | Consistently #1 on coding benchmarks |
| Very long documents (>100K words) | Gemini 2.5 Pro | Claude 3.7 | 1M token context window |
| Sensitive / regulated data | Llama 4 / Phi-4 (local) | Mistral (self-hosted) | Data never leaves your infrastructure |
| Real-time search + synthesis | Perplexity Pro | GPT-4.1 + web tool | Up-to-date, cited information |
| Image / video / audio understanding | Gemini 2.5 Pro | GPT-4o | Native multimodal from the ground up |
| Budget-constrained high volume | Claude Haiku | Gemini Flash | Sub-$1 per million tokens |

**Step 2: Apply constraint filters**

- Is data **sensitive or regulated**? → Self-hosted open source (Llama, Phi-4, Mistral)
- Is **cost per call** the primary constraint? → Flash / Haiku / o4-mini tier
- Does quality of **reasoning** matter most? → o3, Claude 3.7 extended thinking, Gemini 2.5 Pro
- Do you need **real-time information**? → Perplexity, or GPT-4.1 with web search enabled
- Is this a **one-time prototype** or **production at scale**? → Different cost/quality trade-offs apply

**The architect's rule:** Never hard-code a model provider. Use an abstraction layer (LiteLLM, LangChain, or your own wrapper) so you can swap models with a single config change. The best model today is not the best model in six months.

---
