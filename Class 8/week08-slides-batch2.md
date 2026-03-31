# Week 08 — The World of Large Language Models
# Batch 2 of 5: Top Paid LLMs (Slides 7–12)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 7: The LLM Market Map

### Navigating the Landscape

```
                    PROPRIETARY (Closed Source)
                           │
          ┌────────────────┼────────────────┐
          │                │                │
       OpenAI           Anthropic        Google
       ChatGPT           Claude           Gemini
       GPT-4o             │             Perplexity
          │         PAID  │  FREE          │
──────────┼───────────────┼────────────────┼──────────
          │               │                │
       Groq API        Mistral          Meta AI
       (Llama)         (open wts)       (Llama 3)
       Ollama           Gemma            Phi-3
          │                │                │
          └────────────────┼────────────────┘
                           │
                    OPEN SOURCE (Open Weights)
```

**Four quadrants to know:**
- **Paid + Proprietary** — Most capable, easiest to use, subscription cost
- **Free Tier + Proprietary** — Limited usage, API keys required
- **Open Weights** — Download and run anywhere, privacy-first
- **Self-hosted** — Full control, requires infrastructure

---

## Slide 8: OpenAI / ChatGPT

### The Market Leader

**Company:** OpenAI (founded 2015, San Francisco)
**Backed by:** Microsoft ($13B investment), other VCs

**Current models (2025):**

| Model | Best For | Context Window | Speed |
|-------|----------|----------------|-------|
| GPT-4o | General tasks, vision, voice | 128K tokens | Fast |
| o3 / o3-mini | Complex reasoning, math, science | 128K tokens | Slow (thinks longer) |
| GPT-4.5 | Creative tasks, nuanced writing | 128K tokens | Fast |

**Pricing (approximate):**
- **Free tier:** ChatGPT with GPT-4o-mini, limited GPT-4o
- **ChatGPT Plus:** $20/month — full GPT-4o access, image generation
- **ChatGPT Pro:** $200/month — o1 Pro, unlimited usage
- **API:** Pay per token (~$2.50–$15 per million input tokens)

**Key strengths:**
- Largest ecosystem (GPT Store, plugins, integrations)
- Best multimodal capabilities (text, image, voice, video)
- Widest enterprise adoption and third-party support

**Watch out for:** Privacy — your data may train future models (without Enterprise plan)

---

## Slide 9: Anthropic / Claude

### The Thoughtful Alternative

**Company:** Anthropic (founded 2021 by ex-OpenAI team)
**Backed by:** Amazon ($4B), Google ($300M), other investors
**Mission:** AI safety and reliability at the frontier

**Current models (2025):**

| Model | Best For | Context Window | Speed |
|-------|----------|----------------|-------|
| Claude 4 Opus | Deep analysis, complex reasoning | 200K tokens | Medium |
| Claude 4 Sonnet | Balanced: capable + fast + affordable | 200K tokens | Fast |
| Claude 3.5 Haiku | Quick tasks, high-volume applications | 200K tokens | Very fast |

**Pricing (approximate):**
- **Free tier:** Claude.ai — limited daily messages
- **Claude Pro:** $20/month — priority access, 5× more usage
- **Claude Team:** $30/user/month — shared workspace, Projects
- **API:** $3–$15 per million input tokens

**Key strengths:**
- **Longest context window** — 200K tokens (≈150,000 words — entire books)
- **Best for long documents** — contracts, research papers, codebases
- **Superior instruction following** — does exactly what you ask
- **Safety-focused** — less likely to produce harmful outputs
- **Projects** — persistent memory across conversations

**Watch out for:** Smaller ecosystem vs. OpenAI; fewer native integrations

---

## Slide 10: Google Gemini

### The Integrated Giant

**Company:** Google DeepMind (division of Alphabet)
**Advantage:** Deep integration with Google Workspace (Docs, Sheets, Gmail)

**Current models (2025):**

| Model | Best For | Context Window | Speed |
|-------|----------|----------------|-------|
| Gemini Ultra | Most complex tasks, research | 1M tokens | Slow |
| Gemini Pro 1.5 | Balanced everyday tasks | 1M tokens | Medium |
| Gemini Flash | High speed, cost-efficient | 1M tokens | Very fast |
| Gemini Nano | On-device, mobile (Pixel phones) | Small | Ultra fast |

**Pricing (approximate):**
- **Free tier:** Gemini.google.com — Gemini Pro access
- **Google One AI Premium:** $20/month — Gemini Ultra + Workspace AI features
- **Workspace Business:** $30/user/month — Gemini in all Google apps
- **API:** Via Google AI Studio (free tier available) or Vertex AI

**Key strengths:**
- **1 million token context window** — largest of any mainstream model
- **Native multimodal** — built for text, images, audio, video simultaneously
- **Google ecosystem lock-in** — works seamlessly in Gmail, Docs, Drive
- **NotebookLM** — extraordinary tool for research and learning
- **Real-time search** — can access current web information

**Watch out for:** Still catching up to OpenAI/Anthropic on pure text reasoning tasks

---

## Slide 11: Perplexity AI

### Search Reimagined

**Company:** Perplexity AI (founded 2022, San Francisco)
**Backed by:** Amazon, NVIDIA, Jeff Bezos (personal)
**Category:** AI-powered search engine, not a traditional chatbot

**What makes it different:**
> "Every answer comes with sources. Every claim is verifiable."

**Current models:**
- Uses multiple LLMs under the hood (GPT-4o, Claude, Sonar — their own model)
- Automatically selects the best model for each query type

**Pricing:**
- **Free tier:** Unlimited searches, limited Pro searches per day
- **Perplexity Pro:** $20/month — unlimited Pro searches, image generation, file uploads

**Key strengths:**

| Feature | Why It Matters |
|---------|----------------|
| **Always up-to-date** | Searches the web in real time — no knowledge cutoff |
| **Source citations** | Every answer links to original sources |
| **Research mode** | Deep multi-step research on complex topics |
| **Academic mode** | Searches peer-reviewed papers (PubMed, arXiv) |
| **Business use** | Market research, competitor analysis, trend tracking |

**Best used for:** Market research, fact-checking, current events, academic research
**Not ideal for:** Creative writing, long-form generation, conversational tasks

---

## Slide 12: Paid LLMs — Business Comparison

### Choosing the Right Paid Tool

| | **ChatGPT (OpenAI)** | **Claude (Anthropic)** | **Gemini (Google)** | **Perplexity** |
|---|---|---|---|---|
| **Best For** | General tasks, images, voice | Long docs, analysis, coding | Google Workspace users | Research, fact-checking |
| **Context Window** | 128K tokens | 200K tokens | 1M tokens | Real-time web |
| **Starting Price** | $20/month | $20/month | $20/month | $20/month |
| **Free Tier** | Yes (limited) | Yes (limited) | Yes (limited) | Yes (good) |
| **Key Strength** | Ecosystem, multimodal | Long context, precision | Google integration | Real-time accuracy |
| **Weakness** | Privacy (standard plan) | Smaller ecosystem | Text reasoning | Not for creation |
| **Privacy** | ⚠️ Trains on data | ✅ No training by default | ⚠️ Google data | ✅ Sources cited |
| **Enterprise Plan** | Yes ($30/user) | Yes ($30/user) | Yes ($30/user) | Yes ($40/user) |

**Quick decision guide:**
- 📄 **Long document analysis** → Claude (200K context)
- 🖼️ **Images + voice + video** → ChatGPT (GPT-4o)
- 📊 **Google Workspace heavy user** → Gemini
- 🔍 **Research + current events** → Perplexity
- 🧮 **Complex math/science reasoning** → OpenAI o3

---
