# Week 08 — The World of Large Language Models
# Batch 1 of 5: Opening + History (Slides 1–6)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 1: The World of Large Language Models

### BUAN 6v99 — Generative AI for Business
### Class 8 | University of Texas at Dallas | Spring 2026

**Professor:** Antonio de Pádua Paes Jr.

---

> *"Language is the interface between human intelligence and machine intelligence.
> Large Language Models are making that interface seamless."*

**Today's journey:**
- From a 2017 research paper → to tools reshaping every industry
- Understanding the players, the tools, and how to choose the right one for your business

---

## Slide 2: Today's Agenda

### Class Overview — 2 Hours

| # | Section | Time |
|---|---------|------|
| 1 | What is an LLM? — Foundation | 10 min |
| 2 | The History of LLMs — From 2017 to Today | 25 min |
| 3 | Top Paid LLMs — ChatGPT, Claude, Gemini, Perplexity | 20 min |
| 4 | Free & Open-Source LLMs — Llama, Mistral, Ollama | 20 min |
| 5 | The Tools Ecosystem — What's Built on Top | 25 min |
| 6 | Use Cases — Choosing the Right LLM for Your Business | 20 min |

**Wrap-up:** Key takeaways + 8 hands-on exercises you'll complete this week

---

## Slide 3: What Is a Large Language Model?

### The Foundation

A **Large Language Model (LLM)** is an AI system trained on massive amounts of text data to understand and generate human language.

**Three defining characteristics:**

| Characteristic | What it means |
|----------------|---------------|
| **Large** | Billions to trillions of parameters (weights learned during training) |
| **Language** | Operates on text — reads, understands, generates, translates, summarizes |
| **Model** | A mathematical system that predicts what text should come next |

**How it works (simplified):**
> You give it text → it predicts the most likely continuation → that's its "answer"

**Why it matters for business:**
- Can perform tasks that previously required expensive human expertise
- Writing, analysis, coding, research, customer service — all at scale
- No programming knowledge required to use

---

## Slide 4: The History of LLMs — 2017–2019: The Spark

### The Transformer Revolution

**2017 — "Attention Is All You Need"** *(Google Brain)*
- A landmark research paper introducing the **Transformer architecture**
- Key innovation: **self-attention** — the model learns which words relate to which other words in a sentence
- Replaced older RNN/LSTM approaches that struggled with long sequences
- Every major LLM today is built on this architecture

**2018 — BERT** *(Google)*
- Bidirectional Encoder Representations from Transformers
- First model to read text in both directions simultaneously
- Revolutionized search engines — Google still uses it today
- Not a generator — designed for understanding, not producing text

**2018 — GPT-1** *(OpenAI)*
- Generative Pre-trained Transformer — 117 million parameters
- Could generate coherent paragraphs for the first time
- Proof of concept: pre-training on large text corpora works

**2019 — GPT-2** *(OpenAI)*
- 1.5 billion parameters — 10× bigger than GPT-1
- So capable OpenAI initially refused to release it publicly ("too dangerous")
- Could write convincing news articles, stories, and code

---

## Slide 5: The History of LLMs — 2020–2022: The Scaling Era

### When Size Became a Superpower

**2020 — GPT-3** *(OpenAI)*
- **175 billion parameters** — a 100× leap from GPT-2
- Introduced **few-shot learning**: give it 3 examples in your prompt and it learns the task
- Showed emergent abilities — capabilities nobody programmed directly (translation, math, code)
- Powered early ChatGPT prototypes and sparked the AI startup boom

**2021 — Codex** *(OpenAI)*
- GPT-3 fine-tuned on code → became the engine behind GitHub Copilot
- First time AI could meaningfully assist professional software developers

**2021 — DALL-E 1 & 2** *(OpenAI)*
- Proved LLM-style training could generate images from text descriptions
- Launched the multimodal AI era

**2022 — InstructGPT / RLHF**
- OpenAI introduced **Reinforcement Learning from Human Feedback (RLHF)**
- Instead of just predicting text, models learned to follow instructions and be helpful
- This was the breakthrough that made LLMs usable by non-experts

**2022 — ChatGPT (Nov 30)** *(OpenAI)*
- First public consumer LLM product built on GPT-3.5
- **1 million users in 5 days** — fastest product adoption in history
- Changed the public conversation about AI forever

---

## Slide 6: The History of LLMs — 2023–2025: The Modern Era

### Competition, Open Source & Multimodal AI

**2023 — The Year of LLMs**

| Event | Impact |
|-------|--------|
| GPT-4 (OpenAI, Mar 2023) | Passed bar exam, SAT — near-human expert performance |
| Claude 1 & 2 (Anthropic) | Safety-focused alternative; 100K token context window |
| Google Bard → Gemini | Google's response to ChatGPT; integrated into Workspace |
| Llama 1 & 2 (Meta, Jul 2023) | Open-source weights released — anyone could run LLMs locally |
| Mistral 7B (Sep 2023) | Small but powerful European open-source model |

**2024 — Multimodal & Reasoning**

| Event | Impact |
|-------|--------|
| GPT-4o (OpenAI) | Voice + vision + text in real-time — "Her" moment |
| Claude 3 Opus/Sonnet/Haiku | Outperformed GPT-4 on many benchmarks |
| Gemini Ultra 1.0 | Google's most capable model; native multimodal |
| Llama 3 (Meta) | Open-source nearly matching proprietary models |
| o1 / o3 (OpenAI) | "Thinking" models — reasoning before answering |

**2025 — Where We Are Today**
- **Agents**: LLMs take actions, not just answer questions
- **Local models**: Run state-of-the-art AI on a laptop (Ollama, LM Studio)
- **Specialization**: Models fine-tuned for medicine, law, finance, coding
- **Price collapse**: What cost $20 in 2023 costs $0.10 today

---
