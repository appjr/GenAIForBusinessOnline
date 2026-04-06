# Week 11 — Professor Teaching Scripts
# Batch 1 of 5: Opening + The AI Model Landscape (Slides 1–6)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 1 — New AI Tools and Trends
### 🎤 Professor Script

"Good evening, everyone. Welcome to Class 11.

Tonight we're doing something a little different. Every class in this course, we've focused on a specific technique or concept — neural networks, LLMs, computer vision, fine-tuning. Tonight, we zoom out. We look at the landscape.

Because here's the reality of working in AI in 2026: the tools are moving so fast that the most important skill isn't mastering any single one — it's knowing what exists, being able to evaluate what's worth your time, and building judgment about what to use when.

Think about what's happened just in the last twelve months. New models from every major lab. An entirely new category of 'agentic AI' that barely existed two years ago. A standardized protocol — MCP — that changed how AI tools connect to everything else. AI coding assistants that can complete multi-hour programming tasks autonomously. Video generation that crossed the threshold for real business use.

The pace is not slowing down. If anything, it's accelerating.

So tonight, we're going to cover the full picture — models, agents, coding tools, business automation, multimodal AI, and key trends. By the end of class, you'll have a map of this landscape that you can use immediately in your work and career.

Let's get into it."

---

## Slide 2 — Today's Agenda
### 🎤 Professor Script

"Here's what the next two hours look like.

We start with the model landscape — who the players are, what's new from each major lab, and critically, how to choose the right model for a given business problem. That's section one, about twenty-five minutes.

Then we move into the part I'm most excited about — agentic AI. Section two. AI agents are the biggest shift in how AI is used since ChatGPT launched. We'll cover the major frameworks — Claude Agent SDK, OpenAI Agents SDK, LangGraph, CrewAI — and we'll actually look at code for a working business agent.

Section three is AI coding tools. This is directly relevant to every one of you. Whether you're writing Python for data analysis or building production systems, AI coding tools are now table stakes. We'll compare Claude Code, Cursor, Windsurf, GitHub Copilot, and others.

Section four covers business automation and multimodal AI. n8n, Make, Zapier — the automation tools have all added serious AI capabilities. And multimodal AI — vision, audio, video — has crossed the quality threshold for real business use.

And section five brings it all together: key trends, how to build your AI stack, and the career implications of everything we've covered.

We have a full two hours. Let's use them well."

---

## Slide 3 — The AI Landscape in Early 2026
### 🎤 Professor Script

"Let me orient you with a map before we dive into any specific tool.

I want you to think about the AI ecosystem as having five layers, stacked on top of each other.

The bottom layer — infrastructure — is the GPUs, the cloud providers, the specialized inference chips. NVIDIA's H100s. AWS, Azure, GCP. And newer players like Groq, which uses a completely different chip architecture to deliver AI inference ten times faster than GPU-based systems. This is the power grid. You don't think about it day-to-day, but it's what makes everything above it possible.

Layer two is foundation models. This is the layer you've been studying all semester. GPT-4.1, Claude 3.7, Gemini 2.5, Llama 4, DeepSeek. The raw intelligence. The models themselves.

Layer three is integration tools. This is where raw models get connected to the real world. LangChain, LlamaIndex, RAG pipelines, vector databases. If you've built a chatbot that references your company's documents, you've used this layer.

Layer four — and this is the new frontier — is agent frameworks. This is what we're spending significant time on tonight. Claude Agent SDK, OpenAI Agents SDK, LangGraph. The infrastructure for AI that doesn't just answer questions, but takes actions.

And layer five is applications. ChatGPT, Claude.ai, Perplexity, GitHub Copilot — the products that end users interact with directly.

Here's the key insight about this stack: the competitive advantage keeps moving up. In 2022, having access to GPT-3 was an advantage. Today, the models themselves are commoditizing — multiple providers offer GPT-4-class intelligence at similar price points. The advantage now is in layers four and five — how you orchestrate agents and build applications on top of these models.

That's the game we're all playing."

---

## Slide 4 — Frontier Models — Closed Source
### 🎤 Professor Script

"Let me walk you through what's new from the big three — OpenAI, Anthropic, and Google — as of early 2026.

Starting with OpenAI. The big story of the last year is the o-series — reasoning models. o3 and o4-mini. The concept here is powerful: instead of immediately generating a response, these models actually think through the problem. They spend extra compute — sometimes thirty seconds, sometimes a few minutes for hard problems — working out the reasoning step by step before giving you an answer.

The improvement for hard problems is dramatic. Math competition problems, complex code debugging, multi-step logical reasoning — o3 outperforms GPT-4o significantly on these. The tradeoff is cost and latency. But o4-mini changes that equation — it gets most of o3's quality at about one-tenth the price. For business problems that require careful reasoning, o4-mini is often the right call.

Now Anthropic's Claude. Claude 3.7 Sonnet is, as of today, arguably the best model for coding tasks. It consistently tops coding benchmarks. But the bigger news is extended thinking and computer use. Computer use means you can give Claude a task and it will literally control a computer to complete it — move the mouse, click buttons, fill forms, navigate websites. This is agentic AI made real. We'll explore this more in the agents section.

Google's Gemini 2.5 Pro has one remarkable capability that nothing else matches: a one million token context window. One million tokens. That's roughly 750,000 words. You can feed it an entire codebase. A hundred financial reports. A week of audio transcripts. And it processes all of it in a single call. For tasks that require synthesizing large amounts of information, nothing else comes close.

So three different labs, three different strengths. The model that's right for your task depends on what you're actually trying to do — and we'll look at a decision framework for that in slide six."

---

## Slide 5 — The Open Source Revolution
### 🎤 Professor Script

"Now let's talk about open source, because this is where some of the most dramatic changes have happened.

I want to tell you a story. In January 2025, a Chinese AI lab called DeepSeek released two models: V3 and R1. V3 matched GPT-4o in general performance. R1 matched OpenAI's o1 — the original reasoning model — on reasoning benchmarks. Both were released as open source. Anyone could download and run the weights.

And then came the detail that shook the industry: DeepSeek reportedly trained V3 for approximately six million dollars. Compare that to the hundreds of millions — possibly more — that OpenAI and Google spend training their frontier models. The stock market reacted. NVIDIA's stock dropped ten percent in a day on the news. The assumption that only billion-dollar training runs could produce frontier-class models was directly challenged.

What does this mean practically? It means open source models are now competitive with proprietary ones. And being open source means you can run them yourself.

Think about what that unlocks for a business. Zero per-token cost. Your data never leaves your infrastructure. Full customization — you can fine-tune the model on your proprietary data. And models like Phi-4 from Microsoft — just fourteen billion parameters — run on a high-end laptop. Powerful AI inference without any cloud dependency.

The other key open source development is Meta's Llama ecosystem. Llama 4, the latest release, is multimodal — it handles text and images natively — and at smaller parameter counts, it matches GPT-4o-class performance. Meta has made a strategic bet on open source as a way to prevent AI from being dominated by a single closed-source provider. That's been good for the entire ecosystem.

The message for business: the choice between proprietary and open source is now a legitimate strategic decision, not a quality compromise."

---

## Slide 6 — Choosing the Right Model
### 🎤 Professor Script

"Okay, so we've just catalogued a lot of models. Let me give you a practical framework for actually choosing one.

The first question is always: what's the primary use case? And I want to be specific here, not general. Don't ask 'which model is best' — ask 'which model is best for summarizing legal contracts in a law firm context.'

The table on this slide gives you my current recommendations by task type. But let me highlight a few that come up constantly for business analytics students.

For coding and development work — and this means Python scripting, building data pipelines, writing automation — Claude 3.7 Sonnet is the current leader. It's been at the top of coding benchmarks consistently. If I were building something today that requires high-quality code generation, that's where I'd start.

For high-volume text processing — where you're running thousands or millions of API calls — cost matters a lot. GPT-4.1-mini and Gemini 2.0 Flash are the options here. You get very good quality at a fraction of the cost of flagship models.

For analyzing very long documents — think an entire legal agreement, a full annual report, a long transcript — Gemini 2.5 Pro's one million token context window is the right tool. Nothing else processes that much context reliably in a single call.

For sensitive data — healthcare records, financial data, internal HR information — the right answer may be a self-hosted open source model. Llama 3.3 or Phi-4 running in your own infrastructure, where no data ever leaves.

And then the second layer of the decision: constraints. Privacy? Budget? Reasoning quality? This decision tree walks you through it.

One final point I want to emphasize: the best model today may not be the best model in six months. The pace of improvement is that fast. This is why building with abstraction layers — tools like LiteLLM that let you swap providers with one line of code — is good engineering practice. Don't hard-code a dependency on a specific model. Make it easy to switch."

---
