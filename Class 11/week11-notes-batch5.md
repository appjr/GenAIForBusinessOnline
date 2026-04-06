# Week 11 — Professor Teaching Scripts
# Batch 5 of 5: Key Trends & Strategic Outlook (Slides 26–31)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 26 — RAG Improvements & Knowledge Management
### 🎤 Professor Script

"RAG — Retrieval-Augmented Generation — is the technique that's making AI actually useful inside enterprises, and it's matured significantly over the last eighteen months.

The core problem RAG solves: your company has knowledge that the model doesn't. Products, policies, client histories, internal processes, research reports. The model trained on the internet doesn't know any of it. You could fine-tune the model on your data, but fine-tuning is expensive, slow, and produces a model that doesn't update when your knowledge does. RAG is the better answer for most enterprise use cases.

The basic mechanism — retrieve relevant documents, inject them into the prompt, generate an answer grounded in those documents — is well established. What's improved is everything around the basic mechanism.

Hybrid search is a big one. Semantic search — looking for documents by meaning rather than keywords — is powerful but imperfect. Combining it with traditional keyword search gets significantly better retrieval accuracy, especially for domain-specific terminology that the embedding model hasn't seen much of.

Reranking adds a second pass: after initial retrieval gets you twenty candidates, a lightweight model scores them for relevance to the specific question and reorders them. The top five you actually put in the prompt are now much more likely to be the right ones.

GraphRAG, developed by Microsoft Research, is particularly interesting for complex enterprise knowledge. Traditional RAG treats documents as isolated chunks. GraphRAG understands the relationships between entities — a product and its specifications, a customer and their history, a regulation and the policies that implement it — and retrieves based on those relationships. For questions that require connecting information across multiple documents, GraphRAG dramatically outperforms flat retrieval.

The code example at the bottom of the slide is what this looks like in practice. LlamaIndex plus a vector store — Pinecone or Chroma for cloud, PGVector if you're already on PostgreSQL — is the standard stack. Twenty lines of code to index a directory of documents and start querying them."

---

## Slide 27 — Fine-Tuning Accessibility
### 🎤 Professor Script

"Fine-tuning used to be expensive, technically complex, and out of reach for most organizations. That's changed dramatically.

Let me first be precise about when you actually need it, because I see organizations fine-tuning when they should be prompting.

If you need the model to follow a very specific output format consistently — not sometimes, always — fine-tuning is the right tool. If you need the model to adopt a specific tone or style across thousands of outputs and prompting isn't getting there, fine-tuning. If you're calling the API at extremely high volume and want to use a cheaper, faster small model that still performs well on your specific task, fine-tuning.

If you just need the model to have access to your company's documents or recent information — use RAG. If you need better performance on a complex reasoning task — try better prompts or a stronger base model first.

For cases where fine-tuning is the right answer, here's what's available.

OpenAI's fine-tuning API is approachable. You format your training examples as JSON — instruction and ideal response pairs — and submit a training job. Fine-tuning GPT-4o-mini is a good starting point: the small model becomes surprisingly capable on your specific task, and it's much cheaper to call at inference time than the large model.

For open-source models, LoRA and QLoRA have made fine-tuning accessible on consumer hardware. LoRA modifies a small fraction of model weights — you're not training hundreds of billions of parameters, you're training a small adapter that sits on top of the base model. QLoRA adds four-bit quantization, which cuts memory requirements dramatically. Fine-tuning a 7 billion parameter model on a single consumer GPU in a few hours is now routine. Google Colab's free tier has enough GPU to run QLoRA experiments.

Tools like Unsloth, which the code example references, have simplified this further — optimized kernels that make QLoRA training two to five times faster than the naive implementation.

The business opportunity: a fine-tuned domain model that you own and control, running on your infrastructure, outperforming generic models on your specific tasks. That's a competitive moat."

---

## Slide 28 — Building Your AI Stack
### 🎤 Professor Script

"Let's talk strategy. How does an organization actually go from zero to meaningful AI capability without making expensive mistakes?

I've seen organizations try to skip straight to Tier 3 — enterprise AI with custom fine-tuned models and multi-agent workflows — without doing the foundation work first. It almost always fails. Not because the technology doesn't work, but because they don't understand their own needs well enough yet to make good decisions about what to build.

The three-tier framework on this slide reflects what actually works.

Tier 1 is individual productivity tools. ChatGPT, Claude, Cursor, Perplexity. Give your people access to these tools and time to experiment. This is where you discover which use cases actually matter for your organization. It's also where you build the AI fluency that makes everything else work. Cost is low — twenty to fifty dollars per person per month. This phase should happen immediately, in parallel with everything else.

Tier 2 is team workflows. Once you've identified high-value use cases from Tier 1, you start automating them at the workflow level. A Zapier or n8n integration that handles your most common document processing task. A RAG chatbot that answers questions from your internal knowledge base. A daily competitive intelligence briefing generated automatically. This is where you start seeing ROI that's measurable in productivity terms.

Tier 3 is enterprise AI — custom models, agent orchestration, AI-integrated products. This is where the investment is high and the payoff is high. But you earn the right to build here by proving value in Tiers 1 and 2 first.

The mistakes to avoid are worth spending time on. Vendor lock-in is real — don't build your entire stack assuming one LLM provider. Use abstraction layers. 'Automating broken processes' is the one I see most often — teams hoping AI will fix a workflow that's fundamentally disorganized. It won't. Fix the process first, then automate it.

And never skip the evaluation step. Build a testing framework for your AI outputs before you deploy them. What constitutes a good response? How do you measure it? Without this, you can't improve and you can't trust the system."

---

## Slide 29 — Key Trends to Watch
### 🎤 Professor Script

"Let me give you the trends I'm watching most closely, and what I think they mean for your careers.

The collapse of AI costs is the most underappreciated trend in this space. When GPT-4 was released in 2023, it cost sixty dollars per million input tokens. Today, GPT-4-class performance costs under a dollar per million tokens from some providers. That's a sixty-fold cost reduction in two years. If this trajectory continues — and there's no reason to think it won't — AI inference will approach near-zero cost within three to five years.

What does this mean? Cost is no longer the barrier. The barriers are trust, integration, and governance. Organizations that solve those problems will deploy AI at scale.

Agentic AI moving to production is happening now, not in the future. The demos were impressive eighteen months ago. Today, organizations are running agents that handle real customer interactions, that process thousands of documents without human review, that execute multi-step workflows end-to-end. The question has shifted from 'can agents do this?' to 'how do we safely deploy agents at scale?'

Domain specialization is the trend that I think creates the most interesting opportunity for you specifically. General models are powerful, but a model fine-tuned on your industry — finance, healthcare, legal, logistics — outperforms them on industry-specific tasks. And domain expertise is required to fine-tune well. You can't produce a good financial AI without understanding finance. This is where deep domain knowledge becomes a competitive advantage in AI.

The AI governance trend is real and accelerating. The EU AI Act is being enforced. Companies are investing in AI governance functions. 'Responsible AI' has moved from an ethical statement to a compliance requirement. If you can combine technical AI skills with policy and governance knowledge, you're in a very small and very valuable category."

---

## Slide 30 — Career Opportunities
### 🎤 Professor Script

"Let's be direct about career implications, because that's ultimately why you're in this program.

The table on this slide shows emerging roles that largely didn't exist three years ago. The salaries are US market data as of early 2026 — meaningful, and growing. But I want to focus less on specific titles and more on the skills that compound.

First: Python plus API integration. The LLM API call is now the fundamental unit of AI development. If you can write Python that calls an LLM API, parses the response, and does something useful with it — that's the baseline. Everything else builds on this. If there are gaps in your Python fluency, fix them now.

Second: prompt engineering and evaluation. Not the superficial 'prompt hacking' type — the systematic, measurement-driven practice of designing prompts, defining what good output looks like, and measuring the system against that definition. Companies are building evaluation frameworks — 'evals' — as a first-class engineering practice. The people who can design these rigorously are in demand.

Third: RAG and vector databases. This is the backbone of enterprise AI right now. Understanding how to index documents, tune retrieval, and build reliable Q&A systems on proprietary data is a skill that translates to almost any industry.

Fourth: domain expertise, amplified. Here's the counterintuitive insight: as AI handles more of the generic analytical work, domain expertise becomes more valuable, not less. The AI that can analyze healthcare data well needs healthcare professionals guiding it. The AI that produces valuable financial analysis needs finance professionals evaluating it. Your business analytics training plus AI fluency is a combination that's genuinely scarce.

For your portfolio: build two or three projects that demonstrate these skills. An agent that does something useful. A RAG system on a real document corpus. A fine-tuned model on a domain-specific task. Put them on GitHub. Write about what you learned. That portfolio matters more than certifications."

---

## Slide 31 — Closing
### 🎤 Professor Script

"Let me close with something I want you to carry with you beyond this class.

We've covered a lot of ground tonight. Models, agents, coding tools, automation, multimodal AI, trends. It can feel overwhelming — there are so many tools, the landscape changes so fast, how do you possibly keep up?

Here's the reframe I find most useful: you don't need to know all the tools. You need to understand the patterns. The agent loop is the same whether it's Claude SDK, OpenAI Agents, or LangGraph. RAG is RAG regardless of which vector database you use. A fine-tuning job is structurally the same whether the base model is GPT or Llama. When you understand the patterns, learning new tools is fast — you're mapping new syntax to familiar concepts.

The six takeaways on this slide capture what I most want you to leave with tonight.

The model landscape is rich and competitive. Use that competition to your advantage — the prices keep falling, the quality keeps improving.

Agents are here, not coming. If you haven't built one yet, that changes after this week's homework.

AI coding tools are no longer optional for anyone who writes code professionally. Pick one and go deep.

Business automation converges with AI — if there's a repetitive, document-heavy process in your organization, it's automatable now.

Multimodal AI opens up entirely new workflows that were impossible twelve months ago.

And the 80/20 rule applies: AI can handle 60–70% of most knowledge work tasks. The highest value you add is in the 30–40% that requires genuine judgment, domain expertise, and human accountability.

The quote on the slide sums it up: 'The opportunity isn't to use AI for everything. It's to identify the 20% of your work where AI creates 80% of the value — and go deep there.'

Your homework this week will ask you to do exactly that — identify a real business process and automate part of it. Full details in the assignment.

See you next week."

---
