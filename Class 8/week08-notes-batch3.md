# Week 08 — Professor Teaching Scripts
# Batch 3 of 5: Free & Open-Source LLMs (Slides 13–18)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 13 — What Does "Free" Really Mean?
### 🎤 Professor Script

"Before we dive into the specific free and open-source models, I need to make sure we're all speaking the same language. Because 'free' in the AI world means three very different things, and confusing them will lead you to make bad decisions.

The first meaning is a free API tier. This is where a proprietary model — one that the company keeps locked up — gives you limited usage at no cost. Gemini's free API tier. Claude's free plan. OpenAI's free tier. The model is still owned and controlled by the company. Your data still goes to their servers. You're just not paying in dollars — you're paying with usage limits.

The second meaning is open-weight models. This is fundamentally different. When a company releases open weights, they're making the actual internal parameters of the model — the thing that makes it smart — publicly downloadable. You can take those weights, put them on your own server, and run the model yourself. Meta's Llama, Mistral, Google's Gemma, Microsoft's Phi — these are open-weight models. No API calls. No per-token cost after the initial download. Full control.

And the third meaning is free hosted interfaces. This is the best of both worlds for getting started quickly. Someone else runs the open-weight model on their infrastructure and gives you free access. Ollama lets you run Llama on your own laptop. Groq runs Llama on their custom hardware and gives you a free API. Hugging Face hosts hundreds of models with free tiers.

Now, why does this distinction matter for business? Look at the table on this slide. The data privacy column is the critical one. On a free API tier, your data goes to the provider's servers. On a self-hosted open-weight model, your data never leaves your building. If you're in healthcare, finance, legal, or any regulated industry — that's not a minor detail. That's a compliance requirement.

The cost column matters too. A free API tier is free until you hit the limit. At scale — millions of API calls a month — the per-token cost of proprietary models can become substantial. Open-weight models eliminate that completely. The cost becomes infrastructure, which you likely already have.

Keep these three categories in mind as we walk through the specific models."

---

## Slide 14 — Meta Llama 3
### 🎤 Professor Script

"Let's talk about Meta AI and Llama 3, because this is arguably the most important development in the AI industry since ChatGPT — and it gets far less attention in the mainstream press.

Meta — Facebook's parent company — made a strategic decision that is genuinely unusual in the technology industry. They decided to give away their most advanced AI models for free. Not as a marketing play. As a deliberate business strategy.

Here's their reasoning: Meta's business is advertising. They don't sell AI. So making their AI infrastructure open-source doesn't hurt their revenue. But it does accelerate adoption, builds goodwill with the developer community, and creates a massive ecosystem of people who know how to work with Llama — which indirectly benefits Meta's core platforms.

The result is Llama 3, which comes in several sizes. The smallest — 1 billion and 3 billion parameters — can run on a phone or a Raspberry Pi. Genuinely. An AI model on a device with no internet connection. The mid-size models, around 8 to 11 billion parameters, run well on a modern laptop or a basic gaming PC. And the 405 billion parameter version — the full-size flagship — approaches GPT-4 quality and requires serious computing infrastructure.

Where can you access Llama 3? Multiple places. Meta.ai is their consumer interface — free, no account required, works in the browser. Ollama lets you run it locally on your own machine. Groq hosts it in the cloud and gives you a free API tier with extraordinary speed. Amazon, Microsoft Azure, and Google Cloud all offer it as a hosted option.

The business case for Llama is strongest in two scenarios. First, high volume. If you're making millions of API calls a month, eliminating the per-token cost of proprietary models can save you tens of thousands of dollars. Second, sensitive data. If you're processing employee records, patient information, financial data, or anything that cannot leave your network — running Llama locally means zero data exposure.

The catch is technical complexity. Running your own AI infrastructure requires more setup than signing up for ChatGPT. But the tooling around it — Ollama especially — has gotten remarkably approachable."

---

## Slide 15 — Mistral AI
### 🎤 Professor Script

"Mistral AI is a company I want you to pay attention to, because they represent something important: the idea that you don't need to be bigger to be better.

Mistral was founded in April 2023 in Paris, France. Three founders — all former researchers from DeepMind and Meta. They raised funding from Andreessen Horowitz, Lightspeed, NVIDIA, and others. And within months of founding, they released their first model.

The headline result of Mistral 7B — their first open-weight release — was this: a 7 billion parameter model that outperformed Meta's Llama 2 at 13 billion parameters on nearly every benchmark. Half the size, better performance. How? By being extremely careful about data quality and training methodology rather than just throwing more compute at the problem.

Then they released Mixtral. And this is where it gets technically interesting. Mixtral uses an architecture called Mixture of Experts, or MoE. Instead of activating all of the model's parameters for every single token, Mixtral has multiple specialized sub-networks — the experts — and for each token, it only activates the two most relevant ones. The result is a model that has 45 billion total parameters but only activates 12 billion at a time. You get the quality of a large model at the cost of a much smaller one.

Their product line today covers several tiers. The open-weight models — Mistral 7B and Mixtral 8x7B — you can download and run for free. Their commercial API models — Mistral Small, Medium, and Large — sit in different price and performance tiers. And Codestral is their code-specialized model, which competes directly with GitHub Copilot for software development tasks.

Pricing for their API is among the most competitive in the industry — often fifty to eighty percent cheaper than equivalent OpenAI models for similar quality.

And one more thing worth mentioning: Mistral is European. They're GDPR-compliant by design. Their data centers are in the EU. For European companies or any organization doing business in Europe with data residency requirements — this is a meaningful advantage that goes beyond just model quality."

---

## Slide 16 — Google Gemma & Microsoft Phi-3
### 🎤 Professor Script

"We've covered the big proprietary players and the major open-source players. Now let me introduce you to two models that might be the most underappreciated in the entire landscape — Gemma from Google and Phi-3 from Microsoft.

Both of these are what the industry calls small language models. They're not trying to be the most powerful. They're trying to be the most efficient. And they've succeeded remarkably.

Gemma is Google DeepMind's open-weight model family. The same team that builds Gemini. Think of it as Gemini's open-source cousin. The 2 billion and 7 billion parameter versions can run on a laptop or even a phone. Gemma 2, their second generation, at 9 and 27 billion parameters, is genuinely competitive with models twice its size. There's also CodeGemma, which is specialized for software development.

The business case for Gemma is developers who want to embed Google-quality AI into their own applications without paying API costs, and organizations that need AI to run on-device — no internet required, no latency, no data leaving the machine.

Now, Microsoft Phi-3. This one has a fascinating research story behind it. The team at Microsoft Research asked a question: what if instead of training on all the text on the internet — including a lot of garbage — we train exclusively on very high-quality, textbook-like content? Carefully curated educational material. And then they trained a tiny model on that.

The result is astonishing. Phi-3 Mini has 3.8 billion parameters — small enough to run on a smartphone — and it beats GPT-3.5 on many reasoning benchmarks. Phi-3 Medium at 14 billion parameters approaches GPT-4 performance on some tasks.

Why does this matter practically? If Phi-3 can give you ninety percent of GPT-4's quality at one percent of the compute cost, running locally with zero latency and zero API fees — the math is compelling for a wide range of business applications.

Think about a field sales representative with no reliable internet connection who needs AI assistance on their tablet. Or a medical device that needs to process patient data locally. Or a manufacturing quality control system that can't afford cloud API latency. These are the scenarios where small, efficient, on-device models win decisively."

---

## Slide 17 — Groq & Ollama
### 🎤 Professor Script

"We've talked about the models themselves. Now let's talk about two platforms that change how you run those models — Groq and Ollama. These are different from each other in almost every way, but they both address a real limitation: access.

Let's start with Groq. Groq — spelled G-R-O-Q, not to be confused with Elon Musk's Grok — is a hardware and cloud company. They built a completely custom chip called the Language Processing Unit, or LPU, specifically designed for one thing: running LLM inference as fast as physically possible.

How fast? When I say fast, I mean genuinely fast. Standard cloud APIs from OpenAI or Anthropic typically deliver somewhere between 50 and 80 tokens per second — that's roughly the speed you read text. Groq delivers 250 to 800 tokens per second. That's 5 to 10 times faster. Responses appear essentially instantaneously, even for long outputs.

They run open-source models on this hardware — Llama 3, Mixtral, Gemma — and provide free API access with rate limits, and paid tiers that start at extremely competitive prices.

When does speed matter? More than you might think. Voice interfaces cannot have a two-second delay before responding. Real-time translation cannot lag. Customer service chatbots that feel slow get abandoned. High-throughput data processing pipelines where you're running a thousand API calls in a batch — speed becomes cost. Groq is the answer for all of these.

Now, Ollama is the opposite in almost every dimension. Groq is cloud-hosted, fast, and public. Ollama is local, runs on your own machine, and your data never goes anywhere.

Ollama is a piece of software you install on your Mac, Windows, or Linux machine. You then pull any of a hundred-plus supported open-weight models — Llama, Mistral, Phi, Gemma, and more — and run them locally. The command is simple: 'ollama run llama3.2' and you're chatting with an AI model with zero internet required.

For businesses, Ollama's value proposition is absolute data privacy. Your HR documents, your legal contracts, your financial records — you can run sophisticated AI analysis on sensitive data without that data ever touching an external server. No terms of service to worry about. No data retention policies. No breach risk from a third-party vendor.

The trade-off is hardware requirements and setup time. You need at least 8 gigabytes of RAM for smaller models, preferably 16 or more. And while the setup is much simpler than it used to be, it's still more effort than signing up for a web app.

Together, Groq and Ollama represent the two extremes of the open-source deployment story — one optimized for speed in the cloud, one optimized for privacy on your own hardware."

---

## Slide 18 — Free & Open-Source LLMs: Business Comparison
### 🎤 Professor Script

"Let's bring this section together the same way we did for paid models — with a decision framework you can actually use.

Look at this table. Six platforms, six criteria. Let me walk you through the rows that matter most for business decisions.

Privacy is the most important for regulated industries. Anything you run locally — Llama, Mistral, Gemma, Phi-3 through Ollama — is completely private. Your data never leaves your machine. Groq is in the middle — it's cloud-hosted, but it's Groq's cloud rather than OpenAI's or Anthropic's, and their terms are more permissive. If you have strict data residency or compliance requirements, locally-hosted is the only acceptable answer.

Cost is straightforward. The open-weight models are free — you pay for the hardware to run them, which you likely already have. Groq has a generous free tier and then extremely competitive paid tiers. The free tiers of proprietary models come with daily limits that can be frustrating for serious work.

Setup effort is where the proprietary free tiers win decisively. Signing up for Claude.ai takes ninety seconds. Installing Ollama and pulling a model takes about ten minutes but requires a bit of comfort with the terminal. Running your own infrastructure for a 405B parameter Llama model is a serious engineering project.

Let me give you a practical framework for choosing. If your primary concern is data privacy — run locally with Ollama. If your primary concern is speed for a high-throughput application — use Groq. If your primary concern is cost at scale — open-weight models self-hosted. If you just want to try something out quickly at no cost — the free tiers of Claude or Gemini are the path of least resistance.

And here's the strategic takeaway for this entire section: the existence of high-quality open-source models is putting meaningful downward pressure on the pricing of proprietary models. OpenAI and Anthropic cannot charge whatever they want because Meta and Mistral are giving away competitive alternatives. This is good for every business that uses AI."

---
