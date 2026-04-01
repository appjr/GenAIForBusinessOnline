# Week 08 — Professor Teaching Scripts
# Batch 1 of 5: Opening + LLM History (Slides 1–6)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 1 — The World of Large Language Models
### 🎤 Professor Script

"Good evening everyone, and welcome to Class 8.

Tonight's topic is one I'm genuinely excited to teach, because I think it's going to change how you see the entire AI landscape — not just the tools you use, but why they exist, who made them, and how to make smart decisions about which one to reach for in any given situation.

We're talking about Large Language Models. LLMs.

Now, I know that term gets thrown around constantly. You've probably used ChatGPT. Maybe you've tried Claude or Gemini. But there's a big difference between using a tool and understanding it. And tonight, we're going to do both.

Here's the journey we're taking together over the next two hours. We're going to start at the very beginning — 2017, a research paper that changed everything. Then we're going to walk through who the major players are today, what tools have been built on top of these models, and — this is the part that matters most for your careers — how you actually choose the right tool for the right business problem.

By the end of tonight, you won't just know what ChatGPT is. You'll know when to use it instead of Claude. You'll know why a company might run an AI model on their own servers instead of paying for an API. You'll understand the ecosystem.

Let's get into it."

---

## Slide 2 — Today's Agenda
### 🎤 Professor Script

"Let me walk you through what we're covering tonight so you know exactly where we're headed.

We have six sections. We'll start with a quick foundation — what is an LLM, in plain terms, no jargon. That's about ten minutes.

Then we go into history. I love this part. The story of how we got from a 2017 Google research paper to a product that had a million users in five days is one of the most dramatic stories in the history of technology. We'll spend about twenty-five minutes there.

After that, we look at the paid, proprietary LLMs — ChatGPT, Claude, Gemini, Perplexity. Twenty minutes. We go platform by platform and I'll show you what each one is actually good at, because they are not the same tool.

Then we flip to the free and open-source side. Llama, Mistral, Ollama, Groq. Another twenty minutes. This is where things get really interesting for businesses that care about privacy or cost at scale.

Then we look at the tools ecosystem — everything built on top of these models. The productivity tools, the coding assistants, the enterprise software, and the no-code automation platforms. That's another twenty-five minutes.

And we close with use cases. Concrete scenarios. How do you actually choose the right LLM for your specific business problem? That's thirty minutes, and it includes a preview of your homework assignment.

So — a full two hours. Let's make every minute count."

---

## Slide 3 — What Is a Large Language Model?
### 🎤 Professor Script

"Before we go anywhere, let's make sure we have a solid foundation.

What is a Large Language Model? I want you to understand this in a way you can explain to a colleague who's never heard the term.

Three words: Large. Language. Model. Each one means something specific.

Large means it has been trained on an enormous amount of data — we're talking about a significant fraction of all text ever published on the internet, plus books, scientific papers, code, and more. And it has billions — sometimes hundreds of billions — of internal parameters. Think of parameters as the knobs the model adjusts during training to get better at predicting what comes next.

Language means it operates on text. It reads text in, and it generates text out. Now, modern models can also handle images, audio, and video — but text is still the foundation.

And Model means it's a mathematical system. At its core, what an LLM is doing is predicting the most statistically likely next word given everything that came before it. That's it. It's a very sophisticated autocomplete. The magic is that when you train this on enough data with enough parameters, something extraordinary emerges: the model learns to reason, to write coherently, to follow instructions, to solve problems.

Why does this matter for business? Because tasks that used to require expensive human expertise — drafting contracts, analyzing reports, writing code, answering customer questions — can now be done at scale, in seconds, for fractions of a cent per query.

That's the foundation. Now let's talk about how we got here."

---

## Slide 4 — History 2017–2019: The Spark
### 🎤 Professor Script

"The story of LLMs starts with a paper.

In 2017, a team at Google Brain published a research paper called 'Attention Is All You Need.' Eight authors. Forty-three pages. And it changed everything.

The key innovation was something called the Transformer architecture, and specifically a mechanism called self-attention. What self-attention allows a model to do is learn which words in a sentence relate to which other words — no matter how far apart they are. Previous approaches — the RNNs and LSTMs we used before — had to process text sequentially, word by word, and they struggled to maintain context over long distances. The Transformer blew that limitation away.

Every major LLM you interact with today — ChatGPT, Claude, Gemini, Llama — is built on this architecture. Every single one.

Then in 2018, two important things happened almost simultaneously.

Google released BERT. BERT was designed for understanding text — reading it and extracting meaning — not generating it. It's bidirectional, meaning it reads in both directions at once. Google used BERT to massively improve their search results, and they still use descendants of it today.

And OpenAI released GPT-1. The first Generative Pre-trained Transformer. It had 117 million parameters and it could generate coherent paragraphs. Proof of concept that this approach worked.

A year later, GPT-2. One and a half billion parameters — more than ten times bigger. And it was so good at generating convincing text that OpenAI initially refused to release it publicly. They said it was 'too dangerous.' That's an interesting moment to reflect on — the first time a lab delayed an AI release out of concern about misuse.

That's 2017 to 2019. The spark is lit. Now watch what happens when we pour fuel on it."

---

## Slide 5 — History 2020–2022: The Scaling Era
### 🎤 Professor Script

"2020 is when things get wild.

OpenAI releases GPT-3. One hundred and seventy-five billion parameters. That's a hundred times bigger than GPT-2. And the jump in capability is not linear — it's exponential. Something happens when you scale a model to this size that nobody fully expected.

The model develops what researchers call emergent abilities. Capabilities that nobody explicitly programmed. GPT-3 could translate languages it wasn't specifically trained to translate. It could do basic arithmetic. It could write code. It learned these things from reading enough text about them.

The other breakthrough in GPT-3 was few-shot learning. You could put three or four examples of a task right in your prompt, and the model would understand the pattern and follow it. This made LLMs practically useful for a huge variety of tasks without any specialized training.

And then in 2021, OpenAI takes GPT-3 and fine-tunes it on code — billions of lines of GitHub repositories — and releases it as Codex. Codex becomes the engine behind GitHub Copilot. For the first time, a professional developer can describe what they want in plain English and get working code back. That's a massive economic disruption for software development.

Also in 2021, DALL-E. OpenAI uses LLM-style training to generate images from text descriptions. The multimodal era begins.

Now, 2022. This is the year of a critical technique called RLHF — Reinforcement Learning from Human Feedback. Instead of just predicting the next token, models start learning to be helpful. Human trainers rate responses. The model learns what 'good' looks like from a human perspective. This is the breakthrough that makes LLMs usable by non-experts.

And on November 30th, 2022, OpenAI releases ChatGPT — built on GPT-3.5. One million users in five days. One hundred million in two months. The fastest product adoption in the history of technology. The world changes overnight."

---

## Slide 6 — History 2023–2025: The Modern Era
### 🎤 Professor Script

"2023 is the year that everything accelerates at once.

In March, OpenAI releases GPT-4. It passes the bar exam. It scores in the 90th percentile on the SAT. It writes code, analyzes images, reasons through complex problems. Near-human expert performance on a huge range of tasks.

But here's what's interesting — OpenAI is no longer alone.

Anthropic releases Claude. A company founded by ex-OpenAI researchers who left over concerns about safety. Claude is built differently — with a strong emphasis on being harmless and being honest. And critically, it debuts with a 100,000 token context window. That's enormous. You can feed it entire books.

Google releases Bard, which eventually becomes Gemini. They're playing catch-up, but they have an enormous advantage — integration with every Google product billions of people already use.

And then in July 2023, Meta does something that shakes the entire industry. They release the weights for Llama 2. Open source. Free to download. Anyone can run a state-of-the-art LLM on their own servers, for free, with no API calls, no per-token cost, full privacy. The open-source movement explodes.

In September, Mistral releases a 7 billion parameter model that outperforms Llama 2 at 13 billion. Efficiency becomes the new arms race.

2024 brings multimodal everywhere and something new — reasoning models. OpenAI releases o1 and then o3, models that actually think before they answer. They spend extra compute working through a problem step by step. The quality on hard math, science, and logic problems improves dramatically.

And now, in 2025, we're in the era of agents. LLMs that don't just answer questions — they take actions. They browse the web, write files, call APIs, run code. And the cost? What cost twenty dollars per million tokens in 2023 costs ten cents today. The economics are changing as fast as the technology.

That's where we are. Now let's meet the players."

---
