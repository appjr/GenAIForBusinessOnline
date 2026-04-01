# Week 08 — Professor Teaching Scripts
# Batch 2 of 5: Top Paid LLMs (Slides 7–12)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 7 — The LLM Market Map
### 🎤 Professor Script

"Alright, now that we have the history, let's look at the landscape as it exists today.

I want you to think about this market in two dimensions. The vertical axis goes from proprietary — meaning the company keeps the model weights secret — down to open source, where the weights are publicly available. The horizontal axis goes from paid to free.

In the top section, the proprietary world, you have the big names. OpenAI with ChatGPT. Anthropic with Claude. Google with Gemini. And then Perplexity, which is a different category — more of a search engine built on top of LLMs.

Down in the open-source world, you have Meta with Llama. Mistral from France. Google's Gemma. Microsoft's Phi. And then platforms like Groq and Ollama that let you run these open models without building your own infrastructure.

Now here's something important I want you to internalize: the line between paid and free is blurring fast. Almost every paid model has a free tier. And almost every open-source model can be accessed for free through one platform or another. The real distinction is about control, privacy, and quality ceiling.

When you're choosing an LLM for a business use case, you're navigating this map. You're asking: How much do I want to pay? How much control do I need over my data? What quality level does this task require? And how much setup am I willing to do?

Keep this map in mind as we go through each player. By the end, you'll know exactly where to land on it for any given situation."

---

## Slide 8 — OpenAI / ChatGPT
### 🎤 Professor Script

"Let's start with the market leader. OpenAI.

OpenAI was founded in 2015 as a non-profit AI safety research lab. They had a mission to ensure that artificial general intelligence benefits all of humanity. Fast forward a decade, they've taken thirteen billion dollars from Microsoft, they have a consumer product with hundreds of millions of users, and they're valued at over a hundred and fifty billion dollars. Non-profit origins, very for-profit reality.

Their flagship model right now is GPT-4o — and that 'o' stands for omni, meaning it handles text, images, audio, and eventually video natively. It's fast, it's capable, and it's the most widely integrated AI model in the world.

They also have the o-series — o3 and o3-mini — which are their reasoning models. These models are different. Before they answer, they essentially think out loud — they work through the problem step by step. This makes them dramatically better at complex math, science, and multi-step reasoning. The trade-off is they're slower and more expensive.

On pricing: the free tier of ChatGPT gives you GPT-4o-mini, which is actually pretty capable. ChatGPT Plus at twenty dollars a month gives you full GPT-4o access, image generation with DALL-E, and access to the GPT store. ChatGPT Pro at two hundred dollars a month unlocks o1 Pro mode with essentially unlimited usage.

For the API — which is how developers integrate OpenAI into their own products — you pay per token. Roughly two-fifty to fifteen dollars per million input tokens depending on the model.

One thing I want to flag on privacy: on the standard ChatGPT plan, your conversations may be used to train future models. If you're putting confidential business information into ChatGPT on a personal account, be aware of that. The Enterprise plan has zero data retention, but that starts at thirty dollars per user per month.

OpenAI's biggest strength isn't just the model — it's the ecosystem. Three million custom GPTs built by users. Integrations with thousands of third-party apps. The widest developer adoption by far. We'll come back to this when we talk about the tools ecosystem."

---

## Slide 9 — Anthropic / Claude
### 🎤 Professor Script

"Now let's talk about Anthropic and Claude. And this is one I'm particularly interested in, because Anthropic has a fascinating origin story.

In 2021, a group of researchers left OpenAI — including some of the most senior people at the company. Their concern? That the pace of AI development was outrunning our ability to ensure it was safe. So they founded Anthropic with an explicit mission: AI safety research and building AI that is reliably helpful, harmless, and honest.

Their model family is called Claude — currently Claude 4. And there are three tiers within it. Opus is the most powerful, designed for complex analysis and deep reasoning. Sonnet is the balanced one — highly capable but faster and cheaper than Opus. And Haiku is the lightweight, high-speed model for tasks where you need quick responses at scale.

What makes Claude genuinely different from ChatGPT? A few things.

First, the context window. Claude supports up to two hundred thousand tokens. To put that in perspective, that's roughly a hundred and fifty thousand words — you could feed it an entire novel, a full year of emails, or an entire codebase. GPT-4o goes to a hundred and twenty-eight thousand. That gap matters enormously for tasks like contract review, document analysis, or working with large datasets.

Second, instruction following. Claude is exceptionally precise at doing exactly what you ask. If you give it a structured format, a word limit, a specific tone — it follows those instructions more reliably than most other models.

Third, the Projects feature. Claude remembers context across sessions within a project. You can give it background about your company, your writing style, your preferences — and it retains that every time you open a new conversation. That's genuinely useful for ongoing business work.

Pricing is similar to OpenAI — twenty dollars a month for Pro, thirty for Team. And like OpenAI, they have an Enterprise tier with zero data retention and full admin controls.

The honest limitation of Claude is its smaller ecosystem. Fewer native integrations, no image generation built in, smaller developer community. But on pure model quality — especially for long documents and precise task execution — Claude is exceptional."

---

## Slide 10 — Google Gemini
### 🎤 Professor Script

"Google. The company that invented the Transformer architecture and then watched OpenAI build a billion-dollar business with it before they could ship a consumer product. That's one of the great what-ifs of recent technology history.

Their response is Gemini. And Gemini has something that no other model family has: a one million token context window.

Let me put that in perspective. One million tokens is roughly seven hundred and fifty thousand words. You could feed Gemini an entire library of documents. You could give it a year's worth of customer support tickets. You could give it an entire software codebase. The context window isn't just a technical spec — it changes what's possible.

Gemini comes in four flavors. Ultra is the most capable, designed for the hardest tasks. Pro 1.5 is their balanced workhorse. Flash is optimized for speed and cost efficiency — it's extremely fast and cheap to run via API. And Nano runs on-device — it's what powers AI features on Google Pixel phones without any internet connection.

But Gemini's biggest advantage isn't the model itself. It's the integration. If your organization runs on Google Workspace — and a huge percentage of companies do — Gemini is already inside Gmail, Google Docs, Google Sheets, Google Slides, and Google Meet. You're not adopting a new tool. You're activating AI inside the tools your team already uses every day.

And then there's NotebookLM. I want to spend a moment on this because I think it's genuinely one of the most underrated AI products available right now. You upload your documents — PDFs, YouTube videos, Google Docs, websites — and NotebookLM becomes an expert on your specific content. You can ask it questions, get summaries, generate study guides. And it has a feature called Audio Overview that generates a ten-minute podcast — two AI hosts having a real conversation about your documents. It's remarkable.

Pricing follows the same structure: free tier, twenty dollars a month for Google One AI Premium which includes Workspace Gemini features. Enterprise pricing at thirty dollars per user.

Where Gemini falls short is pure text reasoning. On complex multi-step logic problems, it's still a step behind OpenAI and Anthropic on many benchmarks. But for Google Workspace users and for tasks requiring massive context — Gemini is a serious contender."

---

## Slide 11 — Perplexity AI
### 🎤 Professor Script

"Now, Perplexity is an interesting one because it's not quite in the same category as the others. It's not trying to be the most powerful general-purpose LLM. It's doing something more specific — and it does it exceptionally well.

Perplexity is a search engine powered by AI. The core insight behind it is this: every answer should be verifiable. When you ask Perplexity a question, it searches the web in real time, synthesizes what it finds, and gives you a structured answer with cited sources. Every claim traces back to a link you can click.

This solves the single biggest problem with traditional LLMs for research tasks — hallucination. ChatGPT and Claude don't know what happened yesterday. They have a knowledge cutoff. Perplexity doesn't have that problem because it's always searching.

Under the hood, Perplexity uses multiple models depending on the task — GPT-4o, Claude, and their own model called Sonar. You don't really control which one it uses; it selects automatically.

The free tier is genuinely useful — unlimited basic searches, a handful of Pro searches per day. At twenty dollars a month for Pro, you get unlimited deep searches, the ability to upload files, and image generation.

Where Perplexity shines is market research, competitive intelligence, academic research, and fact-checking. If you need to know what a competitor announced last week, what the current market size of an industry is, or what a regulation actually says — Perplexity is the right tool.

Where it doesn't shine is creative tasks, long-form content generation, or conversational depth. It's a research tool, not a writing assistant.

The business case I make for Perplexity is this: use it alongside ChatGPT or Claude, not instead of them. Start your research with Perplexity to get grounded in facts with sources, then bring those insights to Claude or ChatGPT for deeper analysis and content generation."

---

## Slide 12 — Paid LLMs: Business Comparison
### 🎤 Professor Script

"Alright, let's pull this all together into something you can actually use.

I want you to look at this comparison table not as a ranking — there is no single winner — but as a decision matrix. Different tools win in different scenarios.

Let me walk you through the key rows.

Context window: Claude wins here at two hundred thousand tokens. Gemini wins if you need a million. ChatGPT is the most limited of the group at a hundred and twenty-eight thousand, though that's still enormous for most tasks.

Starting price: They're all twenty dollars a month for the consumer tier. You're not choosing based on price at this level.

Privacy: This is where they diverge meaningfully. Standard ChatGPT trains on your data. Claude and Gemini do not by default. For the Enterprise tier of any of these, you get zero data retention. If you're handling sensitive business information, this matters.

Now let me give you the quick decision guide that I actually use in practice.

If you need to analyze a very long document — a hundred-page report, a thick contract, a large codebase — start with Claude. Two hundred thousand tokens, exceptional instruction following.

If you need images, voice, or video — go to ChatGPT with GPT-4o. OpenAI still leads on multimodal.

If your organization lives in Google Workspace — Gemini. The integration advantage is enormous.

If you need current information with sources — Perplexity. Don't ask ChatGPT what happened last month when Perplexity can tell you with citations.

If you need to solve a genuinely hard reasoning problem — math, science, logic — OpenAI's o3 is in a different league for that specific use case.

These tools are not competing for the same job. The smart business professional keeps two or three of them in their toolkit and knows which one to reach for in which situation. That's what we're building toward tonight."

---
