# Week 08 — Professor Teaching Scripts
# Batch 5 of 5: Use Cases + Wrap-up (Slides 26–31)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 26 — Choosing the Right LLM: A Decision Framework
### 🎤 Professor Script

"Alright, we've covered a lot of ground tonight. History. Models. Tools. Now let's bring it all together into something actionable — a decision framework you can actually use tomorrow morning.

The most common mistake I see people make with AI tools is loyalty. They pick one tool — usually ChatGPT because it was first — and they use it for everything. That's like using a hammer for every job because it was the first tool you found. Hammers are great for nails. They're terrible for screws.

So here is how I think about choosing the right tool.

Start with the task. What are you actually trying to do? If you're writing content — blog posts, emails, marketing copy — ChatGPT or Claude are your workhorses. If you need current information with citations — something that happened last month, market data, a competitor's recent announcement — Perplexity is the right answer. If you're working with a very long document — a hundred-page contract, an entire research report — Claude with its two-hundred-thousand token context window is where you go. If you're coding, or want to build something — Cursor or GitHub Copilot. If your entire team lives in Google Workspace — Gemini is already there, already integrated, already paid for.

Then check your budget. The good news is that the free tiers today are genuinely capable. Gemini free, Claude free, and Meta AI on the open-source side — none of these would have seemed possible two years ago. If you're paying twenty dollars a month for any one tool, you're in the premium tier. If you need to go enterprise, that's where Microsoft Copilot 365 or the enterprise versions of Claude and OpenAI come in.

And then — this is the one most people skip — think about your data. If what you're putting into the AI is public information or non-sensitive, any cloud LLM is fine. If it's confidential business strategy, customer data, employee records, anything regulated — you need to either use an enterprise plan with zero data retention, or you need to run locally with Ollama. This is not optional in regulated industries. It's a compliance requirement.

The framework is simple: task first, budget second, data sensitivity third. In that order."

---

## Slide 27 — Use Case: Content Creation & Marketing
### 🎤 Professor Script

"Let me give you a concrete look at what AI-powered content creation actually looks like in practice, because I think this is where most business professionals will see the fastest and most tangible return.

McKinsey published research in 2024 showing that marketing teams using AI produce three to five times more content at sixty percent lower cost. That is not a marginal improvement. That is a structural change in how marketing gets done.

What can AI actually create? Pretty much everything. Blog posts of a thousand to three thousand words — high quality, with editing — in minutes instead of hours. Social media posts for every platform, calibrated to the right tone and format for each. Email campaigns and drip sequences that would take a copywriter days to write. Ad copy for Google, Meta, LinkedIn. Product descriptions for e-commerce catalogs with hundreds of SKUs. Video scripts. Press releases. Presentation decks.

Now I want to be honest about what 'high quality' means here. AI doesn't produce publish-ready content on a first draft the way a great human writer might. What it produces is an excellent starting point that requires editing — but that editing is much faster than starting from scratch. The comparison isn't AI versus a great writer. The comparison is AI-plus-human-editor versus a human writer working alone. The AI-plus-human combination wins on speed and volume almost every time.

The workflow I recommend looks like this. Start by briefing the AI thoroughly — give it your brand voice, your audience, your goal, any key points you want to hit. Then ask it for three variants, not one. You want options. Then a human editor refines the best version, fact-checks it, adds anything proprietary or anecdotal that the AI can't know, and publishes.

I'll give you a real example. A five-person marketing team at a mid-size B2B company used Claude and Jasper to increase their blog output from four posts a month to twenty posts a month without adding a single headcount. Organic traffic grew a hundred and forty percent in six months. That's not a hypothetical. That's what's happening in the market right now."

---

## Slide 28 — Use Case: Research & Analysis
### 🎤 Professor Script

"Research is one of the highest-leverage applications of AI for business professionals, and I think it's dramatically underused relative to its potential.

Let me paint you a picture of the traditional research process. You need to understand a market. You start searching. You find twenty articles. You read them all. You take notes. You synthesize. You write. You go back to check your citations. You realize you missed something. You search again. From start to a solid written analysis — four to eight hours. For a complex topic, more.

The AI-augmented version looks like this. You open Perplexity and do a targeted search on the market, the competitors, the trends. It searches the web in real time and gives you a structured synthesis with clickable citations in minutes. Then you take the most important documents — PDFs, reports, articles — and put them into NotebookLM. It reads everything, connects the dots, and you can ask it questions across all fifty sources simultaneously. Then you bring your notes to Claude for deeper analysis — Claude can hold your entire research brief in context and write a structured output. Total elapsed time: thirty to sixty minutes.

Let me break down what each tool does best in a research workflow.

Perplexity is where you start for anything current. Market size, competitor activity, regulatory changes, recent news — Perplexity searches the web in real time and cites its sources. You can verify every claim.

NotebookLM is your research analyst. You feed it your documents — and it can handle up to fifty sources — and then you interrogate them. Ask it to find everything the documents say about pricing strategy. Ask it to build a timeline of industry events. Ask it to generate a list of questions you should be asking. And then generate the audio podcast — two AI hosts discussing your documents — which is remarkable for consuming dense material on a commute.

Claude is where you go for synthesis and analysis of long, complex documents. Feed it an entire annual report. Ask for the five biggest risk factors. Feed it a legal contract and ask where the liability language is unusual. Two hundred thousand tokens means you almost never hit a limit."

---

## Slide 29 — Use Case: Coding & Development
### 🎤 Professor Script

"I want to make a bold claim and then back it up with evidence.

The claim is this: in 2025, a business analyst with no prior coding experience can build a working, production application using AI tools. Not a toy prototype. A real application.

That would have been false in 2022. It is true today.

Here's how.

For professional software developers, the change is already well-documented. GitHub Copilot has nearly two million subscribers. Developers using it code fifty-five percent faster on average. That's the equivalent of getting a hundred developers and having them suddenly become a hundred and fifty-five. The economic impact on software teams is enormous.

But I want to focus on the people in this room who are not developers. Because I think the opportunity is actually bigger for you.

If you are a business analyst and you need to process ten thousand rows of customer data — segment it, clean it, transform it, summarize it — you can ask Claude or ChatGPT to write you a Python script to do it. You don't need to understand Python. You need to be able to describe what you want clearly, and then evaluate whether the output is correct.

If you are a finance professional and you need a complex Excel macro to automate a monthly reporting process, you can ask an AI to write the VBA code. Paste it into Excel. Test it. If something's wrong, paste the error message back to the AI. Iterate until it works.

If you are a product manager and you want to test whether an idea is viable before spending engineering resources — Replit AI runs in the browser. You can describe an application in plain English, have the AI build it, and demo it to stakeholders — all without a single developer involved.

The new core skill for business professionals is what I call AI-mediated technical communication. The ability to describe what you want in precise enough terms that an AI coding tool can execute it, and then evaluate the result. That is a learnable skill. And it is now a career differentiator."

---

## Slide 30 — Use Case: Business Automation
### 🎤 Professor Script

"Let's talk about what happens when you connect an LLM to the rest of your business software. Because the individual productivity gains we've been discussing are meaningful. But the organizational gains from automation are an order of magnitude larger.

Let me give you three concrete workflows that organizations are running right now.

First: customer email management. A customer sends an email to support. An AI — running through something like n8n or Zapier — reads the email, categorizes it as a billing issue, a technical question, or a complaint. It searches the knowledge base for relevant answers. It drafts a response. It routes the email to the right team. It logs everything in the CRM. A human reviews the draft, edits if needed, and approves it. What used to take fifteen minutes of human attention per email — reading, researching, composing, routing — is now three minutes of review.

Second: sales operations. A sales call ends. The call recording goes to an AI. The AI transcribes it, extracts the action items, identifies what the prospect said their key concerns are, and updates the Salesforce deal record automatically. It drafts a follow-up email for the sales rep to review and send. The rep opens their laptop and sees their entire call already documented and their email already written.

Third: HR onboarding. A new employee starts. An AI onboarding bot is available twenty-four hours a day to answer questions — where's my benefits portal, who do I call for IT, what's the vacation policy. It generates a personalized thirty-day learning plan based on the employee's role. It tracks completion of required training and flags gaps to the manager automatically.

Now look at the ROI table on this slide. Customer service: forty to sixty percent of tier-one tickets resolved without human involvement. Sales: five to ten hours per rep per week freed up. HR: three to five hours saved per hire. Finance: six to eight hours a week on invoice processing and expense categorization. Marketing: ten to fifteen hours a week on content repurposing.

These are not speculative numbers. These are what organizations deploying these tools are reporting. The question is no longer whether AI automation is real. The question is which department you're going to start with."

---

## Slide 31 — Wrap-Up + Your 8 Exercises
### 🎤 Professor Script

"Two hours. Thirty-one slides. Let me give you the five sentences that capture everything.

The Transformer architecture, published by Google researchers in 2017, is the foundation of every LLM you will interact with. The ChatGPT moment in November 2022 changed the trajectory of the technology industry overnight — a hundred million users in two months. You have two broad categories of models to choose from: paid and proprietary, which are the easiest to start with and the most capable; and free and open-source, which give you privacy, customizability, and cost control at scale. The real value creation is not in the models themselves — it's in the tools built on top of them that fit into the workflows of your specific job and industry. And your most important skill as a business professional in this environment is not knowing how to build AI — it's knowing how to choose the right tool for the right task.

Now, for this week, you have eight exercises. I want you to actually do them. Not read about them — do them. Run Ollama on your laptop. Use Groq's free API. Put a document into NotebookLM. Compare Meta AI to ChatGPT on the same prompt. These tools are free to access. The exercises are designed to take thirty to sixty minutes each. By the time you've done all eight, you'll have hands-on experience with every major category of LLM we discussed tonight.

And then next class — I want you to come in ready to talk about what surprised you. What was better than you expected. What was worse. Where you think the gaps are. Because that critical perspective — not just enthusiasm, but evaluated judgment — is what separates someone who uses AI from someone who understands it.

The question you should be asking yourself from now on is never 'should I use AI for this?' That question is over. The answer is almost always yes.

The question is: which AI, for this task, at this cost, with this data sensitivity level?

That's the question of a professional. And that's what we've been building toward all night.

See you next class."

---
