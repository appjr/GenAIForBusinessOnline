# Week 11 — Professor Teaching Scripts
# Batch 4 of 5: Business Automation & Multimodal AI (Slides 20–25)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 20 — AI Business Automation: The New Landscape
### 🎤 Professor Script

"Let's shift to one of the most immediately practical areas we'll cover tonight — AI-powered business automation.

There are two technologies that have each been transformative individually, and together they're rewriting what business processes can look like. The first is workflow automation — tools like Zapier, Make, and n8n that have existed for years, letting you connect apps and automate repetitive tasks based on rules. If email arrives from a customer, create a support ticket. If a form is submitted, add a record to the CRM. Powerful, but rule-based and rigid.

The second is AI — LLMs that can read and understand unstructured content, reason about it, and generate responses or decisions.

When you combine these two, you get something qualitatively different. Not just 'if email arrives, create ticket' — but 'if email arrives, read and understand it, classify the intent, extract the key information, draft a personalized response, and route it to the right person, all without a human touching it.'

The table on the slide quantifies what this means for common business processes. Invoice processing: 90% time savings. Customer support triage: 70% reduction in human handling. These are real numbers from companies deploying these systems today.

I want to be clear: this doesn't mean these jobs disappear. What it means is that the human who used to spend 80% of their time on mechanical classification and routing tasks now spends 80% of their time on the 20% of cases that are genuinely complex or sensitive. That's a much better job. And those professionals are more productive, not less employed.

The question for anyone in a business analytics role is: where in your organization's processes does this opportunity exist?"

---

## Slide 21 — n8n: Open Source AI Automation
### 🎤 Professor Script

"n8n is the automation tool I recommend most often for organizations that have any sensitivity around data privacy, and for developers who want flexibility beyond what Zapier and Make provide.

The key differentiator is self-hosting. When you use Zapier or Make, your data — customer emails, internal documents, financial records — flows through their servers. For many workflows, that's fine. But for workflows touching sensitive data — HR records, patient information, client financials, proprietary business data — you may not want that.

n8n can run entirely inside your infrastructure. Docker container, your server, your data never leaves. And it's open source, which means the code is auditable and extensible.

The AI capabilities are genuinely impressive. There are native nodes for OpenAI, Claude, and Gemini. There's an AI Agent node that essentially gives you a CrewAI-style agent inside your automation workflow — you define the task, give it tools, and it executes. There's also native Ollama integration, which means you can run a workflow that uses a local LLM on your own hardware — zero external API calls, zero cost per token, complete privacy.

The workflow example on the slide is one I've actually built: intelligent email routing. New email arrives, the AI Agent reads it, classifies it, extracts structured data, and then the workflow branches based on classification. Sales lead? CRM entry, Slack notification to the sales team. Support request? Zendesk ticket with an AI-generated draft reply. It's probably four hours to build from scratch and saves hundreds of hours per year.

The learning curve is higher than Zapier — n8n assumes some technical comfort. But if you have Python skills, the logic of building workflows is very accessible.

The free tier is self-hosted — you pay for the server. Managed cloud starts at twenty dollars a month."

---

## Slide 22 — Make & Zapier AI
### 🎤 Professor Script

"Let's talk about Make and Zapier, because these are the tools most commonly found in non-technical teams — marketing, operations, HR — and both have added serious AI capabilities in the last year.

Make is my recommendation for the middle ground — more powerful and visual than Zapier, less technical than n8n. The flowchart interface is genuinely intuitive. You see the data flowing between steps. Complex branching and error handling are visible rather than hidden.

Make's AI features in 2025 include a natural language scenario builder — you describe what you want the automation to do, and Make builds the first draft. It's not perfect, but it cuts setup time by sixty or seventy percent for common use cases. The data transformation AI is particularly useful — 'convert this messy incoming JSON into this clean output format' — and it generates the transformation logic automatically.

Zapier is dominant among non-technical users because of breadth — over seven thousand app integrations — and simplicity. The AI additions are substantial: AI Actions let you insert a GPT-4o step anywhere in your workflow, Zapier Agents give you fully autonomous task completion for less-structured workflows, and Chatbots let you deploy a customer-facing AI connected to your data with almost no technical work.

I want to give you a decision heuristic that I've found useful:

If you're a technical user who cares about data privacy and total cost at scale — n8n self-hosted. If your team is non-technical and needs the fastest time to value — Zapier. If you're somewhere in between, or you need complex visual workflows — Make.

All three are worth knowing. They appear in job descriptions frequently, and being able to build an AI-powered workflow in any of them is a marketable skill right now."

---

## Slide 23 — Multimodal AI: Beyond Text
### 🎤 Professor Script

"Let's shift to multimodal AI, which is the recognition that business information doesn't come only as text.

A customer sends a photo of a damaged product along with their complaint. A financial analyst needs to read a chart embedded in a PDF. A doctor needs to reference a radiology image while writing notes. A quality control inspector takes a photo of a manufacturing defect. All of these involve visual information — and until recently, AI couldn't help with any of them.

Now it can.

Vision AI has crossed the threshold from impressive demo to production-ready capability. The major models — GPT-4o, Claude 3.7, Gemini 2.5 — can all reliably read and reason about images. And they can handle the specific images that come up in business: scanned documents with imperfect quality, complex PDFs with mixed text and graphics, product photos, screenshots.

The use cases in the table on this slide are all live in production at major companies today. Invoice processing via OCR used to require dedicated specialized software. Now you can send a photo of an invoice to a multimodal API and get back structured JSON with all the fields extracted — in seconds, with high accuracy, for fractions of a cent.

Audio has also transformed. OpenAI's Whisper model for transcription is essentially free and highly accurate — it handles accents, technical vocabulary, multiple speakers. Running a meeting and having it transcribed and summarized automatically is now a solved problem. And real-time voice AI — GPT-4o Voice, Gemini Live — means you can have a phone conversation with an AI, with natural latency. This is being deployed in customer service right now.

The common thread across all of these: the information that used to be locked in 'hard' formats — images, audio, video — is becoming as accessible to AI as plain text."

---

## Slide 24 — Video Generation: The New Frontier
### 🎤 Professor Script

"Video generation is the area where I want to be most careful to be accurate about both what's possible today and what's not yet ready.

The state of the art has improved dramatically. OpenAI's Sora, Runway Gen-3, Google Veo 2 — these tools can generate video clips that, for many purposes, look indistinguishable from footage you might shoot yourself. The quality for short clips — fifteen seconds to a minute — is now high enough for real business use.

But let me be specific about what 'business use' means here, because I don't want to oversell it.

What genuinely works in production right now: short marketing clips and social media content, especially for product visualization. A startup can now generate a dozen variations of a thirty-second product demo video in an hour, at essentially zero marginal cost. A/B testing creative has never been cheaper.

Training and onboarding content through tools like HeyGen and Synthesia is also production-ready. You create an AI avatar. You give it a script. It generates a professional-looking presenter video. More importantly: when the policy changes, you update the script and regenerate the video — no reshooting, no re-recording. This is transformative for compliance training that needs frequent updates.

What still has significant limitations: anything over two minutes is often inconsistent. Human faces and hands still occasionally distort in jarring ways. Generating video from a complex written description with precise scene requirements is still challenging.

And the ethical dimension is important. AI-generated video for marketing, training, entertainment — fine, as long as it's disclosed. Using AI video to mislead people about real events — not fine, and in many cases illegal. The disclosure norm is developing rapidly, and professional use should lead, not follow.

My recommendation: start experimenting with HeyGen or Synthesia for internal training content. That's the highest ROI, lowest risk application right now."

---

## Slide 25 — Real Business Applications: Industry Snapshots
### 🎤 Professor Script

"Let me close this section by making it concrete with industry-specific examples. I know you're all headed to different industries after this program, so I want to make sure you can see the specific applications in your field.

Finance and banking is one of the most active sectors for AI adoption right now. Earnings call analysis is a great example — companies like Morgan Stanley have deployed tools that let analysts query an earnings call transcript the moment it's released. Two hours of reading and note-taking becomes a two-minute query. The AI knows the transcript, the history, the analyst's context. The analyst provides the judgment.

Healthcare is where the stakes are highest and the opportunity is largest. The documentation burden on healthcare professionals is enormous — a typical primary care physician spends two hours on documentation for every hour with patients. AI that can draft clinical notes from ambient recordings of patient visits is being deployed at scale. That's two hours of a physician's time per day returned to patient care. The impact is measured in outcomes, not just cost.

Retail and e-commerce is where AI automation is perhaps most visible in its results. Writing product descriptions at scale — ten thousand SKUs, each with a unique, SEO-optimized description — used to require a team of writers and weeks of work. Now it's an afternoon of AI work and a human QA pass.

Consulting and professional services is directly relevant to your careers. Competitive intelligence, RFP responses, contract review — all of these are workflows where AI dramatically accelerates the junior-to-senior pipeline. Junior analysts can produce senior-quality first drafts. Senior professionals spend their time on strategic judgment, not research and formatting.

The pattern I want you to see in all of these: AI handles the gathering, processing, and formatting work. Humans provide domain expertise, judgment, and accountability. This is the partnership that produces results."

---
