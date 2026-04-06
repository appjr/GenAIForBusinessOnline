# Week 11 — New AI Tools and Trends
# Batch 4 of 5: Business Automation & Multimodal AI (Slides 20–25)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 20: AI Business Automation — The Convergence

### When Workflow Automation Meets Intelligence

**Two technologies, each powerful alone — together, transformative:**

| Traditional Automation (Zapier/Make/n8n) | AI (LLMs, Vision, Voice) |
|------------------------------------------|--------------------------|
| Rule-based: if X then Y | Judgment-based: understand X, decide Y |
| Only handles structured data | Handles text, images, audio, documents |
| Brittle — breaks on edge cases | Adaptive — handles variation and ambiguity |
| "If email arrives → create ticket" | "Read email, understand intent, draft reply, route to right team" |
| Replaces mechanical steps | Replaces judgment steps |

**The resulting productivity impact — measured across 2025–2026 deployments:**

| Business Process | Manual Time | With AI Automation | Time Saved | Typical Cost/Run |
|-----------------|------------|-------------------|------------|-----------------|
| Invoice processing (extract, validate, route) | 5 min/invoice | 10 seconds | **97%** | <$0.01 |
| Customer support ticket triage + draft reply | 8 min/ticket | 45 seconds | **91%** | ~$0.05 |
| Contract risk review (50-page document) | 2–3 hours | 3 minutes | **97%** | ~$0.30 |
| Competitive intelligence report (weekly) | 6 hours | 20 minutes | **94%** | ~$1.00 |
| Meeting transcript → action items + summary | 30 min | 2 minutes | **93%** | ~$0.10 |

**The shift in what humans do:** Professionals stop spending 70% of their time on gathering, extracting, formatting, and routing — and spend it instead on judgment calls, client relationships, and strategy. The work becomes higher-leverage, not just faster.

---

## Slide 21: n8n — Open Source AI Automation

### When Privacy and Flexibility Matter

**n8n** (pronounced "nodemation") is the open-source alternative to Zapier — self-hostable, developer-extensible, and with native AI integration built for production workloads.

**Why organizations choose n8n over Zapier for AI workflows:**

| Criterion | n8n | Zapier |
|-----------|-----|--------|
| **Data hosting** | Self-hosted on your servers — data never leaves | All data flows through Zapier's US cloud |
| **AI model support** | Claude, GPT, Gemini, **Ollama (local)** — full choice | OpenAI + limited others |
| **Cost at scale** | Fixed server cost — unlimited operations | Per-operation pricing, expensive at scale |
| **Customization** | Full JavaScript/Python custom nodes | Limited to available integrations |
| **Open source** | Yes — auditable, forkable, extensible | No |
| **Setup complexity** | Medium (Docker, 30 minutes) | Low (browser, 5 minutes) |

**A production AI workflow in n8n — intelligent email routing:**

1. **Trigger:** New email arrives (Gmail / Outlook webhook)
2. **AI Agent Node (Claude):** Read full email → classify as [Sales / Support / Billing / Internal / Spam] → extract key information (sender intent, urgency, required action, deadline)
3. **Branch by classification:**
   - Sales → Create HubSpot lead → Notify sales channel in Slack with AI summary
   - Support → Create Zendesk ticket + AI-generated first draft reply for agent to edit
   - Billing → Route to billing team email with extracted invoice details
   - Spam → Archive silently
4. **Logging:** All actions logged to Google Sheets for weekly review

**Getting started:**
```bash
docker run -it --rm -p 5678:5678 -v ~/.n8n:/home/node/.n8n n8nio/n8n
# Open http://localhost:5678 — full visual editor, no code required
```

**The Ollama integration** is n8n's killer feature for regulated industries: run Llama 4 or Phi-4 locally inside n8n, processing sensitive documents with zero data leaving your network.

---

## Slide 22: Make and Zapier — No-Code AI Automation

### The Tools Your Non-Technical Colleagues Are Already Using

**Make (formerly Integromat) — Visual, Flexible, Powerful**

Make's scenario builder is a visual flowchart: modules are connected with wires, data flows are visible, and branching logic is explicit. It occupies the middle ground between Zapier's simplicity and n8n's technical depth.

**Make AI features added in 2025:**

| Feature | What It Does | Business Example |
|---------|-------------|-----------------|
| **AI Scenario Builder** | Describe what you want in natural language → Make builds first draft | "When a form is submitted, classify it and email the right department" |
| **OpenAI / Claude Module** | LLM step in any scenario | Summarize, classify, extract, generate — anywhere in the flow |
| **AI Data Transform** | "Convert this messy incoming JSON to this clean format" — no regex needed | Normalize data from 5 different vendors into one schema |
| **Make AI Chat** | Conversational debugging — "Why did this scenario fail?" | Reduce troubleshooting time from hours to minutes |

**Pricing:** Free (1,000 ops/mo) · Basic $9/mo · Pro $16/mo · Teams $29/mo

---

**Zapier — The Largest Integration Ecosystem**

7,000+ app integrations. The easiest setup. The most non-technical-friendly interface. And in 2024–2025, Zapier added AI across the entire platform:

| Zapier AI Feature | Description | Use Case |
|-------------------|-------------|---------|
| **AI Actions** | GPT-4o as a step in any Zap — generate, analyze, decide | Classify a lead's industry from their job title |
| **Zapier Agents** | Fully autonomous agents with no fixed trigger/action structure | Monitor a Slack channel and take action on relevant messages |
| **Canvas** | Visual drag-and-drop workspace for planning AI workflows | Design complex pipelines before building them |
| **Zapier Chatbots** | Deploy a customer-facing AI chatbot connected to your data | Answer product FAQs using your documentation |

**Tool selection guide:**

| Your situation | Best choice |
|---------------|------------|
| Sensitive data, need self-hosting, technical team | **n8n** |
| Non-technical team, fastest setup, broad integrations | **Zapier** |
| Visual workflows, medium complexity, moderate budget | **Make** |
| Developer team wanting full control + AI freedom | **n8n** with custom nodes |

---

## Slide 23: Multimodal AI — Vision and Audio in Business

### The Business Data That Wasn't Accessible — Now Is

**The shift:** In 2022, AI worked on text. By 2026, leading models natively process text, images, PDFs, audio, and video in a single API call. This unlocks enormous amounts of previously inaccessible business data.

**Vision AI — what's production-ready today:**

| Capability | Best Model | Business Application | Accuracy Level |
|-----------|-----------|---------------------|---------------|
| Invoice / receipt extraction | GPT-4o, Claude 3.7 | Extract vendor, amount, date, line items → structured JSON | 95–98% on clean scans |
| Contract / legal document review | Claude 3.7 (200K ctx) | Flag risky clauses, extract dates, obligations | Strong — needs human review for legal decisions |
| Dashboard / chart reading | Gemini 2.5, GPT-4o | "What does this chart tell us? What's the trend?" | Very good on standard chart types |
| Product photo QA | Gemini 2.5 Pro | Flag defects, verify packaging, check planogram compliance | Varies by defect complexity |
| Screen / UI automation | Claude Computer Use | Click through any application, fill forms, extract data from GUIs | Production-ready for structured UIs |

**Audio AI — what's production-ready today:**

| Capability | Best Tool | Business Application | Notes |
|-----------|----------|---------------------|-------|
| Meeting transcription | Whisper (OpenAI), Deepgram | Accurate transcripts + speaker diarization | Handles accents, technical terms well |
| Real-time voice agents | GPT-4o Voice, Gemini Live | Phone-based customer service, voice data entry | Latency < 500ms — natural conversation |
| Text-to-speech | ElevenLabs, OpenAI TTS | Narrate reports, e-learning, training content | Near-human quality at $0.15–$0.30/1K chars |
| Voice cloning | ElevenLabs | Consistent brand voice across all content | Requires explicit consent for use |
| Meeting → action items | Otter.ai, Fireflies + LLM | Auto-generate summary + task list from call recording | Standard practice at many companies |

---

## Slide 24: Video Generation — Where Business Use Is Ready

### What's Production-Ready vs. What Still Has Limitations

**The 2026 state of AI video — platform comparison:**

| Platform | Creator | Max Length | Strengths | Weaknesses | Best Business Use |
|----------|---------|-----------|-----------|------------|-------------------|
| **Sora** | OpenAI | ~1 min | Highest photorealism, strong scene consistency | Expensive ($200/mo), limited control | Premium marketing, brand films |
| **Runway Gen-3** | Runway | 10–30 sec | Film-quality, fine camera control, good motion | Short clips only | High-end social content |
| **Google Veo 2** | Google | Several min | Long clips, photorealistic, API access | Limited availability | Enterprise video at scale |
| **Kling 2.0** | Kuaishou | 2–3 min | Fast, good quality, generous free tier | Occasional consistency issues | Social media, product demos |
| **HeyGen** | HeyGen | Unlimited | AI talking-head avatars, multilingual | Presenters only, no scene generation | Training, onboarding, explainer videos |
| **Synthesia** | Synthesia | Unlimited | Enterprise-grade, 140+ languages, compliance | Template-based, less creative flexibility | Corporate training, L&D |

**What's genuinely production-ready for business in 2026:**

1. **Training and onboarding videos (HeyGen / Synthesia):** AI avatar presents content. Update the script → instantly regenerate. Translate to 30 languages at no added cost. Proven ROI: companies report 60–80% reduction in L&D video production time.

2. **Short-form marketing content:** 15–30 second product clips, social media visuals, ad creative variants. Generate 20 variants in an hour for A/B testing. Cost: $0.50–$2.00 per clip.

3. **Product demonstration videos:** Walk through software UI or product features. No reshooting when features change — regenerate from updated screenshots.

**Current limitations (be honest with stakeholders):**
- Clips longer than 2 minutes often have consistency issues
- Human faces and hands still occasionally distort — always review before publishing
- Complex narrative scenes with multiple characters remain difficult
- Always disclose AI-generated content — audience trust matters more than the cost savings

---

## Slide 25: Industry Applications — Where AI Is Creating Measurable Value

### Real Deployments, Real Numbers

**Finance and Banking**

| Application | AI Tools | Measured Impact |
|-------------|---------|----------------|
| Earnings call analysis and Q&A | Claude 3.7, GPT-4o | Analyst prep time: 2 hours → 8 minutes |
| Credit risk narrative generation | Fine-tuned LLM + RAG | Decision documentation: 40% faster |
| Regulatory gap analysis | Claude (200K context) | Compliance review: days → hours |
| Trade surveillance reports | LLM + structured ML | Alert investigation: 50% faster triage |

**Healthcare and Life Sciences**

| Application | AI Tools | Measured Impact |
|-------------|---------|----------------|
| Clinical note drafting (ambient AI) | Specialized models + voice | Physician documentation: −2 hrs/day |
| Prior authorization letters | Fine-tuned LLM | Drafting time: 45 min → 4 min |
| Clinical trial site matching | RAG + LLMs | Patient screening: 3× faster |
| Imaging report first draft | Multimodal AI | Radiologist throughput: +20–30% |

**Retail and E-commerce**

| Application | AI Tools | Measured Impact |
|-------------|---------|----------------|
| Product description generation | GPT-4.1, Claude | 10,000 SKUs/hour vs. 100/day manual |
| Returns reason classification | Fine-tuned LLM | Routing accuracy: 94% (vs. 71% rule-based) |
| Customer chat resolution | Fine-tuned LLM | 70% resolved without human agent |
| Trend-driven copywriting | Claude + RAG | Campaign brief to copy: hours → minutes |

**Consulting and Professional Services**

| Application | AI Tools | Measured Impact |
|-------------|---------|----------------|
| RFP section drafting | Claude + RAG on past proposals | First draft: 30 min (was 1–2 days) |
| Competitive intelligence digest | Perplexity + LLM pipeline | Daily brief: automated (was 4 hrs/week) |
| Contract risk flagging | Claude (long context) | First-pass review: 3 min (was 2 hrs) |
| Slide deck from outline | GPT-4o + PowerPoint API | Presentation first draft: 20 min |

> **The consistent pattern:** AI absorbs 60–80% of the information-gathering and drafting work. Professionals shift their time to judgment, relationships, and quality control — higher-leverage activities. The job isn't eliminated; it's upgraded.

---
