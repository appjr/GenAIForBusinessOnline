# Week 08 — The World of Large Language Models
# Batch 5 of 5: Use Cases + Wrap-up (Slides 25–30)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 25: Choosing the Right LLM — A Decision Framework

### Match the Tool to the Task

**Step 1: Identify the task category**

```
What do you need to do?
        │
        ├─── Write / Create content ────────────────► ChatGPT, Claude, Jasper
        │
        ├─── Research / Find current information ───► Perplexity, Gemini
        │
        ├─── Analyze long documents / PDFs ─────────► Claude (200K context)
        │
        ├─── Code / Build software ─────────────────► Cursor, Copilot, Claude
        │
        ├─── Work inside Google Workspace ──────────► Gemini + NotebookLM
        │
        ├─── Work inside Microsoft 365 ─────────────► Microsoft Copilot 365
        │
        ├─── Automate customer service ─────────────► Intercom Fin, Zendesk AI
        │
        ├─── Sensitive / private data ──────────────► Ollama (local), Claude Enterprise
        │
        └─── High-volume / cost-sensitive ──────────► Groq + Llama, Mistral API
```

**Step 2: Check budget reality**

| Budget | Recommendation |
|--------|---------------|
| $0/month | Gemini free, Claude free, Meta.ai, Ollama local |
| $20/month | ChatGPT Plus OR Claude Pro OR Google One AI Premium |
| $30–50/month | Add a specialist (Perplexity Pro + one coding tool) |
| Enterprise | Microsoft Copilot 365 + Claude/OpenAI Enterprise |

**Step 3: Consider your data sensitivity**
- **Public / non-sensitive data** → Any cloud LLM is fine
- **Confidential business data** → Claude Enterprise (zero retention) or Ollama (local)
- **Regulated industry** (healthcare, finance, legal) → Check compliance certifications (HIPAA, SOC2)

---

## Slide 26: Use Case — Content Creation & Marketing

### AI Is Now Every Marketer's Co-Writer

**The business impact:**
> Marketing teams using AI produce 3–5× more content at 60% lower cost (McKinsey, 2024)

**What AI can create today:**

| Content Type | Best Tool | Quality Level | Time Savings |
|-------------|----------|---------------|-------------|
| Blog posts (1,000–3,000 words) | ChatGPT + Claude | High with editing | 70% |
| Social media posts (all platforms) | Copy.ai, Jasper | High | 80% |
| Email campaigns & sequences | HubSpot AI, Jasper | High | 75% |
| Ad copy (Google, Meta, LinkedIn) | Jasper, Copy.ai | High | 80% |
| Product descriptions (e-commerce) | ChatGPT | High | 85% |
| Video scripts | Claude, ChatGPT | High | 60% |
| Press releases | Claude | High with editing | 65% |
| Presentation decks | Gamma, ChatGPT | Medium | 50% |

**Best practice workflow:**

```
Brief AI with brand voice + audience + goal
         ↓
Generate multiple versions (ask for 3 variants)
         ↓
Human editor refines and fact-checks
         ↓
Publish
```

**Real business example:** A 5-person marketing team at a mid-size B2B company used Claude + Jasper to increase blog output from 4 to 20 posts/month without adding headcount. Organic traffic grew 140% in 6 months.

---

## Slide 27: Use Case — Research & Analysis

### From Hours to Minutes

**The traditional research process:**
Search → Read → Synthesize → Write → Cite → Repeat (4–8 hours)

**The AI-augmented research process:**
Perplexity search → NotebookLM digest → Claude analysis → Human validation (30–60 minutes)

---

**Tool-by-tool breakdown for research:**

**Perplexity AI** — *Start here for current information*
- Research competitor landscape, industry trends, market size
- Always provides sources — easy to verify claims
- Pro tip: Use "Research" mode for multi-step deep dives

**Google NotebookLM** — *Your personal research analyst*
- Upload: PDFs, articles, YouTube videos, Google Docs (up to 50 sources)
- Ask it questions across all sources simultaneously
- Generate: summaries, FAQs, timelines, study guides, and audio podcasts
- Best for: Literature reviews, competitive intelligence, due diligence

**Claude** — *Best for long document analysis*
- Upload entire annual reports, legal contracts, research papers
- Ask: "What are the 5 biggest risks mentioned in this document?"
- 200K token context = entire books or large data files
- Best for: Contract review, financial analysis, policy documents

**ChatGPT with Advanced Data Analysis** — *For structured data*
- Upload CSVs, Excel files
- "Create a chart showing revenue by quarter"
- "Identify anomalies in this dataset"

---

## Slide 28: Use Case — Coding & Development

### Democratizing Software Development

**The paradigm shift:**
> In 2020, you needed a developer to build software.
> In 2025, a business analyst with AI tools can build production applications.

**Who benefits from AI coding tools:**

| Role | How AI Helps | Tool Recommendation |
|------|-------------|---------------------|
| **Software developers** | 55% faster; AI handles boilerplate | Cursor, GitHub Copilot |
| **Business analysts** | Build scripts, automate Excel/data tasks | ChatGPT, Claude |
| **Data scientists** | Generate and debug Python/R code | Cursor, Claude |
| **Product managers** | Prototype ideas without developers | Replit AI, Cursor |
| **Finance professionals** | Automate Excel macros, build models | Claude, ChatGPT |
| **Operations** | Build internal tools, automate workflows | Cursor, Replit |

**What AI coding tools can build (no developer needed):**
- Web scrapers to collect competitor pricing data
- Excel automation that processes 10,000 rows in seconds
- Internal dashboards connected to your database
- Email/Slack notification bots
- Data cleaning and transformation pipelines
- Simple web apps with forms and databases

**The new skill for business professionals:**
Being able to describe what you want to a coding AI (in plain English), evaluate the output, and iterate — is now a core business competency, not a technical skill.

---

## Slide 29: Use Case — Business Automation

### The Operational Transformation

**What business automation with AI looks like:**

```
Customer emails → AI reads and categorizes → Routes to right team
                                           → Drafts response suggestion
                                           → Updates CRM automatically

Sales call ends → AI transcribes → Extracts action items
                                → Updates deal in Salesforce
                                → Sends follow-up email draft

New employee → AI onboarding bot answers questions 24/7
             → Generates personalized training plan
             → Tracks completion and flags gaps
```

**High-ROI automation opportunities by department:**

| Department | Automation Opportunity | Tool | Estimated Time Saved |
|-----------|----------------------|------|---------------------|
| **Customer Service** | Tier-1 ticket resolution | Intercom Fin, Zendesk AI | 40–60% of tickets |
| **Sales** | Lead qualification + outreach personalization | HubSpot AI, Outreach | 5–10 hrs/rep/week |
| **HR** | Job description writing, resume screening | ChatGPT, Claude | 3–5 hrs/hire |
| **Finance** | Invoice data extraction, expense categorization | Claude, GPT-4o | 6–8 hrs/week |
| **Marketing** | Content repurposing across channels | Jasper, Copy.ai | 10–15 hrs/week |
| **Legal** | Contract first-pass review and redlining | Harvey AI, Claude | 50–70% draft time |
| **Operations** | SOP creation, process documentation | Notion AI, Claude | 4–6 hrs/process |

**The automation stack that works:**
1. **LLM** (Claude/GPT-4o) for intelligence
2. **Zapier or Make.com** for connecting apps
3. **Existing software** (Salesforce, Gmail, Slack) as the interface
4. **Human oversight** for exceptions and quality control

---

## Slide 30: Wrap-Up + Your 8 Exercises

### What We Covered Today

**The LLM landscape in 5 sentences:**

1. The Transformer architecture (2017) started everything — every LLM today builds on it
2. The "ChatGPT moment" (Nov 2022) brought AI to a billion people in months
3. You have two broad choices: **paid/proprietary** (easiest, most capable) and **free/open-source** (private, customizable, scalable)
4. The real value isn't the LLM itself — it's the **tools built on top** that fit into your existing workflows
5. Your job as a business professional is to **match the right tool to the right task** — not to be loyal to any one platform

---

**Your 8 hands-on exercises this week:**

| # | Tool | Task |
|---|------|------|
| 1 | ChatGPT | Analyze a business document with GPT-4o |
| 2 | Claude | Long-document risk analysis |
| 3 | Gemini + NotebookLM | Research paper podcast generation |
| 4 | Perplexity | Market research vs. Google Search |
| 5 | GitHub Copilot / Cursor | AI-assisted coding |
| 6 | Meta AI | Compare free vs. paid LLM output |
| 7 | Ollama | Run an LLM locally on your computer |
| 8 | Groq | Blazing-fast API inference |

**Key takeaway:**
> The question is never *"Should I use AI?"*
> The question is *"Which AI tool, for which task, at what cost?"*

**See you next class — bring your exercise results to discuss!**

---
