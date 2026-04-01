# Week 08 — The World of Large Language Models
# Batch 4 of 5: Tools Ecosystem (Slides 19–25)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 19: How the Tools Ecosystem Works

### From Model to Product

**The LLM value chain:**

```
┌─────────────────────────────────────────────────────────┐
│                  FOUNDATION LAYER                        │
│   LLM Core (GPT-4o / Claude / Gemini / Llama / Mistral) │
└───────────────────────┬─────────────────────────────────┘
                        │
                   APIs & SDKs
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
┌──────────────┐ ┌─────────────┐ ┌──────────────┐
│ NATIVE APPS  │ │  3RD PARTY  │ │  ENTERPRISE  │
│ (by company) │ │    TOOLS    │ │ INTEGRATIONS │
├──────────────┤ ├─────────────┤ ├──────────────┤
│ ChatGPT      │ │ Cursor      │ │ Salesforce   │
│ Claude.ai    │ │ Notion AI   │ │ HubSpot AI   │
│ Gemini       │ │ Jasper      │ │ MS Copilot   │
│ NotebookLM   │ │ Perplexity  │ │ Zendesk AI   │
└──────────────┘ └─────────────┘ └──────────────┘
```

**Three types of tools:**

| Type | Who builds it | Example | Relationship to LLM |
|------|--------------|---------|---------------------|
| **Native** | The LLM company itself | Claude.ai, ChatGPT | Direct product |
| **3rd Party** | Independent companies | Cursor, Jasper | Built via API |
| **Enterprise** | Large software vendors | Salesforce Einstein | Embedded feature |

**Key insight:** The LLM itself is a commodity becoming. The **application layer** built on top is where business value is captured.

---

## Slide 20: OpenAI's Native Ecosystem

### Building an AI Empire

**OpenAI's strategy:** Create the platform, not just the model

```
                    OpenAI
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
Consumer          Developer         Enterprise
Products           Tools             APIs
    │                 │                 │
ChatGPT          Assistants        Azure OpenAI
DALL-E 3           API               Service
Sora            GPTs Store         Batch API
ChatGPT Voice   Embeddings        Fine-tuning
```

**Key native products:**

| Product | What It Does | Who Uses It |
|---------|-------------|-------------|
| **ChatGPT** | Conversational AI, web browsing, image gen | Everyone |
| **DALL-E 3** | Text-to-image generation | Designers, marketers |
| **Sora** | Text-to-video generation | Content creators |
| **GPTs Store** | Custom AI assistants you build and share | Power users, businesses |
| **Assistants API** | Build AI agents with memory + tools | Developers |
| **Canvas** | AI-powered document editor | Writers, professionals |

**The GPT Store model:**
- Anyone can create a custom ChatGPT ("GPT") without coding
- Businesses publish GPTs for customer service, internal tools, etc.
- 3M+ custom GPTs created since launch

---

## Slide 21: Anthropic & Google Ecosystems

### Two Different Philosophies

---

### Anthropic / Claude Ecosystem

**Philosophy:** Fewer products, done with exceptional quality

| Product | What It Does |
|---------|-------------|
| **Claude.ai** | Main chat interface — free and Pro tiers |
| **Projects** | Persistent memory workspace — Claude remembers across sessions |
| **Claude API** | Developer access to all models |
| **Claude for Enterprise** | SSO, admin controls, zero data retention |
| **Artifacts** | Generate and edit documents, code, diagrams live in chat |

**Claude's emerging role in coding:** Claude Sonnet 4 is now the most popular model in Cursor and GitHub Copilot backend for complex tasks — a major third-party ecosystem win.

---

### Google's Ecosystem

**Philosophy:** Integrate AI everywhere Google already is

| Product | What It Does | Where It Lives |
|---------|-------------|---------------|
| **Gemini** | Main chat interface | gemini.google.com |
| **NotebookLM** | AI research assistant for your documents | notebooklm.google.com |
| **AI Studio** | Developer playground for Gemini | aistudio.google.com |
| **Vertex AI** | Enterprise AI platform on Google Cloud | cloud.google.com |
| **Gemini in Workspace** | AI in Gmail, Docs, Sheets, Slides, Meet | Your Google apps |
| **Google Search AI** | AI Overviews in search results | google.com |

**NotebookLM — The standout product:**
- Upload PDFs, YouTube videos, Google Docs, websites
- AI creates summaries, FAQs, study guides, and — remarkably — a podcast audio discussion of your documents
- Business use: onboard new employees, analyze industry reports, create training materials

---

## Slide 22: Productivity & Writing Tools

### The Content Creation Layer

**Market size:** $1.8B and growing — every major writing platform is adding AI

**Top tools and their LLM backbone:**

| Tool | Primary LLM | Category | Best For | Price |
|------|------------|----------|----------|-------|
| **Notion AI** | Claude + GPT-4o | Docs & notes | Meeting summaries, wikis, drafts | $10/user/mo add-on |
| **Jasper** | GPT-4o + Claude | Marketing copy | Ad copy, blog posts, brand voice | $49/mo |
| **Copy.ai** | GPT-4 | Sales & marketing | Email sequences, social posts | Free / $49/mo |
| **Grammarly AI** | Proprietary + GPT | Writing assistant | Grammar, tone, clarity, rewrites | Free / $30/mo |
| **Quillbot** | Proprietary | Paraphrasing | Academic rewriting, summarization | Free / $10/mo |
| **Gamma** | GPT-4o | Presentations | Auto-generate slides from a prompt | Free / $10/mo |
| **HeyGen** | Proprietary | Video | AI avatar video from text script | $29/mo |

**The common pattern:** These tools wrap a foundational LLM and add:
- **Brand voice** training so outputs match your company's style
- **Templates** tuned for specific use cases (LinkedIn post, press release, email)
- **Workflow integrations** with CRMs, CMSs, and project management tools
- A **simpler interface** than raw API access

**Business reality:** These tools cost $10–$50/month each but can save hours of work weekly. ROI is typically positive within weeks.

---

## Slide 23: Developer & Coding Tools

### AI Enters the IDE

**Market impact:** GitHub Copilot alone has 1.8M paying subscribers. Developers using AI code 55% faster (GitHub internal study, 2024).

**The major coding AI tools:**

| Tool | LLM Backbone | Platform | Key Feature | Price |
|------|-------------|----------|-------------|-------|
| **GitHub Copilot** | Claude + GPT-4o | VS Code, JetBrains, etc. | In-editor autocomplete | $10/mo individual |
| **Cursor** | Claude 4 Sonnet | Standalone IDE | Full codebase chat + edit | $20/mo |
| **Replit AI** | Custom | Browser-based IDE | Run + deploy in the browser | Free / $25/mo |
| **Amazon CodeWhisperer** | Amazon Titan | AWS ecosystem | Free for individual devs | Free |
| **Tabnine** | Enterprise-hosted | Any IDE | Privacy-first, self-hosted option | $12/mo |
| **Windsurf** | Multiple | Standalone IDE | Agentic coding — does multi-file edits | $15/mo |

**What these tools can do:**
- **Autocomplete** entire functions as you type
- **Explain** legacy code you've never seen
- **Refactor** messy code to best practices
- **Generate tests** automatically
- **Fix bugs** by describing the error in plain English
- **Build features** from a plain English description

**Key insight for business students:** You don't need to be a developer to use Cursor or Replit AI. These tools make coding accessible to analysts, product managers, and business professionals.

---

## Slide 24: Enterprise Business Tools

### AI Embedded in the Software You Already Use

**The enterprise AI wave:** Every major B2B software vendor has embedded AI since 2023. You may already be paying for it.

**CRM & Sales:**

| Tool | AI Feature | What It Does |
|------|-----------|-------------|
| **Salesforce Einstein** | Einstein Copilot | Summarize calls, draft emails, forecast deals |
| **HubSpot AI** | Content Assistant | Write emails, landing pages, social posts |
| **Outreach AI** | Smart sequences | Personalize cold outreach at scale |

**Customer Service:**

| Tool | AI Feature | What It Does |
|------|-----------|-------------|
| **Zendesk AI** | Intelligent triage | Auto-categorize tickets, suggest answers |
| **Intercom Fin** | AI Agent | Resolve 50%+ of support tickets without humans |
| **Freshdesk Freddy** | Answer Bot | Deflect routine questions 24/7 |

**Productivity & Collaboration:**

| Tool | AI Feature | What It Does |
|------|-----------|-------------|
| **Microsoft Copilot 365** | In Word/Excel/Teams/Outlook | Summarize meetings, draft documents, analyze data |
| **Slack AI** | Channel summaries | Catch up on long threads instantly |
| **Zoom AI Companion** | Meeting intelligence | Auto-summaries, action items, follow-up emails |
| **Notion AI** | Docs + databases | Write, edit, ask questions about your workspace |

**The hidden cost opportunity:**
> Most companies pay $30/user/month for Microsoft Copilot 365 but use 20% of its features.
> Learning to maximize these embedded AI tools = immediate ROI with zero additional spend.

---

## Slide 25: No-Code AI Automation

### Build AI Workflows Without Writing a Single Line of Code

**What no-code AI automation means:**
Connect apps, trigger actions, and embed LLM intelligence into multi-step workflows — using a visual drag-and-drop interface instead of code.

**Why it matters:** A marketing manager or operations analyst can now build workflows that would have required a developer in 2022.

---

### n8n — The Power User's Automation Platform

**Type:** Open-source, self-hostable (also cloud-hosted)
**LLM integrations:** GPT-4o, Claude, Gemini, Mistral, Ollama (local), any OpenAI-compatible API
**Website:** n8n.io | **Price:** Free (self-hosted) / $20/mo (cloud)

**What makes n8n unique:**
- Connect **400+ apps** with AI nodes in between
- **AI Agent node** — give the LLM tools (search, database, email) and let it decide what to do
- **Self-hosted option** — your data never leaves your server (critical for regulated industries)
- Uses **any LLM** — swap GPT-4o for Claude or a local Ollama model in one click

**Example workflow:**
```
New email arrives in Gmail
    → n8n extracts key info (GPT-4o)
    → Searches CRM for matching contact (HubSpot node)
    → Drafts a personalized reply (Claude Sonnet)
    → Saves summary to Notion
    → Sends Slack notification
```

---

### Other Leading No-Code AI Platforms

| Platform | LLM Underneath | Best For | Price | Key Differentiator |
|----------|---------------|----------|-------|--------------------|
| **Zapier AI** | GPT-4o (OpenAI) | Simple automations, beginners | Free / $29/mo | Easiest setup, 6,000+ app integrations |
| **Make (Integromat)** | GPT-4o, Claude | Complex multi-branch flows | Free / $9/mo | Visual flow designer, very flexible |
| **Flowise** | Any (OpenAI, Ollama, Anthropic, HuggingFace) | Building chatbots & RAG pipelines | Free (open-source) | Drag-and-drop LangChain flows |
| **Dify** | GPT-4o, Claude, Llama, Gemini, Mistral | AI app builder for teams | Free / $59/mo | Prompt management + RAG + API out of box |
| **Relevance AI** | GPT-4o, Claude | Sales & marketing AI agents | Free / $19/mo | Pre-built agent templates for business |
| **Voiceflow** | GPT-4o, Claude, Gemini | Conversational AI / chatbots | Free / $50/mo | Visual chatbot designer, multi-channel deploy |
| **Stack AI** | GPT-4o, Claude, Llama | Enterprise AI workflows | Free / $199/mo | Enterprise security, SOC2 compliant |

---

**The LLM flexibility advantage of no-code tools:**

Most platforms let you swap the underlying LLM without rebuilding the workflow:

```
Your workflow stays the same ──► Switch GPT-4o → Claude for lower cost
                              ──► Switch to Ollama for 100% private processing
                              ──► Switch to Groq for faster response times
```

**When to use no-code AI automation:**

✅ **Perfect for:**
- Automating repetitive knowledge-work tasks (email triage, report generation)
- Connecting AI to your existing software stack (CRM, Slack, databases)
- Prototyping AI-powered workflows before investing in custom development
- Teams without dedicated engineers

⚠️ **Limitations:**
- Complex logic and custom UI still require code
- Vendor lock-in risk with hosted platforms
- Debugging multi-step flows can be tricky
- Cost scales with usage volume

---
