# Week 08 — Professor Teaching Scripts
# Batch 4 of 5: Tools Ecosystem (Slides 19–25)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 19 — How the Tools Ecosystem Works
### 🎤 Professor Script

"Alright, we've spent a lot of time talking about the models themselves — who makes them, how they compare, what they cost. Now I want to zoom out and show you something more important for your day-to-day work: the ecosystem of tools that sits on top of those models.

Because here's the thing — most of you are never going to interact with a raw LLM directly. You're not going to call the OpenAI API and write JSON. You're going to use applications. And those applications are all built on top of these same underlying models.

Think about it like plumbing. The LLM is the water supply. The API is the pipes. And the apps you actually use — ChatGPT, Notion AI, GitHub Copilot, Salesforce Einstein — those are the faucets. You don't care about the plumbing. You care about the faucet working.

Now, there are three categories of tools in this ecosystem.

The first is native apps — tools built by the same company that makes the model. ChatGPT is OpenAI's native app. Claude.ai is Anthropic's. Gemini is Google's. These have the tightest integration and usually the best access to new features.

The second category is third-party tools — independent companies that built products on top of LLM APIs. Cursor is a coding IDE built on Claude and GPT-4o. Jasper is a marketing tool built on GPT-4o. Perplexity is a search engine built on multiple models. These companies add specialized functionality on top of the foundation.

The third category is enterprise integrations — large software vendors who have embedded AI into products you already use. Microsoft put Copilot into Word, Excel, Teams, and Outlook. Salesforce built Einstein. Zendesk built AI triage. You may already be paying for these capabilities right now.

Here's the insight I want you to hold onto: the model itself is increasingly becoming a commodity. The real business value — and where companies are winning and losing — is in the application layer built on top. The LLM is the engine. What matters is the car."

---

## Slide 20 — OpenAI's Native Ecosystem
### 🎤 Professor Script

"OpenAI started as a research lab. Today they are one of the most strategically important platform companies in the world. And the reason is that they didn't just build a model — they built an ecosystem.

Let me walk you through the pieces.

At the consumer level, you have ChatGPT — which is already familiar to most of you. But inside ChatGPT, there's DALL-E 3 for generating images from text. There's Sora for generating videos from text. There's a voice mode that lets you have a spoken conversation with the AI. And there's a canvas feature for collaborative document editing with AI. That's a lot for twenty dollars a month.

For developers, OpenAI has the Assistants API — which lets you build AI agents with persistent memory, file access, and tool use. They have an embeddings API for building search and recommendation systems. And the GPT Store, where anyone — without writing a single line of code — can create a custom version of ChatGPT trained on their own materials. There are now three million of these custom GPTs, ranging from customer service bots to specialized legal assistants to cooking advisors.

And at the enterprise level, they have Azure OpenAI Service — which is OpenAI's models running inside Microsoft's cloud infrastructure with enterprise security, compliance certifications, and service agreements. That's the version major corporations use when they need GPT-4o but can't put their data on OpenAI's consumer servers.

The strategic play here is clear: OpenAI is trying to be the operating system for AI. Not just a model provider. A platform. The GPT Store is their App Store equivalent. The API is their developer platform. And the enterprise layer is how they win the Fortune 500.

Whether they succeed is still an open question — the competition is intense. But they have a head start measured in years."

---

## Slide 21 — Anthropic & Google Ecosystems
### 🎤 Professor Script

"Let's look at two very different philosophies for building an AI ecosystem.

Anthropic's approach is what I'd call quality over quantity. They have fewer products than OpenAI, but each one is exceptionally well executed. The flagship is Claude.ai — their chat interface with a free tier and a Pro tier. What makes it different from ChatGPT in practice is the Projects feature. Claude remembers things across sessions within a project. You can give it context about your company, your role, your preferences, and it carries that context every time you open a new conversation. For ongoing business work — writing in a consistent style, working on a long project — that's genuinely useful.

They also have Artifacts, which is their equivalent of Canvas — you can generate and edit documents, code, and diagrams live inside the conversation. And their API is where things get interesting for developers: Claude Sonnet is currently the most popular model used in Cursor and is a major part of the GitHub Copilot backend for complex tasks. So even when you're using a third-party coding tool, there's a good chance you're actually running Claude underneath.

Google's approach is the opposite — integration everywhere. Their philosophy is that you shouldn't have to adopt a new tool. AI should be inside the tools you already use.

And they have the scale to execute on that. Gemini is embedded in Gmail, Google Docs, Google Sheets, Google Slides, and Google Meet. If your organization runs on Google Workspace — and a huge proportion of businesses do — Gemini is already available to you without any new software purchase.

But the product I genuinely think is underrated is NotebookLM. You upload documents — PDFs, YouTube videos, articles, Google Docs — and it becomes an expert on your specific content. Ask it questions. Get summaries and study guides. And then it does something I've never seen any other tool do: it generates an Audio Overview — two AI hosts having a ten-minute podcast conversation about your documents. You can hand that to a new employee and they'll understand a complex report in their commute. That is a genuinely new capability."

---

## Slide 22 — Productivity & Writing Tools
### 🎤 Professor Script

"There is a category of tools that I think will affect more business professionals more immediately than any other part of today's lecture. I'm talking about productivity and writing tools — the applications that put AI directly into the content creation workflow.

Every major writing platform is adding AI right now. And the market for these tools is exploding. Let me walk you through the major players.

Notion AI sits inside your notes and wikis. If your team uses Notion — which many startups and tech companies do — AI is already in there. Ask it to summarize a long meeting note. Ask it to turn bullet points into a formal document. Ask it to translate your internal jargon into customer-facing language.

Jasper is specifically designed for marketing copy. It knows how to write ad copy, blog posts, email campaigns. And critically, you can train it on your brand voice — upload examples of your existing content, and it learns to write the way your company writes. That consistency is hard to achieve manually at scale.

Copy.ai is similar but with a stronger focus on sales and marketing sequences. Email drip campaigns, cold outreach, LinkedIn messages.

Grammarly you likely already use for grammar and spell-check. Their AI layer now does much more — it rewrites sentences for tone, adjusts formality, suggests alternatives for clarity. It's a writing coach embedded in every text field you use.

Gamma is worth knowing for presentations specifically. You write a prompt — 'create a ten-slide deck on our Q2 marketing strategy' — and it generates the full deck with layout, imagery, and content. Not perfect, but a genuinely useful starting point.

And I should mention HeyGen, because you'll hear about it in this course. It generates AI avatar videos from a text script. You write what you want a presenter to say, it generates a video of that person saying it. We're using it to produce some of the video content for this course.

The key pattern across all of these tools: they take a foundational LLM, add domain-specific templates and brand training, and wrap it in an interface designed for a specific use case. You're not paying for the model. You're paying for the packaging."

---

## Slide 23 — Developer & Coding Tools
### 🎤 Professor Script

"This is the category that has probably created the most economic disruption of any AI application so far. Developer tools.

GitHub Copilot launched in 2021 and now has nearly two million paying subscribers. GitHub's own internal study found that developers using Copilot code fifty-five percent faster. That is a massive productivity gain. If you have a team of twenty engineers and they're all fifty-five percent more productive, that's effectively eleven engineers worth of work for free.

Let me walk you through the landscape.

GitHub Copilot is the original. It sits inside VS Code, JetBrains, and other IDEs. As you type, it suggests completions — not just the next word, but entire functions, entire classes. It reads the code around you and predicts what you're trying to build. Under the hood, it uses Claude and GPT-4o.

Cursor is what I'd call the next generation. It's a full IDE — you can switch to it from VS Code — and it goes much further than autocomplete. You can open a chat with the AI, show it your entire codebase, and ask it to understand a bug, explain a function, or make a change across multiple files. It's less like an autocomplete tool and more like a pair programmer who has read every line of your code.

Windsurf is similar to Cursor in the agentic space — it can make multi-file edits autonomously, not just suggest what you should type.

Replit AI is interesting for non-developers specifically because you don't install anything. You build and run code in the browser. The AI helps you write it. You can prototype an application in a browser tab.

Now, here's what I want business students to take away from this slide. These tools are not just for software engineers. A data analyst can use Cursor or Claude to write Python scripts to process data. A finance professional can use ChatGPT to generate Excel macros. A product manager can prototype an application without ever hiring a developer. The barrier between 'business person' and 'person who can build software' is lower than it has ever been in history."

---

## Slide 24 — Enterprise Business Tools
### 🎤 Professor Script

"I want to make sure you understand something important about AI adoption in the enterprise context. For most large organizations, the question is not 'should we adopt AI?' The question is 'are we actually using the AI we already paid for?'

Because here's the reality: if your company uses Microsoft 365, you're almost certainly already paying for Microsoft Copilot, or it's being offered to you. If you use Salesforce, Einstein AI is already in your CRM. If you use Zendesk for customer service, their AI features are already part of your subscription. The AI is there. The adoption isn't.

Let me go through the categories.

In CRM and sales, Salesforce Einstein Copilot can summarize sales calls, draft follow-up emails, and give probability scores to deals in your pipeline. HubSpot AI can write entire email campaigns and landing pages from a brief. Outreach personalizes cold outreach at scale — instead of writing the same cold email a hundred times, the AI generates a personalized version for each prospect based on their company and role.

In customer service, Intercom's Fin is the most interesting product right now. It's a fully autonomous AI agent that can resolve over fifty percent of support tickets without a human ever seeing them. Questions it can't answer, it escalates. The economics are remarkable — if your support team handles ten thousand tickets a month and Fin resolves five thousand of them, you've cut your support workload in half.

In productivity, Microsoft Copilot 365 is the most ambitious play. It's embedded in every Office application — Word, Excel, PowerPoint, Teams, Outlook. Summarize a long email thread. Get meeting notes from Teams. Ask Excel to analyze your data and build a chart. Ask PowerPoint to create a slide deck from a document.

The hidden opportunity here is that most companies pay thirty dollars per user per month for Microsoft Copilot 365 and use twenty percent of its features. Learning to maximize these tools you already have is immediate ROI with zero additional spend. That's the first place I'd start in any organization."

---

## Slide 25 — No-Code AI Automation
### 🎤 Professor Script

"The last category of tools I want to cover is one that I think creates the most opportunity for people in this room specifically. Because it requires no coding. No technical background. No developers. And yet it lets you build workflows that are genuinely sophisticated.

I'm talking about no-code AI automation platforms.

Let me give you the basic concept. In the past, if you wanted to connect multiple applications together and have them do something automatically, you needed to write code. API integrations, webhooks, data transformations — all developer work. Zapier changed that in 2012 by making it visual. But Zapier was automating simple, rule-based actions — if this happens, do that.

What no-code AI automation adds is intelligence to those workflows. It's not just 'if new email arrives, forward it.' It's 'if new email arrives, read it with an LLM, determine if it's a complaint or a request, draft an appropriate response, update the CRM record, and send a Slack message to the account manager.' That's a workflow that would have required a developer to build in 2022. Today a business analyst can build it in a tool like n8n or Make in an afternoon.

Let me highlight n8n specifically because I think it's exceptional. It's open-source — you can run it on your own server, which means your data never leaves your building. It integrates with four hundred apps. And its AI Agent node lets you give the LLM tools — the ability to search the internet, query a database, send an email — and then let it decide on its own which tools to use to complete a task. That's getting close to what we call an AI agent.

The other platforms — Zapier, Make, Flowise, Dify, Relevance AI, Voiceflow — each have their sweet spots. Zapier is the most beginner-friendly with six thousand app integrations. Flowise is open-source and specifically designed for building chatbots and retrieval-augmented generation systems. Voiceflow is for conversational AI across channels — web, mobile, WhatsApp, phone.

The critical thing to understand about all of these: most of them let you swap the underlying LLM without rebuilding the workflow. You start with GPT-4o for quality. You find out the cost is too high at volume, so you switch to Mistral. Your client has a privacy requirement, so you switch to Ollama running locally. The workflow stays the same. The intelligence changes. That flexibility is genuinely powerful for business."

---
