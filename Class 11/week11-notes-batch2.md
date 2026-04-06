# Week 11 — Professor Teaching Scripts
# Batch 2 of 5: AI Agent Frameworks & MCP (Slides 7–13)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 7 — The Rise of Agentic AI
### 🎤 Professor Script

"We're now entering the section I consider the most important topic in AI right now.

Agents.

Let me make sure we all have the same mental model. When you use ChatGPT, you ask a question and you get an answer. That's one round. That's a chatbot. It's useful, but it's reactive. You do the work of deciding what to ask, in what order, and how to take the output and actually do something with it.

An AI agent is different. You give it a *task* — a goal — and it figures out how to accomplish that goal, step by step, using tools you've given it access to. It plans. It executes. It observes what happened. It adjusts. It keeps going until the task is done.

Here's a concrete example that will make this real. Let's say you tell an agent: 'Research our top three competitors, extract their pricing from their websites, build a comparison table, and email it to me.' A chatbot can't do this — it would give you instructions for how to do it yourself. An agent actually does it. It browses the competitor websites. It extracts the pricing information. It builds the table. It sends the email. You come back twenty minutes later to a finished deliverable.

This is why agents are transformative. We're not talking about faster question-answering. We're talking about automating multi-hour knowledge work tasks.

The key technical components that make agents possible: tools — functions the AI can call; memory — state that persists across steps; and planning — the ability to decompose a goal into steps. All of the major agent frameworks we're covering tonight provide these three things, in different ways.

Let's look at them."

---

## Slide 8 — Anthropic Claude Agent SDK
### 🎤 Professor Script

"Let's start with the Claude Agent SDK — partly because it's the most relevant to the models we've been working with all semester, and partly because it's one of the most capable.

The Claude SDK is really the Anthropic Python SDK with tools enabled. The core mechanism is simple: you define functions — tools — that describe what the agent can do. The model decides when to call those tools, calls them, gets results, and continues reasoning until it reaches a final answer.

Look at the code example. We define a tool called 'search_web' — it has a name, a description that the model reads to understand when to use it, and an input schema that tells the model what parameters to pass.

Then we send a request to Claude with the tools list included. If Claude decides it needs to search the web to answer the question, it returns a tool use block instead of a text answer. Our code executes the function, gets the result, and feeds it back to Claude. This continues until Claude has everything it needs to give a final answer.

That's the agent loop in its simplest form. The code we'll look at in slide thirteen shows the full loop.

Now, what's special about Claude for agents in 2026? Two things.

First, computer use. This is a capability you can enable through the API where Claude can actually control a computer — move a mouse, type text, click buttons, take screenshots to see what happened. This means any workflow that involves a GUI application — not just APIs — can be automated. Software that doesn't have an API becomes accessible.

Second, extended thinking. For complex reasoning tasks, Claude 3.7 Sonnet can spend additional computation thinking through a problem before responding. You can see the thinking process. For multi-step business analysis tasks, this produces substantially better results."

---

## Slide 9 — OpenAI Agents SDK
### 🎤 Professor Script

"OpenAI released their Agents SDK in early 2025 — and it's production-focused in a way that makes it feel different from earlier agent frameworks.

The SDK has four primitives. Let me walk through each.

Agents — these are your LLMs with instructions and tools. You define what the agent knows, what it's supposed to do, and what tools it can use. Straightforward.

Handoffs — this is where it gets interesting. In a real enterprise workflow, you don't have one AI doing everything. You have specialists. A tier-one support agent handles common issues. For edge cases, it hands off to a specialist agent. For escalations, it hands off to a human review queue. Handoffs make this pattern explicit and easy to implement.

Guardrails — these run validation on inputs and outputs. If a user tries to inject a prompt attack, the guardrail catches it. If the model is about to output something that violates your policy, the guardrail catches it. For production deployments, this is essential.

And tracing — you can see everything the agent did. What tools it called, what inputs it received, what outputs it produced, where it handed off. For debugging and for compliance, this matters a lot.

The code example on the slide shows how these primitives compose naturally. You define a research agent and a writing agent. The research agent has web search tools and a handoff description that tells it when to transfer to the writer. The pipeline runs and returns a finished report.

The strength of OpenAI's SDK is its tight integration with the Assistants API — which gives you persistent threads, file attachments, and built-in vector search for retrieval. If you're already deep in the OpenAI ecosystem, this SDK is a natural extension."

---

## Slide 10 — Google ADK & LangGraph
### 🎤 Professor Script

"Two more frameworks to cover — Google's ADK and LangGraph — and then we'll connect all of these with a code demo.

Google's Agent Development Kit is the enterprise play. If your organization is on Google Cloud, using Vertex AI, needing enterprise billing, compliance, and security guarantees, ADK is the path. It integrates natively with Gemini 2.0 and all of Google's tool ecosystem — Search, Maps, Drive, Gmail, and so on. The multi-agent architecture lets you build hierarchies of specialized agents that delegate to each other.

The code example is compact — you define an agent with a model, tools, and a system instruction. But behind that simplicity is the full Google Cloud enterprise infrastructure. Logging, monitoring, IAM permissions, regional data residency. For a company in a regulated industry — healthcare, finance, government — that infrastructure matters enormously.

LangGraph takes a completely different approach, and I find the mental model really elegant.

The key insight is this: complex agent workflows aren't linear. They have loops. A document review agent might extract information, identify that it needs to do a follow-up search, loop back to retrieve more information, and then generate a final answer. They have branches. An approval workflow takes a different path depending on the output of each step. They have human-in-the-loop steps where execution pauses for human review.

A linear chain can't model any of this cleanly. But a graph can. Nodes are computation steps. Edges are transitions. You can have conditional edges — 'if the output needs review, go to the review node; otherwise go to the end.' You can have cycles — 'retry this step up to three times if it fails.'

LangGraph is powerful but more complex to reason about. My recommendation: use the OpenAI or Anthropic SDK for simpler agent tasks. Reach for LangGraph when your workflow has genuinely complex control flow — parallel branches, retry logic, human approval steps."

---

## Slide 11 — CrewAI & Multi-Agent Teams
### 🎤 Professor Script

"CrewAI is my favorite framework for a specific category of business task, and the reason is in the name: crew.

The mental model that CrewAI gives you is a team. You have a researcher. You have an analyst. You have a writer. You have a reviewer. Each agent has a role, a goal, a backstory that shapes how it interprets tasks, and access to specific tools.

Tasks flow through the crew sequentially — or in parallel if you configure it that way. The output of one task becomes context for the next. The researcher finds the raw information. The analyst draws insights from it. The writer turns insights into prose. The reviewer checks for accuracy and quality.

Why does this produce better results than a single agent? Because you're mimicking how domain experts actually work collaboratively. A researcher brings different strengths than a writer. Forcing one AI to do both in a single prompt produces mediocre research and mediocre writing. Specialization improves quality.

The code example on the slide is a competitive intelligence workflow — something a real consulting firm might automate. The researcher agent fetches and synthesizes market data. The writer turns it into an executive summary. The backstory prompt — 'You write for C-suite audiences who need clarity and brevity' — shapes the tone in ways a generic writing prompt doesn't.

Real business applications that are working right now with CrewAI: automated RFP responses where one agent answers each requirement section; competitive intelligence reports that update daily; financial due diligence workflows; content production pipelines for marketing teams.

The business value proposition is clear: a workflow that used to take a team of junior analysts three days runs in thirty minutes."

---

## Slide 12 — MCP: Model Context Protocol
### 🎤 Professor Script

"MCP is the piece of the puzzle that ties everything together, and I think it's one of the most underappreciated developments in AI tooling right now.

Let me explain the problem it solves, because the problem was genuinely painful.

Before MCP, if you wanted Claude to access your company database, you had to write a custom integration. If you also wanted it to read files from Google Drive, another custom integration. GitHub? Another one. Slack? Another. And when you switched from Claude to GPT-4o, you had to rewrite all those integrations because the APIs were completely different.

Multiply this across every enterprise with hundreds of internal tools and you have an integration nightmare. Every AI product reinventing the same connections, with incompatible implementations.

Anthropic released MCP in late 2024 as an open standard to solve this. Think of it like USB-C. You have one standard plug. Everything that needs to connect to a computer gets a USB-C port. One cable works everywhere.

MCP works similarly. You build an MCP server for your database — once. Now any MCP-compatible AI application — Claude, Cursor, Windsurf, any tool that adopts the standard — can use it. Switch from one AI provider to another? The integrations still work. Build a new AI product? All your existing MCP servers are immediately available to it.

The ecosystem is growing rapidly. There are already MCP servers for PostgreSQL, SQLite, GitHub, Slack, Google Drive, Google Docs, Jira, Linear, file systems, browsers, and hundreds of other tools. Most of them are open source.

What does this mean practically? If you're building enterprise AI today, invest in MCP servers for your internal tools. That investment compounds — it works with every AI tool you adopt in the future. This is the infrastructure layer that makes AI integration sustainable."

---

## Slide 13 — Building Your First Agent: Live Demo
### 🎤 Professor Script

"Okay, enough theory. Let's look at actual code.

This slide shows a business intelligence agent — something you could actually deploy — in about fifty lines of Python. I want to walk through the structure carefully because this pattern repeats across every agent framework you'll ever work with.

Three parts.

First, tool definitions. We define what the agent can do by describing functions. These aren't function implementations — they're descriptions. The name, a plain-language description that the model reads to understand *when* to call this tool, and a JSON schema for the inputs. Claude reads these descriptions and reasons about which tool to use, in what order, with what parameters. The richer your descriptions, the better the tool use.

Second, a tool executor. When the model decides to call a tool, we need to actually run it. This is the bridge between the AI's decision and the real world. In production, these are real functions — fetching from databases, calling APIs, writing files, sending emails. In the demo, they're simulations. The structure is the same.

Third, the agent loop. This is the heart of it. We start with the user's task as a message. We send it to Claude. Claude either gives us a final answer — stop reason 'end_turn' — or it calls a tool. If it calls a tool, we execute it, get the result, and add both the tool call and the result to the message history. Then we send the updated conversation back to Claude. Repeat until Claude is done.

That's it. Every agent framework — OpenAI Agents, LangGraph, CrewAI — is abstracting over this same fundamental loop. Understanding the loop means you're never confused about what's happening underneath.

For your homework this week, you'll be building variations of exactly this pattern. Don't worry about it being perfect — the goal is to run the loop, see a tool get called, and get a result back. Once that clicks, agents will feel intuitive."

---
