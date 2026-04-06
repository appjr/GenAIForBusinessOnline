# Week 11 — New AI Tools and Trends
# Batch 2 of 5: AI Agent Frameworks & MCP (Slides 7–13)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 7: The Rise of Agentic AI

### From "Answer Me" to "Do This For Me"

**The fundamental shift:**

| Dimension | Chatbot (2022–2024) | AI Agent (2025–2026) |
|-----------|-------------------|---------------------|
| **What you give it** | A question | A goal or task |
| **What it returns** | Text | A completed result |
| **How it works** | One round of input/output | Plans, uses tools, iterates, adapts |
| **Memory** | Single session only | Persistent across steps and sessions |
| **Tools available** | None (or limited search) | Web, code execution, files, APIs, databases, email |
| **Supervision needed** | High — human drives every step | Low to Medium — agent drives, human reviews |
| **Business analogy** | Search engine | Junior analyst |

**The agent loop — how every agent framework works:**
1. **Receive** the task and break it into steps
2. **Select** the right tool for each step
3. **Execute** the tool call and observe the result
4. **Reason** about whether the result moves toward the goal
5. **Repeat** until the task is complete — then return the final answer

**Why this matters for business:** Tasks that used to require a junior analyst working 3–4 hours — competitive research, data synthesis, report drafting, invoice processing — can now be completed in minutes by an agent running autonomously. The bottleneck shifts from execution to oversight and validation.

---

## Slide 8: Anthropic Claude Agent SDK

### Building Production Agents with Claude

**What makes the Claude Agent SDK powerful:**
- **Tool use:** Define any Python function as a tool — the model decides when and how to call it
- **Extended thinking:** Claude reasons step-by-step before using tools, reducing errors on complex tasks
- **Computer use:** Claude can control a real browser — click, type, navigate, take screenshots
- **Multi-agent:** Spawn sub-agents for parallel tasks; orchestrate teams of specialized Claudes
- **Long context:** 200K tokens of working memory — entire codebases, long documents, conversation history

**Core pattern — tool definition and agent loop:**
```python
import anthropic
client = anthropic.Anthropic()

tools = [{
    "name": "search_competitors",
    "description": "Search competitor websites for pricing and feature information",
    "input_schema": {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "focus": {"type": "string", "enum": ["pricing", "features", "news"]}
        },
        "required": ["company", "focus"]
    }
}]

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=4096,
    tools=tools,
    messages=[{"role": "user", "content":
        "Research pricing for Salesforce, HubSpot, and Pipedrive. "
        "Produce a comparison table and a recommendation for a 50-person sales team."}]
)
```

**What Claude returns:** Either a final text answer, or a `tool_use` block containing the tool name and input — which your code executes, returning results back into the conversation until the task is done.

**Best business use cases:** Multi-step research and reporting, automated code review, document analysis pipelines, CRM data enrichment, financial modeling workflows

---

## Slide 9: OpenAI Agents SDK

### Four Primitives for Production Agentic Systems

**Released in early 2025 — designed for enterprise deployments**

The OpenAI Agents SDK is built around four composable primitives that map cleanly to real business workflow patterns:

| Primitive | What It Is | Business Example |
|-----------|-----------|-----------------|
| **Agent** | An LLM with a role, instructions, and tools | "Tier-1 Support Agent" with access to your knowledge base |
| **Handoff** | Transfers control to a specialist agent | Escalate complex issues to "Billing Specialist Agent" |
| **Guardrail** | Validates inputs and outputs against rules | Block PII from leaving the system; enforce safe responses |
| **Tracing** | Full audit log of every agent decision and tool call | Required for compliance in finance and healthcare |

**Code pattern — multi-agent handoff:**
```python
from agents import Agent, Runner

triage_agent = Agent(
    name="Support Triage",
    instructions="Classify support tickets and route to the right specialist.",
    handoffs=[billing_agent, technical_agent, escalation_agent]
)

billing_agent = Agent(
    name="Billing Specialist",
    instructions="Resolve billing questions using the customer database tool.",
    tools=[query_billing_db, issue_refund, update_subscription]
)

result = Runner.run(triage_agent, "My invoice shows double charge for March")
```

**Key differentiators vs. other frameworks:**
- Native integration with OpenAI Assistants API (persistent threads, file attachments)
- Built-in tracing dashboard — see every step your agent took
- Guardrails run in parallel, not sequentially — no added latency
- Seamless handoffs with context preservation — the receiving agent knows the full history

---

## Slide 10: Google ADK and LangGraph

### Enterprise Orchestration and Complex Control Flow

**Google Agent Development Kit (ADK)**

Designed for organizations already on Google Cloud — tightly integrated with Vertex AI, Google Search, Google Workspace, and enterprise compliance infrastructure.

| Feature | Detail |
|---------|--------|
| **Model** | Gemini 2.0 Flash / Pro — native, no adapter needed |
| **Tools** | Google Search, Maps, Drive, Gmail, Calendar — all native |
| **Multi-agent** | Hierarchical: agents delegate to sub-agents automatically |
| **Sessions** | Persistent state management across user conversations |
| **Deployment** | Vertex AI — enterprise billing, SOC2, HIPAA, regional data |
| **Evaluation** | Built-in eval framework for measuring agent quality |

```python
from google.adk.agents import Agent
from google.adk.tools import google_search, google_drive_read

market_analyst = Agent(
    name="market_analyst",
    model="gemini-2.0-flash",
    tools=[google_search, google_drive_read],
    system_instruction="""You are a senior market analyst.
    Research using current web data and internal Drive documents.
    Always cite sources and flag uncertainty."""
)
```

---

**LangGraph — Workflows as Graphs**

LangGraph models agent behavior as a **directed graph**: nodes are processing steps, edges are transitions. This enables patterns that linear chains cannot handle:

- **Cycles:** Retry a step if quality is insufficient; iterate until criteria are met
- **Parallel branches:** Run research and data analysis simultaneously; merge results
- **Human-in-the-loop:** Pause execution at a "human review" node; resume when approved
- **Conditional routing:** Branch to different paths based on content classification

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)
workflow.add_node("research", research_step)
workflow.add_node("write", write_step)
workflow.add_node("human_review", pause_for_approval)
workflow.add_node("publish", publish_step)

# Conditional: only route through human review if flagged
workflow.add_conditional_edges("write",
    lambda s: "human_review" if s["needs_review"] else "publish")
workflow.add_edge("human_review", "publish")
```

**When to use LangGraph:** Any workflow with retry logic, parallel execution, human approval gates, or complex branching — compliance workflows, content production pipelines, multi-stage data processing

---

## Slide 11: CrewAI — Role-Based AI Teams

### The Consulting Firm Model for AI

**The core insight behind CrewAI:** Complex knowledge work is done better by a team of specialists than by one generalist — even when all the "people" are AI.

A researcher produces different quality work than a writer. An analyst reasons differently than a strategist. CrewAI makes this explicit: you define a team with roles, goals, and backstories — and the agents actually perform differently as a result.

**Full example — competitive intelligence workflow:**
```python
from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="Senior Market Research Analyst",
    goal="Find accurate, data-backed competitive intelligence",
    backstory="""You are a veteran analyst at McKinsey & Company.
    You cite specific numbers and sources. You never speculate.""",
    tools=[web_search, company_database, news_api],
    verbose=True
)

analyst = Agent(
    role="Strategic Analyst",
    goal="Draw actionable insights from raw research data",
    backstory="You turn data into strategy. You think in frameworks.",
)

writer = Agent(
    role="Executive Communications Specialist",
    goal="Produce crisp, C-suite-ready reports in under 500 words",
    backstory="Your reports get read by CEOs. Clarity above all.",
)

# Tasks flow sequentially — each builds on the previous
research_task = Task(
    description="Research Salesforce's Q1 2026 product announcements and pricing changes",
    agent=researcher,
    expected_output="Structured data: announcements, pricing, quotes, dates"
)
analysis_task = Task(
    description="Identify strategic implications for a mid-market CRM competitor",
    agent=analyst, context=[research_task]
)
writing_task = Task(
    description="Write a one-page executive brief on competitive risk and response options",
    agent=writer, context=[research_task, analysis_task]
)

crew = Crew(agents=[researcher, analyst, writer],
            tasks=[research_task, analysis_task, writing_task],
            process=Process.sequential)
result = crew.kickoff()
```

**Real production deployments today:** Automated RFP responses (one agent per requirement section), daily competitive intelligence briefings, M&A due diligence workflows, content production pipelines

---

## Slide 12: MCP — Model Context Protocol

### The Universal Standard for AI Tool Integration

**The problem MCP solves:**

Before MCP (pre-late 2024), every AI application needed custom integrations for every data source. A team using Claude needed one custom integration for their database, another for Slack, another for GitHub, another for their CRM. Multiply this by dozens of tools and dozens of AI products — every vendor reinventing the same connections in incompatible ways.

**MCP's solution:** One open standard, like USB-C for AI. Build an MCP server once; any MCP-compatible client can use it.

**How MCP works:**

| Component | Role | Example |
|-----------|------|---------|
| **MCP Server** | Exposes data/tools via the standard protocol | Your PostgreSQL database as an MCP server |
| **MCP Client** | AI application that connects to servers | Claude, Cursor, any LLM app |
| **MCP Protocol** | The standard that connects them | JSON-RPC over stdio or HTTP |

**MCP servers available today (April 2026):**

| Server | What AI Can Do | Use Case |
|--------|---------------|---------|
| **PostgreSQL / SQLite** | Query, read, analyze your database | AI-powered data analysis without ETL |
| **GitHub** | Read issues, create PRs, push commits | AI-assisted development workflows |
| **Slack** | Read channels, send messages | AI team communication and monitoring |
| **Google Drive / Docs** | Read and write documents | AI document workflows |
| **File System** | Read and write local files | Local AI automation |
| **Brave Search** | Real-time web search | Grounded, up-to-date AI responses |
| **Jira / Linear** | Read/create tickets | AI-driven project management |

**Why MCP matters for your career:** If you build an MCP server for your company's internal tools, that integration works with every MCP-compatible AI product — now and in the future. One investment, compounding returns as the ecosystem grows.

---

## Slide 13: Building Your First Agent — The Pattern

### The Code Architecture Every Agent Uses

**Every agent framework — regardless of provider — implements this same fundamental loop:**

```python
import anthropic, os
client = anthropic.Anthropic()

# 1. DEFINE TOOLS: Tell the model what capabilities it has
tools = [
    {
        "name": "fetch_webpage",
        "description": "Fetch visible text from a URL. Use for research.",
        "input_schema": {"type": "object",
                         "properties": {"url": {"type": "string"}},
                         "required": ["url"]}
    },
    {
        "name": "save_report",
        "description": "Save the finished report to a file.",
        "input_schema": {"type": "object",
                         "properties": {"filename": {"type": "string"},
                                        "content": {"type": "string"}},
                         "required": ["filename", "content"]}
    }
]

# 2. RUN THE AGENT LOOP
messages = [{"role": "user", "content":
    "Research the top 3 cloud CRM providers, compare pricing, save as crm_report.md"}]

while True:
    response = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=4096,
        tools=tools, messages=messages
    )

    if response.stop_reason == "end_turn":          # 3. DONE
        print(response.content[0].text)
        break

    for block in response.content:                  # 4. EXECUTE TOOLS
        if block.type == "tool_use":
            result = execute_tool(block.name, block.input)
            messages += [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                ]}
            ]
```

**Three things to understand about this pattern:**
- The model **reads** the tool descriptions and decides when to call them — you never tell it which tools to use
- The loop continues until `stop_reason == "end_turn"` — the model signals when it's done
- Any Python function you can write becomes a tool — databases, APIs, email, file system, anything

**This is the foundation.** Every framework (LangGraph, CrewAI, OpenAI Agents) is an abstraction on top of exactly this loop.

---
