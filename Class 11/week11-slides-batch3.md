# Week 11 — New AI Tools and Trends
# Batch 3 of 5: AI Coding Tools (Slides 14–19)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 14: The AI-Powered Development Revolution

### What Changed in 24 Months

**The productivity data (GitHub, StackOverflow surveys, 2025–2026):**

| Development Task | Without AI | With AI Tools | Speedup |
|-----------------|-----------|---------------|---------|
| Write a new function | 15 min | 2 min | **7×** |
| Fix a bug | 30 min | 8 min | **4×** |
| Write boilerplate / scaffolding | 45 min | 30 sec | **90×** |
| Understand an unfamiliar codebase | 2 weeks | 3 days | **5×** |
| Write unit tests | 20 min | 3 min | **7×** |
| Code review coverage | 60% of PRs | 95% (AI reviews all) | **+35%** |

**The before/after for business analytics professionals:**

**Before (2022):** Write every line manually. Google for syntax. Stack Overflow for debugging. 70% of coding time on boilerplate, lookups, and mechanical fixes.

**After (2026):** Describe what you want. Review and validate what AI produces. Focus on logic, architecture, and business problem framing. AI handles the execution details.

**What this changes about skills:**
- **Declining value:** Memorizing syntax, writing boilerplate, translating pseudocode to code
- **Rising value:** Problem decomposition, system design, prompt engineering, output validation, domain expertise
- **New skill:** Knowing when to trust AI output vs. when to verify carefully (security, edge cases, data integrity)

> *"The question is no longer whether you should use AI coding tools. It's which ones, and how to use them without losing your own judgment."*

---

## Slide 15: Claude Code — The Agentic Terminal

### Anthropic's CLI Coding Agent

**What makes Claude Code fundamentally different from Copilot-style tools:**

| Dimension | Claude Code | IDE Autocomplete Tools |
|-----------|------------|----------------------|
| **Interface** | Terminal CLI — works anywhere | IDE plugin — tied to one editor |
| **Scope of awareness** | Your entire codebase | The file you're editing |
| **Mode of operation** | Completes full tasks end-to-end | Suggests what to type next |
| **Actions it can take** | Read files, write files, run tests, execute commands | Insert code suggestions only |
| **Autonomy level** | Full agent — plans and executes | Copilot — assists and waits |
| **Best model** | Claude 3.7 Sonnet (coding benchmark leader) | Varies by tool |

**How it works in practice:**
```bash
npm install -g @anthropic-ai/claude-code   # install once
cd my-project                              # go to your project
claude                                     # launch the agent
```

**Real task examples Claude Code completes autonomously:**
- "Add error handling and retry logic to all API calls in the `src/api/` directory"
- "Write pytest unit tests for the `UserService` class, covering edge cases"
- "Refactor the authentication module to use JWT tokens instead of session cookies"
- "There's a bug in `checkout.py` — the discount isn't applied when cart > $500. Fix it."
- "Explain what this 800-line legacy function does and add inline documentation"

**The agent loop:** Reads relevant files → understands context → makes targeted changes → runs your tests → iterates if tests fail → reports when done

**Pricing:** You pay for Claude API tokens. A typical task costs **$0.05–$0.50**. Complex multi-file refactors run $1–$3. No subscription fee.

---

## Slide 16: Cursor — The AI-Native IDE

### VS Code Rebuilt for AI-Assisted Development

**Why Cursor is the most popular AI coding tool among professional developers (2025–2026):**

Cursor didn't bolt AI onto an existing editor — they forked VS Code and rebuilt every layer with AI as a core primitive. The AI has access to your full project structure, open files, terminal output, and git history. This context depth is what separates Cursor from plugins.

**The four features that matter:**

| Feature | How It Works | What It Unlocks |
|---------|-------------|----------------|
| **Tab Completion** | Completes entire functions, not just lines. Understands the surrounding code's intent. | 3–5× faster for function and class writing |
| **Chat (Ctrl+L)** | Conversational AI with @-mention context. `@UserService` pulls that file into context. | Navigate and understand large codebases instantly |
| **Composer / Agent Mode** | Give a task, not a line. Plans, writes, runs terminal commands, iterates. | Autonomous multi-file implementation |
| **Cursor Rules** | Project-level instructions: style guides, conventions, framework choices, test requirements. | Consistent AI behavior across your whole team |

**Pricing tiers:**

| Plan | Completions | Models | Best For |
|------|------------|--------|---------|
| Free | 2,000/month, limited chat | Cursor-mini | Trying it out |
| Pro ($20/mo) | Unlimited | GPT-4o, Claude 3.7, Gemini 2.5 | Daily professional use |
| Business ($40/user/mo) | Unlimited | All models + priority | Teams with shared rules and admin controls |

**Student recommendation:** Pro at $20/month. If you write Python for data work, this pays for itself in the first hour of use.

---

## Slide 17: Windsurf — The Cascade Agent

### Long-Horizon Autonomous Coding

**Windsurf**, built by Codeium, launched in late 2024 with a different philosophy than Cursor: instead of optimizing for how fast you can type, optimize for how complex a task AI can complete without you touching the keyboard.

**The Cascade Agent — what makes it distinct:**
- Maintains a **coherent understanding of the whole goal** across many steps — not just the current task
- Decisions made in step 10 are consistent with decisions made in step 2
- Rewrites across many files stay architecturally consistent
- Better at greenfield builds (starting from scratch) than at incremental edits to existing code

**Feature comparison:**

| Feature | Cursor | Windsurf |
|---------|--------|----------|
| Base editor | VS Code fork | VS Code fork |
| Agent name | Composer | Cascade |
| Model options | GPT-4o, Claude 3.7, Gemini | Claude 3.7, GPT-4o, Gemini |
| Free tier | 2,000 completions, limited chat | More generous — 5 Cascade flows/day + unlimited autocomplete |
| Unique strength | Deep context on existing codebases | Coherent long-horizon task execution |
| Best for | Working on mature, complex projects | Building new features and services from scratch |
| Pricing (Pro) | $20/month | $15/month |

**When to use Windsurf over Cursor:**
- Building a new module, service, or application from scratch
- You want to describe an entire feature and come back to a working implementation
- Budget is a constraint — more capability in the free tier

**When to use Cursor over Windsurf:**
- Deep editing within a large existing codebase
- You want fast, reactive completions as you type
- Your team already has shared Cursor Rules configured

**Bottom line:** Download both. Do the same task in each. Developers consistently prefer one — but the preference varies by person and project type.

---

## Slide 18: GitHub Copilot and Enterprise AI Coding Tools

### The Enterprise Standard and the Ecosystem

**GitHub Copilot in 2026 — beyond autocomplete:**

| Feature | What It Does | Business Value |
|---------|-------------|---------------|
| **Copilot Autofix** | Scans code for security vulnerabilities; generates fixes automatically | Reduces CVE backlog without dedicated security sprints |
| **Copilot Workspace** | Takes a GitHub Issue, plans the implementation, proposes code changes | From ticket to PR without opening an IDE |
| **Copilot in PRs** | Reviews every pull request, suggests improvements, summarizes changes | Every PR gets an AI review pass before human reviewers |
| **Copilot Chat** | Conversational AI about any codebase — ask architecture questions, debug issues | Junior developers onboard faster; seniors stay in flow |
| **Copilot Extensions** | Connect Jira, Datadog, Sentry, and 30+ tools directly into the chat interface | One AI interface for your entire toolchain |

**Why enterprises choose Copilot:** It's not always the most capable agent — it's where the code already lives. GitHub manages the org's entire codebase, PR history, and issue backlog. Copilot integrates there, not alongside it.

**The broader enterprise AI coding ecosystem:**

| Tool | Vendor | Why It Exists |
|------|--------|--------------|
| **Amazon Q Developer** | AWS | Best for teams building on AWS — Lambda, DynamoDB, CDK-aware |
| **JetBrains AI** | JetBrains | Native in IntelliJ / PyCharm / DataGrip — Java, Kotlin, Python shops |
| **Tabnine** | Tabnine | On-premises deployment — zero data to cloud, required in some regulated industries |
| **Replit AI** | Replit | Browser-based, no install — ideal for learning and quick prototypes |

**Free access for students:** GitHub Copilot Pro is included in the **GitHub Student Developer Pack** at no cost. Apply at github.com/education with your UTD email.

---

## Slide 19: AI Coding Tools — Full Comparison and Decision Guide

### Choosing the Right Tool for Your Work

**Head-to-head comparison:**

| Tool | Interface | Autonomy | Model Quality | Free Tier | Paid Tier | Best For |
|------|-----------|----------|--------------|-----------|-----------|---------|
| **Claude Code** | Terminal CLI | ★★★★★ | Claude 3.7 (best coder) | Pay-as-you-go | ~$20/mo typical | Agentic tasks, multi-file changes |
| **Cursor** | IDE (VS Code) | ★★★★☆ | GPT-4o, Claude, Gemini | 2,000 completions | $20/mo | Daily dev work, large codebases |
| **Windsurf** | IDE (VS Code) | ★★★★☆ | Claude, GPT-4o | 5 flows/day | $15/mo | New features, long tasks |
| **GitHub Copilot** | IDE + GitHub | ★★★☆☆ | GPT-4o | Yes (limited) | $19/mo | Enterprise, GitHub-integrated teams |
| **Amazon Q** | IDE + CLI | ★★★☆☆ | Claude-powered | Yes | $19/mo | AWS-native development |
| **Replit AI** | Browser | ★★☆☆☆ | GPT-4o | Yes (limited) | $20/mo | Learning, no-install prototypes |

**Decision guide by situation:**

- In an enterprise with IT governance requirements → **GitHub Copilot** (approved vendor, SOC2, audit logs)
- Building Python data pipelines and analytics → **Cursor** or **Claude Code**
- Starting a new project from scratch → **Windsurf Cascade** for the first build
- Want maximum autonomy, hands-off task completion → **Claude Code**
- Student learning to code → **Replit AI** (browser, no setup) or **Windsurf** (generous free tier)
- AWS-heavy stack → **Amazon Q Developer**

**The meta-lesson:** These tools are skills, not features. Developers who spend time learning to prompt them well — how to give clear context, how to validate output, when to take manual control — get dramatically more value than those who just install and hope. Pick one and go deep before expanding.

---
