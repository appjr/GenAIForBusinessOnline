# Week 11 — Professor Teaching Scripts
# Batch 3 of 5: AI Coding Tools (Slides 14–19)
# Course: BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

---

## Slide 14 — The AI-Powered Development Revolution
### 🎤 Professor Script

"Let me set the stage for this section with some honest data.

In 2022, GitHub surveyed developers using Copilot and found that they completed tasks 55% faster on average. That was considered remarkable. Since then, the capability of these tools has grown dramatically — and the numbers have only improved.

A function that used to take a developer fifteen minutes to write — understanding requirements, looking up syntax, handling edge cases — now takes two minutes. You describe what you want, the AI writes it, you review and iterate. Boilerplate code that used to be forty-five minutes of mechanical typing? Thirty seconds.

But I want to be careful not to overstate this as just 'speed.' The more profound shift is in what's *possible*. Code that would have been outside a junior developer's skill set is now achievable. A data analyst who knows Python but not web frameworks can now build a simple API with AI assistance. A business analyst can automate a task that would have required a developer's help.

The table on this slide shows the quantitative case. But the qualitative shift matters just as much: the barrier between 'I can imagine this' and 'I can build this' has dropped dramatically.

This creates a real implication for your careers. The developers and analysts who pair AI tools with domain expertise — who can tell the AI what to build and validate what it produces — will be dramatically more productive than those who don't. That's not a distant future prediction. It's the current state of the job market.

The people most at risk are those who resist these tools because they feel like 'cheating.' Using AI coding assistance is not cheating any more than using Google is cheating. It's a professional skill."

---

## Slide 15 — Claude Code
### 🎤 Professor Script

"Claude Code is the tool I've been personally using most heavily over the last several months, and I want to share what makes it different.

Most AI coding tools are IDE plugins. They live inside your editor and help you write the code that's in front of you. They're reactive tools — you write, they suggest.

Claude Code is a terminal-based agent. You run it from your project directory. It reads your entire codebase. You give it a task — not a snippet to complete, but a goal to achieve — and it figures out what to do. It reads the files it needs to understand. It makes targeted changes across multiple files. It runs your tests to verify the changes work. If the tests fail, it diagnoses why and fixes the issue. And it keeps going until the task is done.

Let me give you a concrete example. I have a Python project with a bug in a checkout calculation. I open my terminal in the project directory and type: 'There's a bug in the discount calculation in checkout.py — the discount isn't being applied when the cart total is over $500.' Claude Code reads the file, identifies the issue, fixes it, runs the tests, and confirms they pass. The whole thing takes three minutes. I didn't navigate to the file, I didn't read through the code, I described the problem and it handled it.

This is qualitatively different from autocompletion.

The pricing model is worth understanding. Claude Code uses the Claude API — you pay per token. A typical task might cost ten to fifty cents. For a complex refactor across many files, maybe a couple of dollars. For most development workflows, the cost is negligible compared to the time saved.

One important note: Claude Code is most powerful when you tell it what you want, not how to do it. 'Refactor the authentication module to use JWT' works better than 'In auth.py line 47, change the session storage to use...'. Let it figure out the implementation details."

---

## Slide 16 — Cursor
### 🎤 Professor Script

"Cursor has become the AI coding tool that most professional developers I know have settled on as their daily driver. Let me explain why.

It starts with a philosophical choice Cursor made: instead of building a plugin on top of an existing IDE, they forked VS Code and built AI into every layer. The AI isn't grafted on — it's native. This means the AI has access to context that plugins don't — the full project structure, open files, recent edits, terminal output.

The four features that matter most in practice.

Tab completion — what they call Composer — is not your typical single-line autocomplete. It understands what you're trying to do and can complete entire functions, even multiple functions. It can also make multi-file edits — 'update all the API endpoints to use the new authentication middleware' — and it touches every relevant file in one operation.

Chat mode gives you a conversation partner that knows your entire codebase. You can @-mention specific files, functions, or documentation. 'How does the @UserService handle authentication, and where does it interact with @SessionManager?' The answer comes with full context.

Agent mode is where it becomes autonomous. You describe a task, and Cursor plans the implementation, writes the code, runs tests in the terminal, and iterates until it's working. This is the same pattern as Claude Code, but inside the IDE with full visual feedback.

And Cursor Rules is underrated. You write a set of project-specific rules — use this coding style, always write tests for new functions, follow our REST API conventions — and every AI interaction respects them. This is how teams maintain consistency when everyone's using AI assistance.

Twenty dollars a month for the Pro tier. If you're doing any meaningful amount of Python work, that's an easy ROI calculation."

---

## Slide 17 — Windsurf
### 🎤 Professor Script

"Windsurf, built by Codeium, launched at the end of 2024 and immediately positioned itself as Cursor's main competitor. Let me explain what makes it different, because they're genuinely distinct tools despite covering similar ground.

The standout feature is Cascade — Windsurf's autonomous coding agent. The key claim is 'coherence' — the agent maintains an understanding of the overall goal as it works through a complex task, rather than treating each step independently.

In practice, what this means is that Cascade tends to do better on tasks where you're asking it to build something substantial from scratch — a new feature, a new module, a new service. It maintains the context of what it's building and why, so decisions made in step ten are consistent with decisions made in step two.

Cursor, in contrast, tends to shine on tasks in existing codebases — understanding how things work, making targeted changes, navigating complexity. It's better at 'I have this code, here's what I want to change.'

The other thing worth noting: Windsurf has a more generous free tier. If you want to try AI-powered development without committing fifteen or twenty dollars a month, Windsurf's free plan includes more capability than Cursor's.

My honest advice? Download both. They take fifteen minutes to set up. Do the same task in each. Most developers develop a preference pretty quickly. I've seen people love Cursor and find Windsurf frustrating, and vice versa. It's a personal workflow thing.

What they share: both are dramatically more capable than any AI coding tool that existed two years ago. Either one will make you a more productive developer."

---

## Slide 18 — GitHub Copilot & Enterprise Tools
### 🎤 Professor Script

"GitHub Copilot is the tool you're most likely to encounter in an enterprise job, so let's understand it well.

The reason Copilot is dominant in enterprise isn't purely technical — it's ecosystem. GitHub is where enterprise code lives. Copilot integrates into the GitHub workflow at every level: in your IDE while you're writing code, in GitHub.com when you're reviewing PRs, in the issue tracker when you're planning work. That tight integration with where code actually lives is a genuine competitive advantage.

The enterprise features have also matured significantly. Copilot Autofix scans your code for security vulnerabilities and automatically generates fixes — this alone is a significant value proposition for security-conscious organizations. Copilot in pull requests means every code review gets an AI pass before human reviewers see it. Copilot Workspace lets you take a GitHub issue — 'implement user authentication' — and Copilot will plan the implementation and propose the code changes, all before you've opened your IDE.

The other tools worth knowing:

Amazon Q Developer is what Copilot is for AWS — if you're building on AWS services, using Lambda, DynamoDB, or CDK, Q Developer has context about those services that general tools don't. It can also help with migrations, which is a huge enterprise use case.

JetBrains AI matters if your organization uses IntelliJ or PyCharm — common in enterprise Java and Kotlin shops. It's native in those IDEs rather than being a plugin.

For all of you: GitHub Copilot has a free tier and a free student plan. If you're a student, apply for the Student Developer Pack at github.com/education. You get Copilot Pro for free. That's the full product, not a trial. Use it."

---

## Slide 19 — AI Coding Tools Comparison
### 🎤 Professor Script

"Let me synthesize this into practical guidance.

The comparison table on the slide gives you a structured view. Let me highlight the key differentiators.

Autonomy is the dimension that separates these tools most sharply. Claude Code is the most autonomous — it's an agent that completes tasks end-to-end, not just a suggestion engine. If you want to say 'fix the authentication bug in my project' and come back to a fixed project, Claude Code is the right tool. Cursor and Windsurf are close behind in agent mode. Copilot is still more suggestion-oriented, though it's catching up.

The interface choice matters too. If you have a strong preference for working in VS Code or a JetBrains IDE, Cursor or Copilot make sense — they're IDE-native. If you work heavily in the terminal, or you're running automation scripts, Claude Code's CLI interface is actually an advantage.

For the practical question of what to use for this course and your career: I'd recommend picking Cursor or Windsurf as your daily driver IDE, and adding Claude Code when you have complex multi-step tasks that benefit from autonomous execution.

And the key message I want to leave you with from this section: these tools are skills, not magic. The developers who get the most out of AI coding tools are the ones who've invested time learning to prompt them well. Knowing how to describe a problem clearly, how to give the AI enough context, how to validate the output — that's the actual skill. The tool is just the interface.

Go build something this week. That's the only way to develop this skill."

---
