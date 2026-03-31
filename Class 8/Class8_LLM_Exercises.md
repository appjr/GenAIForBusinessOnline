# Class 8 — Hands-On Exercises: Testing the World's Best LLMs
## BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

**Instructions:** Complete all 8 exercises. For each one, write a short paragraph (3–5 sentences) reflecting on your experience — what surprised you, what worked well, and what limitations you noticed. Bring your notes to the next class for group discussion.

**Time estimate:** Each exercise takes 15–30 minutes. Plan for 3–4 hours total.

---

## Exercise 1: ChatGPT — Executive Document Analysis

**Tool:** ChatGPT (chat.openai.com) — free account is sufficient; GPT-4o recommended
**Topic:** Business document summarization and strategic analysis

### Objective
Learn to use ChatGPT to extract business intelligence from a real document and compare behavior across different models.

### Steps

1. Find a publicly available business document — options:
   - A company's annual report (find on their investor relations page)
   - A McKinsey or BCG report (search "McKinsey PDF 2024")
   - The UTDallas BUAN 6v99 course syllabus PDF

2. Open ChatGPT. If you have Plus ($20/mo), select **GPT-4o**. Otherwise use the free default.

3. Upload the document using the paperclip icon. Then send this prompt:
   ```
   Please provide:
   1. An executive summary in 3 bullet points (max 50 words each)
   2. The top 3 strategic risks or challenges mentioned
   3. 3 actionable recommendations based on this document
   4. Rate the overall quality/credibility of this document on a scale of 1–10 and explain why
   ```

4. If you have access, repeat the same prompt using **o3-mini** (reasoning model). Notice the difference in depth and time taken.

### Expected Output
- A structured 4-part analysis of your document
- If using two models: a comparison of how o3-mini reasons vs. GPT-4o

### Reflection Questions
- How accurate was the summary? Did it miss anything important?
- How long did it take? Did the model "hallucinate" any facts?
- Would you trust this output without reading the source document?
- How could you use this in a real business workflow?

---

## Exercise 2: Claude — Long-Document Risk Analysis

**Tool:** Claude (claude.ai) — free account (limited messages per day)
**Topic:** Contract and legal document analysis

### Objective
Experience Claude's exceptional ability to handle very long documents and extract structured insights with high precision.

### Steps

1. Find a long document to analyze — options:
   - Any publicly available business contract or Terms of Service (Apple's ToS, a lease agreement, an employment contract template)
   - A long research report (try: search "industry report PDF 2024 free")
   - A public government regulation document

2. Open Claude.ai. Click the paperclip to upload the document.

3. Send this prompt:
   ```
   I need a thorough risk analysis of this document. Please:
   1. Identify the top 5 risks or obligations I should be aware of
   2. Flag any clauses that are unusual or one-sided
   3. List any deadlines, renewal dates, or time-sensitive terms
   4. Summarize what I'm agreeing to in plain language (no legal jargon)
   5. Give me 3 questions I should ask before signing/accepting this
   ```

4. Follow up with: *"What is the single most important thing I should pay attention to in this document?"*

5. Compare Claude's response with what ChatGPT gives you for the same document.

### Expected Output
- A structured risk report for your chosen document
- A plain-language explanation of its key terms
- A comparison note between Claude and ChatGPT

### Reflection Questions
- Did Claude catch things ChatGPT missed (or vice versa)?
- How did Claude handle ambiguous language in the document?
- Would a lawyer still be needed after this analysis? When?
- What businesses could save the most money using this approach?

---

## Exercise 3: Google Gemini + NotebookLM — Research Podcast

**Tool:** NotebookLM (notebooklm.google.com) — free with Google account
**Topic:** AI-powered research synthesis

### Objective
Use NotebookLM to digest multiple sources and generate an audio "podcast" discussion — one of the most impressive AI features available for free today.

### Steps

1. Go to **notebooklm.google.com** and sign in with your Google account.

2. Create a new Notebook. Add 3–5 sources on a topic of your choice:
   - Paste URLs of news articles (try: "LLMs in healthcare 2024")
   - Upload a PDF
   - Add a YouTube video URL
   - Paste text from a Google Doc

   Suggested topic: *"The impact of AI on [your intended career field]"*

3. Once sources are loaded, try these queries in the chat:
   - *"What are the main themes across all my sources?"*
   - *"What do these sources disagree about?"*
   - *"Create a FAQ about this topic based only on my sources"*
   - *"Create a timeline of key events mentioned across my sources"*

4. Click **"Audio Overview"** (the headphone icon). Click Generate. Wait 2–5 minutes.

5. Listen to the ~10-minute AI-generated podcast of two AI hosts discussing your sources.

### Expected Output
- A summary of key themes across your sources
- A generated FAQ and timeline
- A downloadable MP3 podcast about your research topic

### Reflection Questions
- How accurate was the podcast compared to your source material?
- Did the AI "hallucinate" anything not in your sources?
- How could this be used for business: training, onboarding, competitive intelligence?
- What limitations did you notice in NotebookLM?

---

## Exercise 4: Perplexity AI — Research vs. Google Search

**Tool:** Perplexity AI (perplexity.ai) — free account
**Topic:** Business and market research

### Objective
Compare AI-powered research (Perplexity) with traditional search (Google) to understand when each is better.

### Steps

1. Choose a research question relevant to your area of study or future career. Examples:
   - *"What is the current market size and growth rate of the electric vehicle industry?"*
   - *"What are the biggest challenges facing the retail industry in 2025?"*
   - *"How are banks using AI in risk management right now?"*

2. **Google Search first:** Search your question on Google. Note:
   - How many results you need to read to find the answer
   - How long it takes to synthesize a complete answer
   - The date of the most recent relevant result

3. **Perplexity next:** Go to perplexity.ai. Type the same question. Note:
   - How the answer is presented (synthesized vs. links)
   - What sources it cites
   - Whether it mentions any dates or current events
   - Try clicking "Pro Search" if available for a deeper answer

4. If you have a Pro account, click **"Research"** mode for a multi-step deep-dive analysis.

5. Ask a follow-up question to Perplexity: *"What are 3 investment opportunities or business risks related to this trend?"*

### Expected Output
- Two research results (Google vs. Perplexity) for the same question
- A 1-paragraph comparison: accuracy, completeness, time spent, citation quality
- A follow-up business insight from Perplexity

### Reflection Questions
- In which scenarios is Perplexity clearly better than Google? When is Google still better?
- How did the source quality compare?
- Would you trust Perplexity's answer without checking sources? Why or why not?
- How could your future employer use Perplexity for competitive intelligence?

---

## Exercise 5: GitHub Copilot or Cursor — AI-Assisted Coding

**Tool:** GitHub Copilot (free for students) OR Cursor (cursor.com — free tier)
**Topic:** Coding with AI assistance

### Objective
Experience how AI coding tools can help non-developers write functional scripts for business tasks.

### Setup Options (choose one)
- **Option A:** Activate GitHub Copilot free student access at github.com/education
- **Option B:** Download Cursor from cursor.com (free tier, no credit card)
- **Option C:** Use Replit.com (browser-based, no install) with AI features

### Steps

1. Open your chosen tool and create a new Python file.

2. Type this comment (don't write any code — let the AI do it):
   ```python
   # Read a CSV file, calculate the average, min, max, and total
   # for each numeric column, and print a formatted summary report
   ```

3. Press Tab or Enter and watch the AI autocomplete the code. If nothing appears, press Ctrl+Enter (Copilot) or type a few characters.

4. Run the code. If it fails, copy the error message, paste it into the AI chat, and ask it to fix the bug.

5. Now try this harder prompt in the chat:
   ```
   Write a Python script that:
   - Reads a list of company names from a text file (one per line)
   - For each company, generates a one-sentence description of what they do
   - Saves the results to a new CSV file with columns: Company, Description
   Note: Use the OpenAI API for the descriptions (add a comment where the API key goes)
   ```

6. Review the generated code. Note what it got right, what it got wrong.

### Expected Output
- A working CSV analysis script
- A company description generator script (may need your own API key to run fully)
- Notes on how the AI performed

### Reflection Questions
- Could someone with no coding background use this to build something useful?
- What was the AI's biggest mistake or limitation?
- How does this change the role of software developers?
- What business tasks in your career could you automate with this approach?

---

## Exercise 6: Meta AI — Free vs. Paid LLM Comparison

**Tool:** Meta AI (meta.ai or in WhatsApp/Instagram) — completely free
**Topic:** Quality comparison across free and paid LLMs

### Objective
Test whether free LLMs have caught up to paid alternatives for common business tasks.

### Steps

1. Go to **meta.ai** (no account required) or open Meta AI in WhatsApp or Instagram.

2. Choose one of the following business prompts:
   - **Option A (Writing):** *"Write a professional LinkedIn post announcing that I just completed a Master's in Business Analytics with a specialization in AI. Make it engaging, include a key learning, and invite connections."*
   - **Option B (Analysis):** *"I'm considering opening a coffee shop near a university campus. What are the top 5 risks I should plan for, and what are 3 things successful campus coffee shops do that others don't?"*
   - **Option C (Strategy):** *"Our company's customer satisfaction scores have dropped 15% in 6 months. Walk me through a structured approach to diagnosing the root cause and proposing solutions."*

3. Copy the exact same prompt into ChatGPT (free tier), Claude (free tier), and Gemini (free tier).

4. Create a simple comparison table:

| Criteria | Meta AI | ChatGPT (free) | Claude (free) | Gemini (free) |
|---------|---------|----------------|---------------|---------------|
| Quality of answer (1–10) | | | | |
| Depth of insight | | | | |
| Structure/formatting | | | | |
| Response length | | | | |
| Tone (professional?) | | | | |
| Surprised me? | | | | |

### Expected Output
- 4 responses to the same prompt
- A completed comparison table with your ratings
- A recommendation: which free tool would you use for this task and why?

### Reflection Questions
- Did the free versions perform as well as you expected?
- Was there a clear winner for this type of task?
- What would justify paying $20/month for a premium plan?
- How has the gap between free and paid AI changed in the last year?

---

## Exercise 7: Ollama — Run an LLM Locally on Your Computer

**Tool:** Ollama (ollama.com) — free, runs on your machine
**Topic:** Private, local AI for sensitive business data

### Objective
Install and run an LLM completely locally — no internet required after setup, no data ever sent to the cloud.

### System Requirements
- Mac with Apple Silicon (M1/M2/M3) or Mac with 8GB+ RAM
- Windows or Linux with 8GB+ RAM (GPU preferred but not required)
- 5–8 GB of free disk space

### Steps

1. Go to **ollama.com** and download Ollama for your OS. Install it.

2. Open your Terminal (Mac: Cmd+Space → Terminal; Windows: PowerShell)

3. Pull and run the Llama 3.2 model (3B — lightweight, fast):
   ```bash
   ollama pull llama3.2
   ollama run llama3.2
   ```

4. You'll see a `>>>` prompt. You're now chatting with a local LLM. Try:
   ```
   Explain the difference between supervised and unsupervised machine learning in simple business terms
   ```

5. Now test it with something you'd NEVER send to a cloud LLM:
   ```
   I'm considering leaving my current employer. Here are my key concerns:
   [write 3-4 sentences about hypothetical job concerns]
   What questions should I be asking myself before making this decision?
   ```

6. Time the response. Note the speed compared to ChatGPT.

7. Try a larger model (if you have 16GB+ RAM):
   ```bash
   ollama pull llama3.1:8b
   ollama run llama3.1:8b
   ```

8. Type `/bye` to exit.

### Expected Output
- A working local LLM on your machine
- Response to a sensitive question (processed 100% locally)
- Speed and quality notes comparing Llama 3.2 vs. 3.1:8b

### Reflection Questions
- How did response quality compare to ChatGPT or Claude?
- What types of sensitive business data would benefit from local AI processing?
- What industries have the most to gain from private, local LLMs?
- What are the limitations of running AI locally vs. in the cloud?

---

## Exercise 8: Groq — Blazing-Fast AI Inference

**Tool:** Groq Console (console.groq.com) — free API tier
**Topic:** Speed-optimized AI for real-time applications

### Objective
Experience inference at 10× the speed of typical cloud APIs — and understand when speed matters for business applications.

### Steps

1. Go to **console.groq.com** and create a free account.

2. Navigate to **Playground**. You'll see a chat interface with model selection.

3. Select **Llama-3.3-70B-Versatile** (the highest quality free model on Groq).

4. Note the "tokens/second" counter that appears after each response.

5. Run this prompt and record the response time:
   ```
   What are the top 5 business use cases for large language models in the
   financial services industry? For each use case, include: the specific task
   AI handles, estimated ROI, and one real company already doing it.
   ```

6. Now go to **claude.ai** and run the exact same prompt. Note the response time.

7. Go to your Groq playground settings and change the model to **Mixtral-8x7B**. Run the same prompt again. Compare quality and speed.

8. (Optional) If you're comfortable with APIs: Get a free Groq API key from the console, and use it in a Python script or any HTTP client to make an API call.

### Expected Output
- Response time comparison: Groq vs. Claude vs. ChatGPT
- Quality comparison of Llama 3.3-70B vs. Mixtral-8x7B on the same prompt
- A list of business scenarios where this speed matters

### Reflection Questions
- How many tokens per second did Groq achieve vs. standard cloud LLMs?
- In what business scenarios does response speed (latency) matter most?
- Could Groq replace a paid LLM for your use case? Why or why not?
- What trade-offs exist between speed, quality, and cost across these platforms?

---

## Submission Guidelines

**What to submit:**
1. A document (Word, PDF, or Google Doc) with your reflections for all 8 exercises
2. At least one screenshot per exercise showing the AI output you received
3. Your completed comparison table from Exercise 6

**Discussion preparation:**
Come to the next class ready to share:
- Your single most surprising discovery across all exercises
- The tool you would recommend to a colleague for a specific task (and why)
- One limitation or failure you encountered that changed your view of AI capabilities

**Grading:** Completion + thoughtful reflection. There are no "wrong" answers — the goal is hands-on experience and critical thinking.

---

*Class 8 Exercises — BUAN 6v99 Generative AI for Business — Spring 2026*
*University of Texas at Dallas | Professor Antonio de Pádua Paes Jr.*
