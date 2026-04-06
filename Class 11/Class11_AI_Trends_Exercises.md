# Class 11 — Hands-On Exercises: Exploring New AI Tools & Trends
## BUAN 6v99 — Generative AI for Business | UTDallas Spring 2026

**Instructions:** Complete all 8 exercises. For each, write a short reflection (3–5 sentences) covering: what you did, what surprised you, and one business application you can envision. Bring your notes to the next class for discussion.

**Time estimate:** 15–30 minutes per exercise. Plan 3–4 hours total.

---

## Exercise 1: Model Shootout — Reasoning vs. Speed

**Tools:** Claude.ai (free), ChatGPT (free), Google Gemini (free)
**Topic:** Comparing frontier models on a hard business reasoning problem

### Objective
Experience firsthand how different model families (standard vs. reasoning) approach complex analysis.

### Steps

1. Formulate a complex, multi-step business question. Examples:
   - *"A retail company's Q4 sales dropped 18% despite running a bigger promotion than last year. Walk me through a structured diagnosis — what are the most likely causes, what data would you need to confirm each hypothesis, and what are your top 3 recommendations?"*
   - *"I'm evaluating two job offers: Company A offers $95k salary, strong brand name, high travel. Company B offers $85k + equity, small startup, remote. Walk me through a structured decision framework, including what questions I should ask before deciding."*

2. Send the **exact same prompt** to:
   - Claude.ai (Claude 3.7 Sonnet if available, otherwise standard)
   - ChatGPT (o4-mini if available; otherwise GPT-4o)
   - Google Gemini 2.0 Flash

3. Rate each response on:

| Criterion | Claude | ChatGPT | Gemini |
|-----------|--------|---------|--------|
| Logical structure of the answer | /10 | /10 | /10 |
| Depth of analysis | /10 | /10 | /10 |
| Actionability of recommendations | /10 | /10 | /10 |
| Response time (fast/medium/slow) | | | |
| **Total** | /30 | /30 | /30 |

4. Note: if you can access o3 or Claude Extended Thinking, try those too.

### Reflection Questions
- Did any model structure the problem in a way you hadn't thought of?
- How did reasoning quality differ from model speed?
- Which would you recommend for your specific industry's analytical tasks?

---

## Exercise 2: Build a Minimal AI Agent

**Tools:** Python (any IDE), Anthropic Claude API or OpenAI API
**Topic:** Agent architecture — tools, loop, reasoning

### Objective
Run the agent loop from Slide 13 yourself. See a tool get called and results returned.

### Setup
```bash
pip install anthropic python-dotenv
```

### Steps

1. Create a file called `my_first_agent.py`. Copy and paste this code:

```python
"""
Class 11 Exercise 2: My First AI Agent
"""
import anthropic
import os
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

# Define two simple tools
tools = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city (simulated)",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform a mathematical calculation",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"}
            },
            "required": ["expression"]
        }
    }
]

def run_tool(tool_name: str, tool_input: dict) -> str:
    """Execute the tool."""
    if tool_name == "get_weather":
        # Simulated weather data
        weather_data = {
            "Dallas": "75°F, Sunny",
            "New York": "62°F, Cloudy",
            "Seattle": "55°F, Rainy",
        }
        city = tool_input["city"]
        return weather_data.get(city, f"Weather data not available for {city}")

    elif tool_name == "calculate":
        try:
            # IMPORTANT: eval() is only safe here because we control the input
            # Never use eval() on untrusted user input in production
            result = eval(tool_input["expression"])
            return str(result)
        except Exception as e:
            return f"Calculation error: {e}"

    return "Tool not found"

def run_agent(task: str) -> str:
    """Run the agent loop."""
    messages = [{"role": "user", "content": task}]
    print(f"\nTask: {task}\n" + "="*50)

    while True:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )

        print(f"Stop reason: {response.stop_reason}")

        if response.stop_reason == "end_turn":
            final_text = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            return final_text

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → Calling tool: {block.name}({block.input})")
                result = run_tool(block.name, block.input)
                print(f"  ← Tool returned: {result}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    # Test 1: Simple tool use
    result1 = run_agent("What's the weather in Dallas?")
    print(f"\nAgent answer: {result1}\n")

    # Test 2: Multi-step reasoning with tools
    result2 = run_agent(
        "I'm visiting Dallas for a conference. The conference venue holds 250 people "
        "and charges $45/person for catering. What's the total catering cost, "
        "and what's the weather like there so I know how to pack?"
    )
    print(f"\nAgent answer: {result2}")
```

2. Create a `.env` file with your API key: `ANTHROPIC_API_KEY=your_key_here`

3. Run the script: `python my_first_agent.py`

4. Watch the tool calls print to the console. Note how the agent decides when to call tools.

5. **Modify the agent:** Add a third tool of your own choice — a business-relevant one. Ideas: `lookup_company_info`, `convert_currency`, `check_calendar_availability`. Make it return realistic simulated data.

### Reflection Questions
- How did the agent "decide" which tool to call? How does it know?
- What happens if you ask a question the tools can't help with?
- What would make this agent genuinely useful for a business task?

---

## Exercise 3: Try Claude Code (or Cursor Agent Mode)

**Tool:** Claude Code (CLI) or Cursor Agent Mode
**Topic:** Agentic coding — autonomous multi-step task completion

### Objective
Experience an AI coding agent completing a real task end-to-end, not just suggesting completions.

### Option A: Claude Code (Terminal)

```bash
# Install
npm install -g @anthropic-ai/claude-code

# Navigate to a Python project (or create a new folder)
mkdir ai_test_project && cd ai_test_project

# Launch Claude Code
claude
```

Give it this task:
```
Create a Python script called analyze_sales.py that:
1. Generates a sample dataset of 100 sales records with: date, product, region, and amount
2. Calculates: total sales by product, total sales by region, monthly trends
3. Prints a formatted summary report
4. Saves the data to sales_data.csv
```

Watch what it does: reads context, writes the file, potentially runs it to verify.

### Option B: Cursor Agent Mode

1. Download Cursor from cursor.com
2. Open a new folder as a project
3. Press Ctrl+I to open Composer
4. Type the same task prompt as Option A
5. Watch it write the code, then ask it to run and fix any issues

### Steps for both options

1. Run the task described above
2. When the code is generated, run it yourself to verify it works
3. Ask a follow-up: "Add a bar chart visualization using matplotlib and save it as sales_chart.png"
4. Evaluate what the agent got right and wrong

### Reflection Questions
- Did the AI complete the task correctly on the first try?
- How did it handle the follow-up request — did it maintain context?
- What would you still need to do manually that the AI couldn't handle?
- How does this change what you could build without formal software training?

---

## Exercise 4: Build a No-Code AI Workflow

**Tool:** Zapier (zapier.com) — free tier, OR Make (make.com) — free tier
**Topic:** AI-powered business automation without code

### Objective
Build a working AI automation that processes real information and produces a useful output.

### Scenario: AI Email Summarizer + Triage

Build a workflow that:
1. **Trigger:** New email arrives (or use a manual trigger for the free tier)
2. **AI Step:** Classify the email and generate a structured summary
3. **Output:** Log to a Google Sheet OR send a Slack message

### Steps (Zapier version)

1. Go to **zapier.com** and create a free account
2. Create a new Zap. Click **"Create"**
3. **Trigger:** Choose "Email by Zapier" → "New Inbound Email" (Zapier gives you a custom email address to use)
4. **Action:** Add an "AI by Zapier" step
   - Use this prompt:
   ```
   Analyze this email and return:
   1. CATEGORY: one of [Sales, Support, Internal, Spam, Other]
   2. PRIORITY: High / Medium / Low
   3. SUMMARY: one sentence
   4. ACTION NEEDED: yes or no, with what action

   Email: {{Body}}
   ```
5. **Final Action:** Add a "Google Sheets" step to log the AI output, or a "Slack" step to notify a channel

6. Test by sending an email to your Zapier address and watching it flow through

### Steps (Make version)

1. Go to **make.com** and create a free account
2. Create a new Scenario
3. **Module 1:** Google Forms or Email trigger
4. **Module 2:** OpenAI → "Create Completion" with the analysis prompt
5. **Module 3:** Google Sheets → "Add a Row" with the results

### Reflection Questions
- How long did it take to build this workflow vs. coding it from scratch?
- What would break if the input format changed?
- What business process in your future career could benefit from this pattern?

---

## Exercise 5: RAG — Chat with Your Own Documents

**Tool:** Python + LlamaIndex (local), OR Google NotebookLM (no-code)
**Topic:** Retrieval-Augmented Generation for private knowledge

### Objective
Build a system that answers questions grounded in documents you provide — not the model's training data.

### Option A: Python + LlamaIndex

```bash
pip install llama-index llama-index-llms-anthropic
```

```python
"""
Class 11 Exercise 5: Chat with your documents using RAG
"""
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.anthropic import Anthropic

# Setup
Settings.llm = Anthropic(model="claude-haiku-4-5-20251001")

# Step 1: Create a folder called 'my_docs' and add 3+ text or PDF files
# (Use course slides, articles, company reports, etc.)
import os
os.makedirs("my_docs", exist_ok=True)

# For this exercise, let's create sample documents
sample_docs = [
    ("my_docs/company_policy.txt",
     """ACME Corp Remote Work Policy (2026)
     Employees may work remotely up to 3 days per week.
     Home office stipend: $500/year for equipment.
     Core hours are 10am-3pm Central Time.
     VPN required for all company system access.
     Remote work agreements must be renewed annually."""),

    ("my_docs/benefits_guide.txt",
     """ACME Corp Benefits Summary
     Health insurance: BCBS PPO, company pays 80% premium.
     401k: 4% company match, vesting after 2 years.
     PTO: 15 days/year, increases to 20 days after 3 years.
     Parental leave: 12 weeks paid for primary caregiver.
     Professional development: $2,000/year for courses and conferences."""),

    ("my_docs/expense_policy.txt",
     """ACME Corp Expense Reimbursement Policy
     Meals: up to $75/day when traveling.
     Hotel: up to $200/night in major cities.
     Flights: book economy 14+ days in advance when possible.
     Submit expenses within 30 days of incurring them.
     Manager approval required for expenses over $500.""")
]

for filepath, content in sample_docs:
    with open(filepath, "w") as f:
        f.write(content)

# Step 2: Index the documents
print("Indexing documents...")
documents = SimpleDirectoryReader("my_docs").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

# Step 3: Ask questions
questions = [
    "How many remote work days per week are allowed?",
    "What is the 401k match policy?",
    "How much can I spend on a hotel when traveling?",
    "What is the deadline to submit expense reports?",
    "Does the company offer parental leave?"
]

print("\n=== RAG Q&A ===")
for question in questions:
    response = query_engine.query(question)
    print(f"\nQ: {question}")
    print(f"A: {response}")
```

Run this and verify the answers are correct based on the documents.

**Then add your own real documents** — course slides saved as .txt, a public company policy, a news article — and ask questions about them.

### Option B: Google NotebookLM (No-code)

1. Go to **notebooklm.google.com** and sign in
2. Create a new Notebook
3. Add 3–5 documents (PDFs, Google Docs, URLs, or pasted text)
4. Ask questions in the chat — note how answers cite specific sources
5. Try: "What are the main themes across all my sources?" and "What do the sources disagree about?"

### Reflection Questions
- Did the RAG system correctly use the documents to answer questions?
- Did it refuse to answer questions that weren't in the documents?
- What would happen if the documents contradicted each other?
- For what business use case at your future employer would this be most valuable?

---

## Exercise 6: Explore an AI Agent Framework

**Tool:** n8n (n8n.io) — free self-hosted OR n8n Cloud free trial
**Topic:** AI agents in a no-code workflow context

### Objective
Build a multi-step AI agent workflow that handles a business research task.

### Steps

1. **Install n8n locally via Docker:**
```bash
docker run -it --rm \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```
Or sign up for **n8n Cloud free trial** at n8n.io

2. Open n8n at http://localhost:5678

3. Create a new workflow. Build this structure:
   - **Start:** Manual trigger
   - **Node 1:** Set node — define a company name as a variable (e.g., "Salesforce")
   - **Node 2:** AI Agent node — configure with Anthropic or OpenAI credentials
     - System prompt: *"You are a business research analyst. When given a company name, provide: 1) What they do, 2) Their main product/service, 3) Their approximate revenue or market position, 4) Their top 2 competitors."*
     - User message: Use the company name from Node 1
   - **Node 3:** Set node — format the output
   - **Node 4:** No-op (or connect to email/sheet in advanced version)

4. Run the workflow for 3 different company names and review outputs

5. Modify the AI prompt to focus on a specific industry or question type

### Reflection Questions
- How does the n8n AI Agent node differ from directly calling an API?
- What advantage does a visual workflow tool offer over Python code for this task?
- What types of team members (non-technical) could use this workflow builder?

---

## Exercise 7: Multimodal AI — Vision in Business

**Tool:** Claude.ai (file upload) or GPT-4o (ChatGPT with image)
**Topic:** Using vision AI for business document processing

### Objective
Experience how multimodal AI handles real business documents and images.

### Steps

1. Find or create 2–3 business documents with visual content:
   - A screenshot of a dashboard or chart (from Excel, Tableau, any tool)
   - A page from a PDF report that contains a table or graph
   - A product photo or marketing image
   - A screenshot of a website's pricing page

2. **Test 1 — Chart analysis:** Upload a chart/graph image and ask:
   ```
   Analyze this chart. Tell me:
   1. What metric is being measured?
   2. What is the key trend or finding?
   3. What questions would you ask the data team to validate this insight?
   4. What business decision could this chart support?
   ```

3. **Test 2 — Document extraction:** Upload a table or structured document and ask:
   ```
   Extract all data from this table into a clean markdown table format.
   Then summarize the 3 most important insights from this data.
   ```

4. **Test 3 — Pricing page analysis:** Upload a competitor's pricing page screenshot and ask:
   ```
   Analyze this pricing page. What pricing strategy are they using?
   What value propositions do they emphasize? Who is the target customer for each tier?
   ```

5. Compare Claude vs. GPT-4o on the same image for one of your tests.

### Reflection Questions
- How accurate was the AI at reading the visual content?
- What types of business documents would benefit most from this capability?
- What are the risks of using AI to process sensitive documents (e.g., financial statements)?

---

## Exercise 8: Evaluate a Fine-Tuned vs. Base Model

**Tool:** Python + OpenAI API or Hugging Face
**Topic:** Understanding when and why fine-tuning helps

### Objective
See the difference between a base model and a task-adapted model on the same prompt — and understand how to measure the improvement.

### Steps

1. Go to **OpenAI Playground** (platform.openai.com/playground) — free with an API key

2. **Establish a baseline:** Select `gpt-4o-mini` as your model. Test it on a domain-specific task. Example:
   ```
   Write a professional LinkedIn recommendation for a data analyst with 3 years of experience
   who is strong in Python and SQL, led a successful pricing optimization project, and is
   detail-oriented and collaborative.
   ```
   Save this response.

3. **Add style instructions via system prompt:** Change the system prompt to:
   ```
   You write LinkedIn recommendations in a distinctive style:
   - Always start with a specific project achievement
   - Use concrete metrics when possible
   - Keep recommendations to exactly 150 words
   - End with a forward-looking statement about the person's potential
   ```
   Run the same prompt again. Compare.

4. **Explore a fine-tuned model:** Hugging Face (huggingface.co) hosts thousands of fine-tuned models. Search for a domain-specific model in your field:
   - Finance: search "finance LLM"
   - Legal: search "legal BERT" or "legal-roberta"
   - Medical: search "BioMedLM" or "clinical BERT"

   In the model's Space (if available), test it on a domain question and compare to GPT-4o-mini.

5. **Document the comparison:**

| Criterion | Base Model | Prompted Model | Fine-Tuned Model |
|-----------|-----------|----------------|------------------|
| Follows domain conventions | | | |
| Factual accuracy | | | |
| Style consistency | | | |
| Setup effort | None | Minutes | Hours–Days |
| Cost to run | Standard | Standard | Lower (smaller) |

### Reflection Questions
- How much did the system prompt improve the output vs. the base model?
- When would the extra effort of fine-tuning be worth it?
- If you were at a company, what would you fine-tune a model on?

---

## Submission Guidelines

**What to submit:**
1. A document with your reflections for each of the 8 exercises (3–5 sentences each)
2. Screenshots or code showing at least 4 of the 8 completed exercises
3. Your code from Exercise 2 (the agent) — even if it's not perfect

**Discussion preparation:**
Come to the next class ready to share:
- The one tool that surprised you most — positively or negatively
- One concrete business application you would actually build
- One question about agentic AI or workflow automation you couldn't answer

**Grading:** Completion + genuine reflection. There are no "wrong" answers.

---

*Class 11 Exercises — BUAN 6v99 Generative AI for Business — Spring 2026*
*University of Texas at Dallas | Professor Antonio de Pádua Paes Jr.*
