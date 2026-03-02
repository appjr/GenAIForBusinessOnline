# Week 7: Text Generation, Chatbots - Slides Batch 5 (Slides 21-27)

**Course:** BUAN 6v99.SW2 - Generative AI for Business
**Topic:** Ethics, Case Studies, Lab & Key Takeaways

---

## Slide 21: Ethics & Responsible Chatbot Design

### Building AI You Can Be Proud Of

**The 5 Ethical Principles for Chatbot Deployment:**

```
1. TRANSPARENCY   - Users must know they're talking to AI
2. HONESTY        - Don't let the bot make things up
3. FAIRNESS       - Test for bias across user demographics
4. PRIVACY        - Handle conversation data responsibly
5. SAFETY         - Protect vulnerable users
```

**Principle 1: Transparency — Disclose the AI**

```python
# ❌ WRONG: Pretending to be human
bad_system = """
You are Jennifer, a human customer service representative.
If asked if you're an AI, say you are a real person.
"""

# ✅ CORRECT: Be honest about being AI
good_system = """
You are Aria, an AI assistant for TechShop.
If asked whether you're an AI, confirm that you are — be honest.
You can say: "Yes, I'm an AI assistant! I'm here to help with
your questions. For complex issues, I can connect you with a
human agent."
"""

# In your UI:
# ✅ Show a robot icon, not a person photo
# ✅ Label the chat "AI Assistant" not "Agent Jennifer"
# ✅ Include a note: "You're chatting with an AI. Human agents available if needed."
```

**Principle 2: Handling Hallucinations (Honesty)**

LLMs can confidently state false information. Mitigate this with:

```python
# Strategy 1: RAG grounds answers in real data (best solution)
# Strategy 2: Explicit instructions to admit uncertainty

anti_hallucination_system = """
CRITICAL RULES about information accuracy:
1. Only state facts you are confident about
2. If uncertain, say: "I'm not certain about this — let me suggest
   you verify with [official source]."
3. Never invent product specifications, prices, or policies
4. When in doubt, say: "That's a great question. To get the most
   accurate answer, please contact our team at support@techshop.com"
5. Do not extrapolate or make assumptions about specific cases
"""

# Strategy 3: Confidence signals
def generate_with_confidence_check(prompt: str) -> str:
    """Ask the model to flag uncertain responses."""
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
Answer the question. If you are NOT fully confident in any part of your answer,
add [VERIFY] before that part. Example:
"The return window is 30 days [VERIFY - please confirm on our website]."
"""},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content
```

**Principle 3: Fairness — Testing for Bias**

```python
def test_chatbot_for_bias(bot_chat_function, topic: str) -> dict:
    """
    Test if a chatbot responds differently based on demographics.
    Reveals potential bias in AI responses.
    """
    # Test same question with different demographic contexts
    test_prompts = [
        f"As a 65-year-old woman, I want to know: {topic}",
        f"As a 22-year-old man, I want to know: {topic}",
        f"I'm from rural Alabama. {topic}",
        f"I'm from New York City. {topic}",
        f"English is my second language. {topic}"
    ]

    responses = {}
    for prompt in test_prompts:
        response = bot_chat_function(prompt)
        responses[prompt[:50]] = response

    # Manual review: are responses meaningfully different?
    # Automated: compare response lengths, sentiment, helpfulness scores
    return responses

# Bias testing areas to check:
bias_concerns = {
    "Differential responses": "Does the bot answer differently based on demographics?",
    "Tone differences": "Is the tone more/less professional for different users?",
    "Assumption making": "Does the bot assume different intent based on background?",
    "Language support": "Does quality drop significantly for non-native speakers?"
}
```

**Principle 4: Privacy**

```python
# ✅ Privacy-respecting chatbot practices

# 1. Don't collect what you don't need
minimal_logging = {
    "log": ["session_id", "timestamp", "message_count", "resolved"],
    "do_not_log": ["full_conversation_text", "user_email", "IP_address"]
}

# 2. Data retention — auto-delete old conversations
def setup_retention_policy():
    """
    Conversation data should be deleted after a set period.
    Most chatbots need 30-90 days max for quality improvement.
    """
    return {
        "anonymize_after_days": 7,    # Remove PII
        "delete_after_days": 90,       # Full deletion
        "never_delete": ["aggregated metrics", "quality scores"]
    }

# 3. Clear privacy disclosure
privacy_disclosure = """
💬 Chat Privacy Notice:
This conversation is with an AI assistant.
• Conversations may be reviewed for quality improvement
• We don't store your name or personal information
• Refresh to start a new, separate conversation
• For privacy questions: privacy@company.com
"""

# 4. Detect and don't store accidentally shared PII
def detect_pii(text: str) -> bool:
    """Detect if text contains PII that shouldn't be stored."""
    import re
    patterns = {
        "SSN": r'\b\d{3}-\d{2}-\d{4}\b',
        "Credit Card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
        "Email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "Phone": r'\b\d{3}[.- ]?\d{3}[.- ]?\d{4}\b'
    }
    for name, pattern in patterns.items():
        if re.search(pattern, text):
            return True  # Contains PII — handle carefully
    return False
```

**Principle 5: Protecting Vulnerable Users**

```python
safety_system = """
SPECIAL USER PROTECTION PROTOCOLS:

1. CRISIS SITUATIONS (mental health, self-harm mentions):
   Immediately provide: "I hear you're going through something difficult.
   Please reach out to the 988 Suicide & Crisis Lifeline (call or text 988).
   They're available 24/7 and are ready to help."
   Then end the conversation gently — do not attempt to counsel.

2. EMERGENCY SITUATIONS:
   If user mentions medical emergency, fire, crime in progress:
   "Please call 911 immediately."
   Do not engage with the substance of the emergency.

3. MINORS (if context suggests user is a child):
   - Use age-appropriate language
   - Decline to discuss adult topics
   - Suggest parental guidance for decisions
   - Do not request personal information

4. FINANCIAL VULNERABILITY:
   If user seems to be in financial crisis:
   - Do not upsell or pressure
   - Mention payment plans or assistance programs
   - Provide human escalation path
"""
```

---

## Slide 22: Real-World Case Studies

### Success Stories & Lessons Learned

**Case Study 1: Klarna — Replacing Human Agents at Scale**

**Background:**
- Global payments company (Klarna) deployed a ChatGPT-powered customer service assistant
- Handles 35 languages, available 24/7 globally

**Results (first month):**

```python
klarna_results = {
    "conversations_handled": "2.3 million",
    "equivalent_fulltime_agents": 700,
    "resolution_rate": "75%",
    "repeat_inquiries": "-25% (better first-contact resolution)",
    "customer_satisfaction": "Equal to human agents",
    "cost_savings_per_year": "$40 million",
    "error_rate_reduction": "25% fewer errors in returns/refunds"
}

for metric, value in klarna_results.items():
    print(f"{metric}: {value}")

# Key insight: AI is now handling the same workload as 700 employees
# at a fraction of the cost — but with same CSAT scores
```

**Lessons Learned:**
- ✅ Start with well-defined use cases (not "do everything")
- ✅ Invest heavily in the knowledge base and RAG
- ✅ Human escalation path is critical — 25% still needed humans
- ✅ Quality is measured by CSAT, not just cost savings
- ❌ Lesson: Don't announce "replacing X employees" — employee trust matters

---

**Case Study 2: Bank of America — Erica Financial Assistant**

**Background:**
- 40 million+ users interact with "Erica" (Bank of America's AI)
- Handles routine banking inquiries entirely within the app

```python
erica_capabilities = {
    "balance_inquiries": "Real-time account balances",
    "transaction_search": "Find transactions by merchant, amount, date",
    "spending_insights": "Personalized spending analysis",
    "bill_reminders": "Upcoming payment alerts",
    "credit_score": "Free FICO score monitoring",
    "fraud_alerts": "Proactive suspicious activity notifications"
}

erica_stats_2025 = {
    "total_interactions": "2 billion+",
    "monthly_active_users": "18 million",
    "avg_interactions_per_user": "18/month",
    "task_completion_rate": "94%",
    "avg_response_time_seconds": 0.8,
    "reduction_in_call_center_volume": "30%"
}
```

**Why It Works:**
1. **Narrow scope**: Erica does banking — nothing else
2. **Personalized data**: Has access to YOUR account (with auth)
3. **Proactive**: Alerts you before problems, not just reactive
4. **Integrated**: In the app you already use daily
5. **Safe escalation**: One tap to reach a human banker

---

**Case Study 3: Duolingo — Max, the AI Tutor**

**Background:**
- Language learning app added "Duolingo Max" with AI conversation practice
- Uses GPT-4 to create unlimited realistic conversation scenarios

```python
duolingo_max_features = {
    "explain_my_answer": "Explains why your answer was wrong or right",
    "roleplay": "Practice conversation as: waiter, doctor, hotel clerk, etc.",
    "pronunciation": "Feedback on speaking with AI",
    "adaptive_difficulty": "Adjusts to your level in real-time"
}

# Business impact
duolingo_results = {
    "subscription_tier": "Max tier (premium)",
    "price_premium": "$5/month more than base subscription",
    "user_retention_improvement": "+40% for Max users vs. base",
    "lesson_completion": "+65% for Max users",
    "revenue_impact": "Significant contributor to premium growth"
}
```

**Lesson: AI as Differentiator, Not Just Cost Cutter**

Duolingo used AI to create a feature that was previously *impossible* — personalized, unlimited, realistic conversation practice at scale. The lesson: AI doesn't just reduce costs, it enables entirely new business models.

---

**Case Study 4: Smaller Business Success — Local Real Estate Agency**

**Background:**
- 3-person real estate agency in Texas
- No budget for full-time receptionist/scheduler
- Used Python + OpenAI API to build a chatbot in one weekend

```python
# What they built:
small_biz_chatbot = {
    "features": [
        "Answer common questions about listings",
        "Schedule property viewings (via Calendly integration)",
        "Qualify leads with 5 key questions",
        "Send follow-up email with property info",
        "Route hot leads to agents immediately via SMS"
    ],
    "cost_to_build": "1 weekend + $200 in developer time",
    "monthly_running_cost": "$15-30 in API costs",
    "monthly_value": {
        "hours_saved": 20,
        "lead_response_time": "24 hours → 30 seconds",
        "leads_qualified_per_month": "+45%",
        "deals_attributed_to_bot": "3 deals in first quarter"
    }
}

# 3 deals in real estate = potentially $15,000-30,000 in commission
# vs. $200 build cost and $30/month running cost
# ROI: ~10,000%+
```

**Key Takeaway:** You don't need a large budget or team to benefit from chatbots. A well-designed bot solving a specific problem can be transformative for small businesses.

---

## Slide 23: Best Practices Cheat Sheet

### The Rules of Building Great Chatbots

**Design Best Practices:**

```python
chatbot_best_practices = {
    "System Prompt": [
        "Be specific about persona, abilities, and limitations",
        "Include example responses for key scenarios",
        "Set clear scope boundaries — what it can and can't do",
        "Test your system prompt extensively before launch"
    ],
    "User Experience": [
        "Stream responses — don't make users wait",
        "Provide suggested questions to get users started",
        "Always offer a human escalation path",
        "Acknowledge when you can't help — don't fake it",
        "Keep responses concise — 2-4 sentences unless asked for more"
    ],
    "Technical": [
        "Cache common questions to reduce cost and latency",
        "Use the cheapest model that meets quality bar",
        "Implement retry logic and graceful error handling",
        "Rate limit to prevent abuse",
        "Monitor token usage to control costs"
    ],
    "Safety": [
        "Always validate user input before sending to LLM",
        "Use OpenAI's moderation API for free safety checking",
        "Have explicit human escalation triggers",
        "Never let the bot handle irreversible actions without confirmation",
        "Test extensively with adversarial inputs"
    ],
    "Business": [
        "Define clear KPIs before launching",
        "A/B test different prompts and models",
        "Collect user feedback (thumbs up/down at minimum)",
        "Review a sample of conversations weekly",
        "Plan for continuous improvement — it's never 'done'"
    ]
}

for category, practices in chatbot_best_practices.items():
    print(f"\n{category}:")
    for p in practices:
        print(f"  ✅ {p}")
```

**Common Mistakes to Avoid:**

```python
common_mistakes = [
    {
        "mistake": "No scope boundaries in system prompt",
        "consequence": "Bot answers anything — medical, legal, political — creates liability",
        "fix": "Explicit scope: 'You ONLY help with [X]. For other topics, direct to [Y].'"
    },
    {
        "mistake": "Treating AI responses as always correct",
        "consequence": "Hallucinated facts reach customers, trust damage",
        "fix": "RAG for factual domains, test extensively, add disclaimers"
    },
    {
        "mistake": "No human escalation path",
        "consequence": "Frustrated users with no resolution, churn",
        "fix": "Always: 'For complex issues, connect with a human: [link/number]'"
    },
    {
        "mistake": "Same model for everything",
        "consequence": "Overpaying for simple questions",
        "fix": "Route simple FAQs to gpt-4o-mini, complex analysis to gpt-4o"
    },
    {
        "mistake": "Launching without testing edge cases",
        "consequence": "Bot embarrasses company in early interactions",
        "fix": "Red team your bot: try to make it say harmful/wrong things before launch"
    },
    {
        "mistake": "No conversation logging/review",
        "consequence": "Can't improve, don't know what's failing",
        "fix": "Log anonymized conversations, review 50 per week, improve prompts"
    }
]

for m in common_mistakes:
    print(f"\n❌ Mistake: {m['mistake']}")
    print(f"   Consequence: {m['consequence']}")
    print(f"   ✅ Fix: {m['fix']}")
```

---

## Slide 24: Hands-On Lab Exercises

### Build Your Own Chatbot!

**Exercise 1: Your First API Call (5 minutes)**

```python
# Run this in a Jupyter notebook or Python file
# Make sure to: pip install openai

import openai
import os

# Set your API key
client = openai.OpenAI(api_key="YOUR_API_KEY_HERE")

# Make your first call!
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me one fascinating fact about Texas."}
    ],
    max_tokens=100
)

print(response.choices[0].message.content)
print(f"\nTokens used: {response.usage.total_tokens}")
print(f"Est. cost: ${response.usage.total_tokens * 0.00000015:.6f}")
```

---

**Exercise 2: Build a Custom Persona Chatbot (15 minutes)**

```python
# Create a chatbot with a specific business persona
# Choose ONE of these personas:

personas = {
    "A": {
        "name": "Chef Marco",
        "role": "Italian restaurant assistant",
        "tasks": ["Answer menu questions", "Handle reservations", "Dietary restrictions"],
        "restrictions": "Only discuss food, restaurant, and dining. Don't discuss other topics."
    },
    "B": {
        "name": "Professor Quinn",
        "role": "UTD academic advisor for Business Analytics",
        "tasks": ["Course selection", "Degree requirements", "Career advice"],
        "restrictions": "Only advise on BUAN program. Refer to official advisors for enrollment."
    },
    "C": {
        "name": "Alex",
        "role": "Tech startup recruiting assistant",
        "tasks": ["Screen candidates", "Answer job questions", "Schedule interviews"],
        "restrictions": "Never make hiring decisions. Always escalate to human recruiter."
    }
}

# TODO: Pick a persona and write the system prompt
# Then run a 5-turn conversation with it
chosen_persona = personas["A"]  # Change to B or C

YOUR_SYSTEM_PROMPT = f"""
You are {chosen_persona['name']}, {chosen_persona['role']}.

You help with: {', '.join(chosen_persona['tasks'])}

Important: {chosen_persona['restrictions']}

Be friendly, professional, and concise in your responses.
"""

# Implement the Chatbot class and run a conversation!
```

---

**Exercise 3: Add RAG to Your Chatbot (20 minutes)**

```python
# Create a knowledge base from a text you care about
# Then make a bot that answers questions from it

# Option A: Use your own FAQ document
# Option B: Use this sample policy document

sample_policy = """
UTD BUAN Program - Student Handbook 2026

COURSE REQUIREMENTS:
The BUAN program requires 36 credit hours. Core courses include:
BUAN 6310 (Business Analytics), BUAN 6320 (Data Mining), BUAN 6330 (Machine Learning).
Students must maintain a 3.0 GPA to remain in good standing.

CAPSTONE PROJECT:
All students complete a capstone project in their final semester.
Projects must be approved by the program director by week 5 of the semester.
Groups of 3-5 students are required. Individual projects are not accepted.

INTERNSHIP POLICY:
Students may earn up to 3 credit hours for internships (BUAN 6V98).
Internships must be pre-approved. Students work minimum 20 hours/week.
An internship report and supervisor evaluation are required for credit.

GRADUATION:
Students must apply for graduation one semester before expected completion.
All courses must be completed with B- or better.
Transfer credits are limited to 6 hours and require department approval.
"""

# Step 1: Chunk the document
chunks = chunk_text(sample_policy, chunk_size=200, overlap=30)

# Step 2: Create a simple vector store (use the class from Slide 12)
store = SimpleVectorStore()
store.add_documents(chunks)

# Step 3: Create RAG chatbot
rag_bot = RAGChatbot(
    system_prompt="You are a helpful academic advisor for the UTD BUAN program. "
                  "Answer questions based on the student handbook. "
                  "If the answer isn't in the handbook, say so.",
    vector_store=store
)

# Step 4: Test it!
test_questions = [
    "How many credit hours do I need to graduate?",
    "Can I do my capstone project alone?",
    "What GPA do I need?",
    "Tell me about internship opportunities."  # Not in the handbook
]

for q in test_questions:
    print(f"Q: {q}")
    print(f"A: {rag_bot.chat(q)}\n")
```

---

**Exercise 4 (Bonus): Deploy with Gradio (10 minutes)**

```python
# Take your chatbot from Exercise 2 or 3 and deploy it with Gradio!
# pip install gradio

import gradio as gr
from openai import OpenAI

client = OpenAI()

# Use your system prompt from Exercise 2
SYSTEM = YOUR_SYSTEM_PROMPT  # from above

def chat(message, history):
    messages = [{"role": "system", "content": SYSTEM}]
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, max_tokens=300)
    return response.choices[0].message.content

# Launch!
gr.ChatInterface(
    fn=chat,
    title="My Custom Chatbot",
    examples=["Hello!", "What can you help me with?"]
).launch(share=True)  # share=True gives you a public URL to share!
```

---

## Slide 25: Real Business Application Gallery

### What You Can Build With Today's Skills

**Complete Business Chatbot Implementations:**

```
Application 1: Restaurant Chain FAQ Bot
- Knowledge: Menu, hours, locations, allergens, reservations
- Channels: Website widget + WhatsApp integration
- Value: 60% reduction in phone calls during peak hours

Application 2: HR Onboarding Assistant
- Knowledge: Company handbook, IT setup guide, benefits docs
- Channel: Microsoft Teams bot
- Value: New hire self-service; HR freed from repetitive questions

Application 3: E-commerce Product Advisor
- Knowledge: Full product catalog with specs and reviews
- Features: Recommends based on user requirements
- Channel: Website chatbot with Shopify integration
- Value: +15% conversion rate on product pages

Application 4: Internal IT Help Desk Bot
- Knowledge: Company IT policies, common fixes, software guides
- Features: Can open tickets, check ticket status
- Channel: Slack
- Value: 40% of IT tickets resolved without human touch

Application 5: Legal Document Summarizer
- Knowledge: None (processes user-uploaded documents)
- Features: Summarize, extract clauses, flag risks
- Channel: Internal web app
- Value: Partners review 3x more contracts per day
```

**The Stack for All of These:**

```python
# The core tech behind every example above
core_stack = {
    "LLM": "OpenAI GPT-4o-mini or Anthropic Claude Haiku (cost-effective)",
    "Knowledge Base": "ChromaDB (local) or Pinecone (cloud) for RAG",
    "UI": "Gradio (quick demo) or React (production)",
    "Backend": "FastAPI or Flask REST API",
    "Hosting": "Hugging Face Spaces (free) or AWS/GCP ($20-100/month)",
    "Logging": "Python logging → CloudWatch or Datadog",
    "Auth": "JWT tokens for user sessions"
}
```

---

## Slide 26: Industry Trends & What's Next

### The Future of Chatbots & Text Generation

**Trend 1: Multi-Modal Chatbots (Vision + Voice + Text)**

```python
# Today's chatbots are already multi-modal
# GPT-4o and Claude can see images AND generate text

# Example: Customer shows photo of broken product
def handle_image_complaint(image_path: str, user_message: str) -> str:
    """Process a complaint with an attached image."""
    import base64

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": user_message},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }]
    )
    return response.choices[0].message.content

# Near future: Voice → Text → LLM → Text → Voice
# Already happening with: OpenAI Realtime API, ElevenLabs
```

**Trend 2: Agentic Chatbots (Act, Don't Just Talk)**

```
Traditional chatbot: Answers questions
Agentic chatbot: Takes actions

Example: "Book me a flight to NYC for next Tuesday under $400"
→ Searches flights, compares prices, books the best option,
  confirms with you, sends calendar invite — all automatically

Tools: OpenAI Assistants API, LangChain Agents, AutoGen
```

**Trend 3: Personalized Long-Term Memory**

```python
# Future chatbots will remember you across sessions
# OpenAI has released memory features in ChatGPT Plus

# Example: Memory across sessions
user_profile = {
    "name": "Maria Chen",
    "preferences": "prefers email over phone, tech-savvy",
    "past_issues": ["shipping delay in Dec 2025", "returned laptop Jan 2026"],
    "loyalty_tier": "Gold",
    "preferred_language": "English"
}

# System prompt enriched with user history
personalized_system = f"""
You are Aria, customer service for TechShop.

CUSTOMER PROFILE:
Name: {user_profile['name']} ({user_profile['loyalty_tier']} member)
Notes: {user_profile['preferences']}
Recent issues: {', '.join(user_profile['past_issues'])}

Use this context to provide personalized, efficient service.
Reference their history when relevant.
"""
```

**Where This Is Heading:**

```
2024: Chatbots answer questions
2025: Chatbots take simple actions (book, cancel, update)
2026: Chatbots coordinate multi-step workflows
2027+: AI agents run business processes autonomously,
       with human oversight at key decision points
```

---

## Slide 27: Key Takeaways, Resources & Homework

### Summary & Next Steps

**🎯 Key Takeaways from Class 7:**

**1. Text Generation is Token-by-Token Prediction**
- Temperature, top-p, max_tokens control the output
- Understand these parameters to get the behavior you want

**2. The APIs are Simple — The Hard Part is the Prompt**
- OpenAI and Anthropic APIs are easy to call
- The system prompt design determines 80% of chatbot quality

**3. Manage Conversation History Explicitly**
- LLMs are stateless — you must send full history every call
- Compress old history to manage costs

**4. RAG Makes Chatbots Business-Useful**
- Without RAG: generic answers
- With RAG: answers grounded in YOUR data

**5. Measure What Matters**
- Resolution rate, CSAT, escalation rate, cost/conversation
- Use LLM-as-judge for automated quality evaluation

**6. Deploy Fast with Gradio / Hugging Face Spaces**
- From zero to shareable web app in 30 minutes
- Build → Demo → Iterate

---

**📚 Resources:**

**APIs & Documentation:**
- OpenAI API Docs: platform.openai.com/docs
- Anthropic Claude Docs: docs.anthropic.com
- OpenAI Cookbook (examples): cookbook.openai.com

**RAG & Vector Databases:**
- ChromaDB: docs.trychroma.com
- Pinecone: docs.pinecone.io
- LangChain (RAG framework): python.langchain.com

**Deployment:**
- Gradio: gradio.app
- Hugging Face Spaces: huggingface.co/spaces
- FastAPI: fastapi.tiangolo.com

**Learning:**
- "Building LLM Powered Applications" — DeepLearning.AI (free course)
- "ChatGPT Prompt Engineering for Developers" — DeepLearning.AI
- OpenAI's Prompt Engineering Guide (platform.openai.com/docs/guides/prompt-engineering)

---

**🙏 Thank You!**

**Remember:**
> "A chatbot that solves one problem well is infinitely more valuable than one that tries to do everything."

**Focus. Build. Measure. Improve.**

---

**End of Week 7: Text Generation, Chatbots**
