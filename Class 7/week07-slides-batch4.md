# Week 7: Text Generation, Chatbots - Slides Batch 4 (Slides 16-20)

**Course:** BUAN 6v99.SW2 - Generative AI for Business
**Topic:** Business Use Cases, Platforms & Evaluation

---

## Slide 16: Business Chatbot Use Cases

### Where Chatbots Create Real Value

**The Chatbot Value Matrix:**

```
                    HIGH COMPLEXITY
                         │
           Medical        │     Legal Document
           Triage Bot     │     Analysis Bot
    ───────────────────────────────────────────
    LOW VOLUME            │               HIGH VOLUME
           Sales Demo     │     Customer Service Bot
           Scheduling     │     HR FAQ Bot
                         │
                    LOW COMPLEXITY
```

**Use Case 1: Customer Service Automation**

**Problem:** 70% of customer service questions are repetitive

```python
# Typical customer service questions the bot can handle:
faq_topics = [
    "Order status and tracking",
    "Return and refund policy",
    "Product specifications",
    "Shipping times and costs",
    "Account password reset",
    "Billing and payment questions",
    "Store hours and locations",
    "Warranty information"
]

# Business impact calculation
def customer_service_roi(
    monthly_tickets: int,
    automation_rate: float,      # % of tickets bot can handle
    human_cost_per_ticket: float, # $
    bot_cost_per_message: float   # $
) -> dict:
    """Calculate ROI for customer service chatbot."""
    automated_tickets = monthly_tickets * automation_rate
    human_handled = monthly_tickets * (1 - automation_rate)

    # Costs
    original_cost = monthly_tickets * human_cost_per_ticket
    new_human_cost = human_handled * human_cost_per_ticket
    bot_cost = automated_tickets * bot_cost_per_message * 5  # avg 5 messages/ticket

    total_new_cost = new_human_cost + bot_cost
    monthly_savings = original_cost - total_new_cost
    annual_savings = monthly_savings * 12

    return {
        "monthly_tickets": monthly_tickets,
        "automated_tickets": int(automated_tickets),
        "human_handled": int(human_handled),
        "original_monthly_cost": original_cost,
        "new_monthly_cost": round(total_new_cost, 2),
        "monthly_savings": round(monthly_savings, 2),
        "annual_savings": round(annual_savings, 2),
        "roi_percent": round((monthly_savings / total_new_cost) * 100, 1)
    }

result = customer_service_roi(
    monthly_tickets=5000,
    automation_rate=0.65,         # 65% of tickets automated
    human_cost_per_ticket=8.50,   # $8.50 per human-handled ticket
    bot_cost_per_message=0.0      # $0.00 per API call (Ollama — local/free)
)

print("Customer Service Chatbot ROI")
print("=" * 40)
for key, value in result.items():
    if isinstance(value, float) and value > 100:
        print(f"{key}: ${value:,.2f}")
    elif isinstance(value, float):
        print(f"{key}: {value}%")
    else:
        print(f"{key}: {value:,}")

# Output:
# monthly_tickets: 5,000
# automated_tickets: 3,250
# human_handled: 1,750
# original_monthly_cost: $42,500.00
# new_monthly_cost: $16,391.25
# monthly_savings: $26,108.75
# annual_savings: $313,305.00
# roi_percent: 159.3%
```

**Use Case 2: Internal Knowledge Base Bot (HR / IT)**

```python
# Employee FAQ Bot - answers from company handbook, IT policies, HR docs
hr_bot_system = """
You are HRBot, an internal assistant for Acme Corporation employees.

You help with:
- PTO and leave policies
- Benefits enrollment questions
- IT support procedures
- Onboarding information
- Company policies and procedures

You CANNOT:
- Process actual leave requests (direct to HR portal)
- Access personal employee records
- Make promises about compensation or benefits changes
- Provide legal or tax advice

Always end sensitive topics with: "For official decisions, please contact HR directly at hr@acme.com"
"""

# Impact: Reduces HR ticket volume by 40-60%
# Available 24/7 vs. HR's 9-5 hours
# Consistent answers vs. varying human responses
```

**Use Case 3: Sales & Lead Qualification Bot**

```python
lead_qualification_system = """
You are Alex, a sales development assistant for CloudPro Software.

Your goal is to:
1. Learn about the prospect's business challenges
2. Qualify them against our ICP (Ideal Customer Profile):
   - Company size: 50-500 employees
   - Budget: $10K+/year
   - Current pain: manual data processes, outdated software
3. Schedule a demo if qualified
4. Politely end the conversation if not qualified

Qualifying questions to ask (one at a time):
- "What industry is your company in?"
- "How many employees do you have?"
- "What's the biggest challenge with your current data processes?"
- "Do you have a budget allocated for software this quarter?"

If qualified: "Great! I'd love to connect you with our solutions team. Could you share
your email so I can send a calendar link for a 30-minute demo?"

If not qualified: "Thank you for your time! Our product may not be the right fit right
now, but I'd be happy to share some free resources on [relevant topic]."
"""

# Business impact:
# - Qualifies leads 24/7 (never miss a website visitor)
# - Consistent qualification process
# - Frees sales reps for closing, not prospecting
# - Typical result: 3x more demos booked per SDR headcount
```

**Use Case 4: Document Analysis & Summarization**

```python
def analyze_document(document_text: str, analysis_type: str = "summary") -> str:
    """
    Analyze a business document using an LLM.

    analysis_type options:
    - "summary": Executive summary
    - "risks": Risk identification
    - "action_items": Extract action items
    - "sentiment": Stakeholder sentiment
    - "comparison": Compare with requirements
    """
    from openai import OpenAI
    # Free: Ollama runs locally — install at ollama.com, then: ollama pull mistral
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

    analysis_prompts = {
        "summary": "Provide a 3-bullet executive summary of this document. "
                   "Focus on key decisions, outcomes, and next steps.",
        "risks": "Identify the top 5 business risks mentioned or implied in this document. "
                 "Rate each as High/Medium/Low and explain briefly.",
        "action_items": "Extract all action items from this document. "
                        "Format as a checklist with owner (if mentioned) and deadline (if mentioned).",
        "sentiment": "Analyze the tone and sentiment of this document. "
                     "Who are the key stakeholders and what is their apparent position/concern?"
    }

    prompt = analysis_prompts.get(analysis_type, analysis_prompts["summary"])

    response = client.chat.completions.create(
        model="mistral",  # Use a higher-quality local model for document analysis
        messages=[
            {"role": "system", "content": "You are an expert business analyst. "
             "Provide precise, actionable analysis. "
             "Be specific and cite evidence from the document when possible."},
            {"role": "user", "content": f"{prompt}\n\nDOCUMENT:\n{document_text}"}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content

# Example: Analyze a contract
# with open("vendor_contract.pdf", "rb") as f:
#     text = extract_pdf_text(f)
# risks = analyze_document(text, analysis_type="risks")
# print(risks)
```

**Use Case 5: Personalized Content Generation at Scale**

```python
import pandas as pd
from openai import OpenAI

# Free: Ollama runs locally — install at ollama.com, then: ollama pull llama3.2
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

def generate_personalized_emails(customer_data: list[dict]) -> list[dict]:
    """
    Generate personalized marketing emails for each customer.
    Much better than "Dear Customer" templates!

    Args:
        customer_data: List of customer dicts with: name, company, industry,
                      purchase_history, last_interaction

    Returns:
        List of personalized emails with subject and body
    """
    emails = []

    for customer in customer_data:
        prompt = f"""
Write a personalized re-engagement email for this customer:

Customer: {customer['name']}
Company: {customer['company']}
Industry: {customer['industry']}
Last purchase: {customer.get('last_purchase', 'Unknown')}
Days since last interaction: {customer.get('days_inactive', 90)}
Previous products: {', '.join(customer.get('products', []))}

Requirements:
- Friendly but professional tone
- Reference their specific industry
- Mention their previous purchase if available
- Include one relevant tip or insight for their industry
- End with a soft call-to-action
- Subject line + email body
- Keep to 150 words max
"""
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "You are an expert email marketer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=300
        )
        emails.append({
            "customer": customer['name'],
            "email_content": response.choices[0].message.content
        })

    return emails

# Generate 100 personalized emails in under 2 minutes!
```

---

## Slide 17: Enterprise Chatbot Platforms

### When to Build vs. Buy

**The Build vs. Buy Decision Framework:**

```python
def build_or_buy(requirements: dict) -> str:
    """
    Simple framework for deciding build vs. buy for chatbots.
    """
    score = 0

    # Factors that favor BUILDING custom
    if requirements.get('needs_deep_integration', False): score += 2
    if requirements.get('high_security_requirements', False): score += 2
    if requirements.get('unique_workflow', False): score += 2
    if requirements.get('large_dev_team', False): score += 1
    if requirements.get('budget', 0) > 50000: score += 1

    # Factors that favor BUYING a platform
    if requirements.get('fast_time_to_market', False): score -= 2
    if requirements.get('small_team', False): score -= 2
    if requirements.get('standard_use_case', False): score -= 2
    if requirements.get('no_coding_resources', False): score -= 3

    if score > 2:
        return "BUILD: Custom solution gives you the flexibility you need"
    elif score < -2:
        return "BUY: A platform will get you live faster and cheaper"
    else:
        return "HYBRID: Use a platform with custom API integrations"
```

**Major Chatbot Platforms:**

**1. Dialogflow CX (Google)**

```
Strengths:
✅ Visual conversation flow builder
✅ Multi-language support (30+ languages)
✅ Telephony integration (voice bots)
✅ Deep Google Cloud integration
✅ Free tier for development

Weaknesses:
❌ Complex to set up for beginners
❌ Can get expensive at scale
❌ Less flexible than custom solutions

Best for: Enterprise contact centers, phone bots, Google ecosystem users

Pricing: Free tier → $0.007/text request, $0.06/voice minute
```

**2. Amazon Lex**

```
Strengths:
✅ Same technology as Alexa
✅ Deep AWS integration (Lambda, Connect, S3)
✅ Voice + text
✅ Built-in slot filling for structured data collection
✅ Multi-turn dialog management

Weaknesses:
❌ Requires AWS expertise
❌ Less capable LLM vs. GPT-4/Claude
❌ Limited to AWS ecosystem

Best for: AWS customers, voice-based chatbots, contact centers

Pricing: $0.004/text request, $0.008/voice request
```

**3. Microsoft Azure Bot Service + Copilot Studio**

```
Strengths:
✅ Integrates with Microsoft 365 (Teams, Outlook, SharePoint)
✅ Copilot Studio: low-code bot builder
✅ Enterprise security and compliance
✅ Power Platform integration

Weaknesses:
❌ Best value only if already in Microsoft ecosystem
❌ Copilot Studio can feel limited for complex needs

Best for: Microsoft shops, internal employee bots, Teams integration

Pricing: Free base + $200/month for Copilot Studio Pro
```

**4. Custom Build (OpenAI/Anthropic API + Python)**

```
Strengths:
✅ Maximum flexibility and control
✅ Latest models (GPT-4o, Claude Sonnet)
✅ Best quality AI responses
✅ Full customization of logic and UX
✅ No vendor lock-in

Weaknesses:
❌ Requires development resources
❌ You handle infrastructure, monitoring, scaling
❌ More setup time

Best for: Unique use cases, developers, high-quality AI requirement

Pricing: Pay only for API tokens used (~$0.001-0.01 per conversation)
```

**Platform Comparison Matrix:**

| Platform | Setup Difficulty | AI / NLU Quality | Starting Cost | Voice Support | Multi-Language | Key Integrations | Scalability | Best For |
|----------|-----------------|------------------|---------------|---------------|----------------|------------------|-------------|----------|
| **Dialogflow CX** *(Google)* | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Good | Free tier → $0.007/text req | ✅ Yes (telephony) | ✅ 30+ languages | Google Cloud, BigQuery, Twilio | ⭐⭐⭐⭐⭐ Enterprise | Contact centers, voice bots, Google ecosystem |
| **Amazon Lex** *(AWS)* | ⭐⭐ Hard | ⭐⭐⭐ Good | $0.004/text req, $0.008/voice | ✅ Yes (Connect) | ✅ Yes | Lambda, S3, Connect, Kendra | ⭐⭐⭐⭐⭐ Enterprise | AWS customers, contact centers, Alexa-style bots |
| **Azure Copilot Studio** *(Microsoft)* | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Good (Azure OpenAI) | Free base + $200/mo Pro | ✅ Teams voice | ✅ Yes | M365, Teams, SharePoint, Power Automate | ⭐⭐⭐⭐⭐ Enterprise | Microsoft shops, internal HR/IT bots, Teams |
| **Custom Build** *(OpenAI / Anthropic API)* | ⭐⭐ Hard | ⭐⭐⭐⭐⭐ Excellent | ~$0.001–0.01/conversation | 🔧 Custom (Whisper + TTS) | ✅ Model-dependent | Any API, Slack, CRM, DB | ⭐⭐⭐ DIY infra | Unique use cases, developers, max quality/control |
| **Ollama** *(Local / Free)* | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ Good (llama3.2, mistral) | **$0 — 100% free** | ❌ No | ✅ Model-dependent | Any local app or API | ⭐⭐ Single machine | Privacy-first, dev/testing, no-cost prototyping |
| **Intercom Fin** | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Good (OpenAI-powered) | $74+/mo + $0.99/resolution | ❌ No | ✅ Limited | Stripe, Salesforce, HubSpot, Zendesk | ⭐⭐⭐⭐ High | SaaS customer support, sales chat, e-commerce |
| **Tidio** | ⭐⭐⭐⭐⭐ Very Easy | ⭐⭐⭐ Basic | Free → $29+/mo | ❌ No | ✅ Limited | Shopify, WooCommerce, WordPress | ⭐⭐⭐ Medium | Small businesses, e-commerce, quick deployment |
| **Zendesk AI** | ⭐⭐⭐⭐ Easy | ⭐⭐⭐⭐ Good | $55+/agent/mo | ❌ No | ✅ Yes | Salesforce, JIRA, Slack, email | ⭐⭐⭐⭐⭐ Enterprise | Customer support teams, ticketing, help desks |

**How to Choose:**

| Your Situation | Recommended Path |
|----------------|-----------------|
| Learning / prototyping | **Ollama** (free, local, no limits) |
| Small business, no developers | **Tidio** or **Intercom** |
| Microsoft 365 company | **Azure Copilot Studio** |
| AWS infrastructure already | **Amazon Lex** |
| High-volume phone support | **Dialogflow CX** |
| Need best AI quality + full control | **Custom build** (OpenAI / Anthropic) |
| Customer support SaaS team | **Zendesk AI** or **Intercom Fin** |

---

## Slide 18: Chatbot Cost Analysis

### Understanding and Optimizing API Costs

**Token-Based Pricing Explained:**

```python
def calculate_chatbot_cost(
    monthly_conversations: int,
    avg_turns_per_conversation: int,
    avg_user_message_tokens: int,
    avg_bot_response_tokens: int,
    system_prompt_tokens: int,
    model: str = "llama3.2"
) -> dict:
    """
    Calculate monthly API costs for a chatbot deployment.
    """
    # Pricing per 1M tokens (approximate, March 2026)
    pricing = {
        "llama3.2": {"input": 0.0, "output": 0.0},        # Ollama — free/local
        "mistral": {"input": 0.0, "output": 0.0},          # Ollama — free/local
        "gpt-4o": {"input": 2.50, "output": 10.00},        # OpenAI — paid
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},    # OpenAI — paid
        "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},  # Anthropic — paid
        "claude-haiku-4-5": {"input": 0.80, "output": 4.00}     # Anthropic — paid
    }

    rates = pricing.get(model, pricing["llama3.2"])

    # Each API call includes: system prompt + full history + new message
    # History grows with each turn in conversation
    avg_context_growth = avg_turns_per_conversation * (avg_user_message_tokens + avg_bot_response_tokens) / 2

    # Calculate tokens per conversation
    # Input = system prompt + growing history + user message (each turn)
    # Output = bot response (each turn)
    input_tokens_per_conversation = (
        system_prompt_tokens * avg_turns_per_conversation +  # System repeated each call
        avg_context_growth * avg_turns_per_conversation +    # Growing context
        avg_user_message_tokens * avg_turns_per_conversation # User messages
    )
    output_tokens_per_conversation = avg_bot_response_tokens * avg_turns_per_conversation

    # Monthly totals
    monthly_input_tokens = input_tokens_per_conversation * monthly_conversations
    monthly_output_tokens = output_tokens_per_conversation * monthly_conversations

    input_cost = (monthly_input_tokens / 1_000_000) * rates["input"]
    output_cost = (monthly_output_tokens / 1_000_000) * rates["output"]
    total_cost = input_cost + output_cost

    return {
        "model": model,
        "monthly_conversations": monthly_conversations,
        "monthly_input_tokens": int(monthly_input_tokens),
        "monthly_output_tokens": int(monthly_output_tokens),
        "input_cost_usd": round(input_cost, 2),
        "output_cost_usd": round(output_cost, 2),
        "total_monthly_cost_usd": round(total_cost, 2),
        "cost_per_conversation_cents": round((total_cost / monthly_conversations) * 100, 2)
    }

# Example: Customer service bot
costs = calculate_chatbot_cost(
    monthly_conversations=10_000,
    avg_turns_per_conversation=6,
    avg_user_message_tokens=50,
    avg_bot_response_tokens=150,
    system_prompt_tokens=300,
    model="llama3.2"  # Free with Ollama; swap for "gpt-4o-mini" to compare cloud pricing
)

print("=== Monthly Cost Analysis ===")
for k, v in costs.items():
    if "_usd" in k:
        print(f"{k}: ${v:,.2f}")
    elif "_tokens" in k:
        print(f"{k}: {v:,}")
    elif "_cents" in k:
        print(f"{k}: {v}¢")
    else:
        print(f"{k}: {v}")

# Output (approximate):
# model: gpt-4o-mini
# monthly_conversations: 10,000
# monthly_input_tokens: 42,000,000
# monthly_output_tokens: 9,000,000
# input_cost_usd: $6.30
# output_cost_usd: $5.40
# total_monthly_cost_usd: $11.70
# cost_per_conversation_cents: 0.12¢  ← Less than a penny per conversation!
```

**Cost Optimization Strategies:**

```python
class CostOptimizedChatbot:
    """
    Chatbot with built-in optimization strategies.
    Uses Ollama locally — cost is always $0, but strategies still apply for latency.
    """

    def __init__(self, system_prompt: str):
        from openai import OpenAI
        self.client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        self.system = system_prompt
        self.history = []
        self.total_cost = 0.0

    def chat(self, user_message: str,
             use_cache: bool = True,
             cache: dict = None) -> str:
        """
        Chat with cost optimization:
        1. Semantic cache for repeated questions
        2. Model routing (cheap for simple, expensive for complex)
        3. History compression to reduce context tokens
        """
        # Strategy 1: Semantic cache
        # Check if we've answered a similar question before
        if use_cache and cache is not None:
            for cached_q, cached_a in cache.items():
                # Simple check — in production use embedding similarity
                if user_message.lower().strip() == cached_q.lower().strip():
                    return f"[CACHED] {cached_a}"

        # Strategy 2: Model routing
        # Simple questions → cheap model, complex → expensive
        model = self._select_model(user_message)

        # Strategy 3: Compress history if too long
        if len(self.history) > 10:
            self.history = self._compress_history(self.history)

        messages = [{"role": "system", "content": self.system}] + self.history
        messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=512  # Cap response length
        )

        answer = response.choices[0].message.content
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})

        # Track cost — $0 for Ollama, but logging token usage is still useful for
        # latency optimization (fewer tokens = faster responses)
        usage = response.usage
        self.total_cost = 0.0  # Ollama is free/local

        return answer

    def _select_model(self, message: str) -> str:
        """Route to appropriate model based on task complexity."""
        # Simple heuristic: use cheap model for short, simple questions
        complex_keywords = ["analyze", "compare", "explain why", "evaluate",
                           "strategy", "comprehensive", "detailed report"]
        is_complex = (
            len(message) > 200 or
            any(kw in message.lower() for kw in complex_keywords)
        )
        return "mistral" if is_complex else "llama3.2"

    def _compress_history(self, history: list) -> list:
        """Summarize old history to save tokens."""
        old_messages = history[:-4]  # All but last 4 messages
        recent = history[-4:]        # Keep last 4 verbatim

        text = "\n".join([f"{m['role']}: {m['content']}" for m in old_messages])
        summary = self.client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "system", "content": "Summarize this conversation in 2 sentences."},
                {"role": "user", "content": text}
            ],
            max_tokens=100
        ).choices[0].message.content

        return [{"role": "user", "content": f"[Summary: {summary}]"}] + recent
```

---

## Slide 19: Evaluating Chatbot Quality

### How to Measure What Matters

**The Four Pillars of Chatbot Evaluation:**

```
1. ACCURACY    - Does it answer correctly?
2. SAFETY      - Does it stay appropriate?
3. HELPFULNESS - Does it actually help the user?
4. EFFICIENCY  - Does it do it cost-effectively?
```

**Automated Evaluation with LLM-as-Judge:**

```python
from openai import OpenAI
import json

# Free: Ollama runs locally — install at ollama.com, then: ollama pull mistral
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

def evaluate_chatbot_response(
    user_question: str,
    bot_response: str,
    ground_truth: str = None,
    context: str = None
) -> dict:
    """
    Use GPT-4 to evaluate a chatbot response.
    This is the "LLM-as-judge" pattern — very powerful!

    Returns scores for: accuracy, helpfulness, safety, clarity
    """
    evaluation_prompt = f"""
Evaluate this chatbot response on 4 dimensions. Return JSON only.

USER QUESTION: {user_question}

BOT RESPONSE: {bot_response}

{"CORRECT ANSWER: " + ground_truth if ground_truth else ""}
{"CONTEXT/POLICY: " + context[:500] if context else ""}

Score each dimension from 1-5 (5=excellent) and provide brief reasoning:
{{
  "accuracy": {{"score": int, "reasoning": str}},
  "helpfulness": {{"score": int, "reasoning": str}},
  "safety": {{"score": int, "reasoning": str}},
  "clarity": {{"score": int, "reasoning": str}},
  "overall_score": float,
  "improvement_suggestion": str
}}

Scoring guide:
- accuracy: Does it answer correctly? 1=wrong, 5=perfectly accurate
- helpfulness: Does it actually help? 1=useless, 5=exactly what user needed
- safety: Is it appropriate? 1=harmful, 5=perfectly safe
- clarity: Is it clear? 1=confusing, 5=crystal clear
"""

    response = client.chat.completions.create(
        model="mistral",  # Use higher-quality model for evaluation
        messages=[
            {"role": "system", "content": "You are an expert chatbot evaluator. Be critical and precise."},
            {"role": "user", "content": evaluation_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    result = json.loads(response.choices[0].message.content)
    result['overall_score'] = (
        result['accuracy']['score'] +
        result['helpfulness']['score'] +
        result['safety']['score'] +
        result['clarity']['score']
    ) / 4
    return result

# Evaluate a batch of responses
test_cases = [
    {
        "question": "What's your return policy?",
        "bot_response": "You can return items within 30 days.",
        "ground_truth": "Items can be returned within 45 days of purchase."
    },
    {
        "question": "My order hasn't arrived after 3 weeks!",
        "bot_response": "I'm sorry to hear that. Please provide your order number so I can look into this for you right away.",
        "ground_truth": None  # Human judgment needed
    }
]

for case in test_cases:
    result = evaluate_chatbot_response(**case)
    print(f"\nQuestion: {case['question'][:50]}...")
    print(f"Overall Score: {result['overall_score']:.1f}/5")
    print(f"Accuracy: {result['accuracy']['score']}/5 - {result['accuracy']['reasoning']}")
    print(f"Suggestion: {result['improvement_suggestion']}")
```

**Business Metrics Dashboard:**

```python
from datetime import datetime, timedelta
import random

class ChatbotMetricsDashboard:
    """Track and report on chatbot business metrics."""

    def __init__(self):
        self.conversations = []

    def log_conversation(self, conversation_id: str, resolved: bool,
                         escalated: bool, turns: int, duration_seconds: int,
                         user_rating: int = None, tokens_used: int = 0):
        """Log a completed conversation."""
        self.conversations.append({
            "id": conversation_id,
            "timestamp": datetime.now(),
            "resolved": resolved,
            "escalated": escalated,
            "turns": turns,
            "duration": duration_seconds,
            "rating": user_rating,
            "tokens": tokens_used
        })

    def get_summary(self, days: int = 30) -> dict:
        """Generate a KPI summary for the last N days."""
        cutoff = datetime.now() - timedelta(days=days)
        recent = [c for c in self.conversations if c['timestamp'] > cutoff]

        if not recent:
            return {"error": "No data in period"}

        total = len(recent)
        resolved = [c for c in recent if c['resolved']]
        escalated = [c for c in recent if c['escalated']]
        rated = [c for c in recent if c['rating'] is not None]

        return {
            "period_days": days,
            "total_conversations": total,
            "resolution_rate": f"{len(resolved)/total*100:.1f}%",
            "escalation_rate": f"{len(escalated)/total*100:.1f}%",
            "avg_turns_to_resolve": sum(c['turns'] for c in resolved) / len(resolved) if resolved else 0,
            "avg_duration_seconds": sum(c['duration'] for c in recent) / total,
            "csat_score": sum(c['rating'] for c in rated) / len(rated) if rated else None,
            "total_tokens": sum(c['tokens'] for c in recent),
            "avg_tokens_per_conv": int(sum(c['tokens'] for c in recent) / total),
            "estimated_cost_usd": 0.0  # Ollama runs locally — no cost
        }

# Example usage with simulated data
dashboard = ChatbotMetricsDashboard()

# Simulate 500 conversations
for i in range(500):
    dashboard.log_conversation(
        conversation_id=f"conv_{i}",
        resolved=random.random() > 0.25,          # 75% resolved
        escalated=random.random() < 0.15,          # 15% escalated
        turns=random.randint(2, 8),
        duration_seconds=random.randint(60, 600),
        user_rating=random.choice([3, 4, 4, 5, 5, 5]),
        tokens_used=random.randint(500, 3000)
    )

summary = dashboard.get_summary(days=30)
print("=== 30-Day Chatbot Performance ===")
for k, v in summary.items():
    print(f"{k}: {v}")
```

**Key Benchmarks to Track:**

| Metric | Poor | Good | Excellent |
|--------|------|------|-----------|
| **Resolution Rate** | < 50% | 65-75% | > 80% |
| **CSAT Score** (1-5) | < 3.0 | 3.5-4.0 | > 4.2 |
| **Escalation Rate** | > 30% | 15-25% | < 15% |
| **Avg Turns to Resolve** | > 8 | 4-6 | < 4 |
| **First Contact Resolution** | < 50% | 65% | > 75% |
| **Response Latency** | > 5s | 1-3s | < 1s |

---

## Slide 20: Integrating Chatbots into Business Systems

### Real-World Architecture Patterns

**Pattern 1: Simple Web Integration (Most Common)**

```python
from flask import Flask, request, jsonify
from openai import OpenAI
import os

app = Flask(__name__)
# Free: Ollama runs locally — install at ollama.com, then: ollama pull llama3.2
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

# In-memory session store (use Redis in production!)
sessions = {}

@app.route('/api/chat', methods=['POST'])
def chat():
    """
    REST API endpoint for chatbot.
    Request: {"session_id": str, "message": str}
    Response: {"response": str, "session_id": str}
    """
    data = request.json
    session_id = data.get("session_id", "default")
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Message is required"}), 400

    # Get or create session history
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]
    history.append({"role": "user", "content": user_message})

    # Build messages
    messages = [
        {"role": "system", "content": "You are a helpful customer service assistant."}
    ] + history[-20:]  # Last 20 messages

    # Call API
    response = client.chat.completions.create(
        model="llama3.2",
        messages=messages,
        max_tokens=512
    )

    bot_response = response.choices[0].message.content
    history.append({"role": "assistant", "content": bot_response})

    return jsonify({
        "response": bot_response,
        "session_id": session_id
    })

@app.route('/api/chat/reset', methods=['POST'])
def reset():
    """Clear a conversation session."""
    data = request.json
    session_id = data.get("session_id", "default")
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"status": "reset", "session_id": session_id})

# Add to your website with a simple JavaScript fetch:
"""
async function sendMessage(message) {
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            session_id: getUserSessionId(),
            message: message
        })
    });
    const data = await response.json();
    displayMessage(data.response);
}
"""

if __name__ == '__main__':
    app.run(debug=True)
```

**Pattern 2: Slack Bot Integration**

```python
# pip install slack-bolt
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from openai import OpenAI

app = App(token=os.environ.get("SLACK_BOT_TOKEN"),
          signing_secret=os.environ.get("SLACK_SIGNING_SECRET"))

# Free: Ollama runs locally — install at ollama.com, then: ollama pull llama3.2
client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
conversation_history = {}  # Store by user ID

@app.event("app_mention")
def handle_mention(event, say, client_slack):
    """Respond when the bot is @mentioned in a channel."""
    user_id = event['user']
    user_message = event['text'].split('>', 1)[-1].strip()  # Remove mention

    if user_id not in conversation_history:
        conversation_history[user_id] = []

    history = conversation_history[user_id]
    history.append({"role": "user", "content": user_message})

    # Limit to last 10 messages
    recent_history = history[-10:]

    response = client.chat.completions.create(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are an internal assistant for Acme Corp employees."},
        ] + recent_history
    )

    bot_response = response.choices[0].message.content
    history.append({"role": "assistant", "content": bot_response})

    say(f"<@{user_id}> {bot_response}")

# Run with: python app.py
# Set up at api.slack.com/apps → Event Subscriptions
```

**Architecture for Production Scale:**

```
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│   Frontend   │    │   API       │    │  OpenAI/Claude  │
│  (Website,  │───▶│  Gateway    │───▶│  API            │
│  Mobile,    │    │  (Flask/    │    │                 │
│  Slack)     │    │  FastAPI)   │    └─────────────────┘
└─────────────┘    └──────┬──────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
      ┌───────────┐ ┌──────────┐ ┌───────────┐
      │  Redis    │ │  Vector  │ │   DB      │
      │  Session  │ │  Store   │ │  (User    │
      │  Cache    │ │  (RAG)   │ │   Data)   │
      └───────────┘ └──────────┘ └───────────┘
```

**Production Checklist:**

```python
# ✅ Security
# - API keys in environment variables, never in code
# - Rate limiting per user (prevent abuse)
# - Input validation and sanitization
# - Output filtering for sensitive data

# ✅ Reliability
# - Retry logic with exponential backoff
# - Graceful degradation if API is down
# - Health check endpoint
# - Error logging and alerting

# ✅ Performance
# - Async API calls (don't block)
# - Response streaming for better UX
# - Cache frequent questions
# - CDN for static assets

# ✅ Observability
# - Log all conversations (with user consent)
# - Track token usage and costs
# - Monitor latency and errors
# - A/B test different prompts/models

# ✅ Compliance
# - Privacy policy for AI chat
# - Data retention policy
# - GDPR/CCPA compliance for EU/CA users
# - Clear disclosure that it's AI, not human
```

---

**End of Batch 4 (Slides 16-20)**

*Continue to Batch 5 for Ethics, Case Studies, Lab Exercises & Key Takeaways (Slides 21-27)*
