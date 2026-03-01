# Week 7: Text Generation, Chatbots - Slides Batch 2 (Slides 6-10)

**Course:** BUAN 6v99.SW2 - Generative AI for Business
**Topic:** OpenAI API, Anthropic API & Prompt Engineering

---

## Slide 6: OpenAI API Deep Dive

### Getting Started with the OpenAI API

**Setup:**

```bash
# Install the OpenAI Python SDK
pip install openai

# Set your API key (never hardcode in code!)
export OPENAI_API_KEY="sk-..."
```

```python
import openai
import os

# Initialize client (reads OPENAI_API_KEY from environment)
client = openai.OpenAI()

# Or explicitly pass the key
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
```

**Current Models (March 2026):**

| Model | Context | Best For | Cost (per 1M tokens) |
|-------|---------|----------|----------------------|
| **gpt-4o** | 128K | Best overall, vision | $2.50 in / $10 out |
| **gpt-4o-mini** | 128K | Fast, cost-effective | $0.15 in / $0.60 out |
| **o1** | 200K | Complex reasoning | $15 in / $60 out |
| **o3-mini** | 200K | Reasoning, affordable | $1.10 in / $4.40 out |

**Simple Text Generation:**

```python
from openai import OpenAI

client = OpenAI()

def generate_text(prompt: str, system: str = "You are a helpful assistant.",
                  model: str = "gpt-4o-mini", temperature: float = 0.7) -> str:
    """
    Generate text using the OpenAI API.

    Args:
        prompt: The user's message/question
        system: System prompt defining behavior
        model: OpenAI model to use
        temperature: Creativity level (0=factual, 1=creative)

    Returns:
        Generated text response
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=1024
    )
    return response.choices[0].message.content

# Example usage
result = generate_text(
    prompt="Summarize the key benefits of AI in healthcare in 3 bullet points.",
    system="You are a healthcare industry analyst. Be concise and evidence-based.",
    temperature=0.3
)
print(result)
```

**Streaming Responses (Real-Time Output):**

```python
def generate_text_streaming(prompt: str, system: str = "You are helpful."):
    """
    Generate text with streaming - shows output as it's generated.
    Like watching ChatGPT type in real time.
    """
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        stream=True  # Enable streaming
    )

    full_response = ""
    print("Assistant: ", end="", flush=True)

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            text_chunk = chunk.choices[0].delta.content
            print(text_chunk, end="", flush=True)
            full_response += text_chunk

    print()  # New line when done
    return full_response

# This prints tokens as they arrive — much better UX!
generate_text_streaming("Tell me a short story about a robot learning to paint.")
```

**Structured Output with JSON Mode:**

```python
import json

def extract_structured_data(text: str) -> dict:
    """
    Use OpenAI to extract structured data from unstructured text.
    Great for parsing emails, resumes, invoices, etc.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """Extract information and return ONLY valid JSON.
                Format: {"name": str, "email": str, "company": str, "request": str}
                If a field is missing, use null."""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        response_format={"type": "json_object"},  # Forces JSON output
        temperature=0
    )

    return json.loads(response.choices[0].message.content)

# Example
email_text = """
Hi, I'm Sarah Johnson from TechCorp. I'm writing to inquire about
your enterprise pricing. My email is sarah@techcorp.com.
"""

data = extract_structured_data(email_text)
print(data)
# Output:
# {
#   "name": "Sarah Johnson",
#   "email": "sarah@techcorp.com",
#   "company": "TechCorp",
#   "request": "inquiry about enterprise pricing"
# }
```

**Error Handling:**

```python
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
import time

def robust_generate(prompt: str, max_retries: int = 3) -> str:
    """Generate text with error handling and retry logic."""
    client = OpenAI()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500
            )
            return response.choices[0].message.content

        except RateLimitError:
            wait = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)

        except AuthenticationError:
            raise ValueError("Invalid API key. Check your OPENAI_API_KEY.")

        except APIError as e:
            print(f"API error: {e}. Attempt {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                raise

    return ""
```

---

## Slide 7: Anthropic API (Claude)

### Building with Claude

**Setup:**

```bash
pip install anthropic

export ANTHROPIC_API_KEY="sk-ant-..."
```

**Current Claude Models (March 2026):**

| Model | Context | Best For | Cost (per 1M tokens) |
|-------|---------|----------|----------------------|
| **claude-opus-4-6** | 200K | Most capable | $15 in / $75 out |
| **claude-sonnet-4-6** | 200K | Best balance | $3 in / $15 out |
| **claude-haiku-4-5** | 200K | Fast & cheap | $0.80 in / $4 out |

**Key Difference from OpenAI:**

Claude uses a slightly different API structure — the `system` prompt is a top-level parameter, not a message:

```python
import anthropic

client = anthropic.Anthropic()

def generate_with_claude(prompt: str, system: str = "You are a helpful assistant.",
                         model: str = "claude-haiku-4-5-20251001",
                         max_tokens: int = 1024) -> str:
    """
    Generate text using the Anthropic Claude API.

    Key difference: system prompt is a top-level parameter, not a message.
    """
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,          # System prompt is separate from messages
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

# Example
result = generate_with_claude(
    prompt="What are the top 3 risks of deploying AI in financial services?",
    system="You are a risk management expert with 20 years in financial services. "
           "Provide concise, actionable insights."
)
print(result)
```

**Claude Streaming:**

```python
def claude_streaming(prompt: str, system: str = "You are helpful."):
    """Stream Claude responses for real-time output."""
    client = anthropic.Anthropic()

    full_response = ""
    print("Claude: ", end="", flush=True)

    with client.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            full_response += text

    print()
    return full_response
```

**Multi-turn Conversation with Claude:**

```python
def claude_chat(conversation_history: list, system: str) -> str:
    """
    Send a multi-turn conversation to Claude.

    Args:
        conversation_history: List of {"role": str, "content": str} messages
                              Roles must alternate: user, assistant, user, ...
        system: System prompt

    Returns:
        Claude's response text
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=system,
        messages=conversation_history
    )
    return response.content[0].text

# Example usage
history = [
    {"role": "user", "content": "I'm thinking about launching a SaaS product."},
    {"role": "assistant", "content": "That's exciting! What problem does it solve?"},
    {"role": "user", "content": "It helps small restaurants manage their inventory."}
]

system = "You are an experienced startup advisor. Ask clarifying questions and provide actionable advice."
response = claude_chat(history, system)
print(response)
```

**Claude's Unique Features:**

```python
# 1. Very long documents (200K context = ~150,000 words!)
# Great for analyzing entire books, codebases, or legal documents

with open("annual_report.txt", "r") as f:
    document = f.read()

analysis = generate_with_claude(
    prompt=f"Here is an annual report. Identify the top 5 business risks:\n\n{document}",
    system="You are a financial analyst. Be specific and cite evidence from the document.",
    model="claude-opus-4-6",
    max_tokens=2000
)

# 2. Vision capabilities (analyze images)
import base64

def analyze_image(image_path: str, question: str) -> str:
    """Analyze an image with Claude Vision."""
    client = anthropic.Anthropic()

    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine media type
    if image_path.endswith(".png"):
        media_type = "image/png"
    elif image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        media_type = "image/jpeg"
    else:
        media_type = "image/png"

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ]
    )
    return message.content[0].text

# Analyze a chart or diagram
# result = analyze_image("sales_chart.png", "What trend do you see in this sales data?")
```

**OpenAI vs. Claude Comparison:**

| Feature | OpenAI (GPT-4o) | Anthropic (Claude) |
|---------|-----------------|-------------------|
| **Context Window** | 128K tokens | 200K tokens |
| **System Prompt** | Message with role="system" | Top-level parameter |
| **JSON Mode** | Native support | Use prompt engineering |
| **Streaming** | Yes | Yes |
| **Vision** | Yes (GPT-4o) | Yes (claude-sonnet+) |
| **Safety** | Moderate | Strong (Constitutional AI) |
| **Best For** | General tasks, JSON | Long documents, safety |
| **Pricing** | Varies by model | Varies by model |

---

## Slide 8: Prompt Engineering for Chatbots

### Designing Effective System Prompts

**The System Prompt is Everything**

The system prompt defines your chatbot's:
- **Persona**: Who it is
- **Scope**: What it can/can't do
- **Style**: How it communicates
- **Rules**: What constraints it follows

**Framework: The CARE System Prompt Template**

```
C - Character: Who is the assistant?
A - Abilities: What can it do?
R - Restrictions: What must it avoid?
E - Examples: How should it respond?
```

**Example: Customer Service Bot**

```python
customer_service_system = """
CHARACTER:
You are Aria, a friendly and professional customer service agent for TechShop,
an online electronics retailer. You have a warm, helpful personality and always
aim to resolve issues on the first contact.

ABILITIES:
- Answer questions about products in our catalog
- Help with order status, returns, and refunds
- Troubleshoot common technical issues with our products
- Escalate complex issues to a human agent when needed

RESTRICTIONS:
- Never make up product specifications or prices — say you'll look it up
- Don't promise refunds or replacements without order verification
- If a customer is angry, stay calm and empathetic, never argue
- Don't discuss competitor products
- For legal or complex billing disputes, always escalate to a human agent

STYLE:
- Be concise: aim for 2-3 sentences per response
- Use the customer's name if they share it
- Always end with a question or clear next step
- Use casual but professional language (contractions are fine)

EXAMPLES:
User: "My laptop won't turn on."
You: "I'm sorry to hear that! Let's try a quick fix — hold the power button
for 10 seconds to force a restart. Did that help, or is the screen still blank?"

User: "I want a refund."
You: "Of course, I'd be happy to help with that. Could you share your order
number so I can pull up the details?"
"""
```

**Prompt Engineering Techniques:**

**1. Zero-Shot Prompting (Just Instructions)**

```python
# No examples — model uses training knowledge
zero_shot_system = """
You are a sentiment analyzer for customer reviews.
Analyze the review and respond with exactly one word: POSITIVE, NEGATIVE, or NEUTRAL.
"""

response = generate_text(
    prompt="The product arrived on time but the quality was disappointing.",
    system=zero_shot_system,
    temperature=0
)
print(response)  # "NEGATIVE"
```

**2. Few-Shot Prompting (With Examples)**

```python
# Include examples to demonstrate the exact format you want
few_shot_system = """
You are a product description writer. Write compelling, concise product descriptions.

Examples:

Product: Wireless Bluetooth Earbuds
Description: Crystal-clear sound meets 30-hour battery life. These earbuds deliver
premium audio whether you're at the gym or in a meeting — with a charging case
that keeps you powered all day.

Product: Stainless Steel Water Bottle
Description: Stay hydrated in style. Our vacuum-insulated bottle keeps drinks cold
24 hours, hot 12 hours — and its sleek design fits every cup holder.

Now write a description for the product I give you, following the same format.
"""

response = generate_text(
    prompt="Product: Ergonomic Office Chair",
    system=few_shot_system,
    temperature=0.7
)
print(response)
```

**3. Chain-of-Thought Prompting**

```python
# Ask the model to "think step by step" for complex reasoning
cot_system = """
You are a financial advisor. When analyzing investment decisions, always:
1. First, identify the key factors
2. Analyze risks and rewards for each
3. Consider the user's stated goals
4. Then provide your recommendation with clear reasoning

Show your thinking process before giving the final answer.
"""

response = generate_text(
    prompt="I have $10,000 to invest. I'm 35 years old, risk-tolerant, "
           "and want to retire at 65. Should I put it in an index fund or crypto?",
    system=cot_system,
    temperature=0.3
)
print(response)
# Model will reason through each factor before answering
```

**4. Persona + Constraints Combo**

```python
legal_assistant_system = """
You are a legal document assistant for a law firm.

PERSONA: You are knowledgeable about contract law, but you are not a licensed attorney.

CRITICAL RULES:
1. ALWAYS begin any legal analysis with: "This is general information, not legal advice."
2. Never tell a user what they "should" do legally — say what is "typically" done
3. When uncertain, say so explicitly and recommend consulting an attorney
4. Focus on explaining concepts clearly, not providing legal strategy

SCOPE: You can help with:
- Explaining legal terms and concepts
- Summarizing documents (with the disclaimer above)
- Identifying common contract clauses
- Answering general process questions

DO NOT: Draft contracts, provide case strategy, or make definitive legal claims.
"""
```

**Common Mistakes in System Prompts:**

```python
# ❌ BAD: Too vague
bad_system = "Be helpful."

# ❌ BAD: Contradictory instructions
contradictory_system = """
Always be brief.
Always provide comprehensive explanations with examples.
"""

# ❌ BAD: No restrictions on scope
open_ended_system = "You are a medical assistant. Answer any questions."

# ✅ GOOD: Specific, structured, with guardrails
good_system = """
You are MediInfo, a health information assistant.
Provide general health education only — never diagnose or prescribe.
For emergencies, always direct users to call 911 or visit an ER.
Keep responses under 150 words and use plain language.
End each response with: "For personal medical advice, consult your doctor."
"""
```

---

## Slide 9: Building a Multi-Turn Chatbot

### Managing Conversation History

**The Core Challenge: LLMs are Stateless**

Every API call is completely independent. The model has NO memory between calls. We must maintain history ourselves and send it with every request.

```
Call 1:
  Sent:     [user: "My name is Alex"]
  Received: [assistant: "Nice to meet you, Alex!"]

Call 2 (WRONG — model forgot Alex):
  Sent:     [user: "What's my name?"]
  Received: [assistant: "I don't know your name."]  ← Wrong!

Call 2 (CORRECT — we send full history):
  Sent:     [user: "My name is Alex"]
             [assistant: "Nice to meet you, Alex!"]
             [user: "What's my name?"]
  Received: [assistant: "Your name is Alex!"]  ← Correct!
```

**Building a Chatbot Class:**

```python
from openai import OpenAI
from datetime import datetime
import json

class Chatbot:
    """
    A multi-turn chatbot with conversation memory.
    Manages message history and provides a clean interface.
    """

    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini",
                 temperature: float = 0.7, max_tokens: int = 1024,
                 max_history: int = 20):
        """
        Initialize the chatbot.

        Args:
            system_prompt: The chatbot's persona and instructions
            model: OpenAI model to use
            temperature: Response creativity (0-2)
            max_tokens: Maximum tokens per response
            max_history: Maximum messages to keep in history
                        (older messages are dropped to save cost)
        """
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history = max_history
        self.conversation_history = []
        self.total_tokens_used = 0
        self.created_at = datetime.now()

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        Automatically manages conversation history.

        Args:
            user_message: The user's input

        Returns:
            The assistant's response text
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Trim history if too long (keep recent messages)
        if len(self.conversation_history) > self.max_history:
            # Always keep the first message for context, trim middle
            self.conversation_history = self.conversation_history[-self.max_history:]

        # Build full messages list: system + history
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history

        # Call the API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # Extract assistant response
        assistant_message = response.choices[0].message.content

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        # Track token usage
        self.total_tokens_used += response.usage.total_tokens

        return assistant_message

    def reset(self):
        """Clear conversation history (start fresh)."""
        self.conversation_history = []
        print("Conversation reset.")

    def save_conversation(self, filename: str):
        """Save the conversation to a JSON file."""
        data = {
            "system_prompt": self.system_prompt,
            "created_at": self.created_at.isoformat(),
            "total_tokens": self.total_tokens_used,
            "messages": self.conversation_history
        }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Conversation saved to {filename}")

    def get_stats(self) -> dict:
        """Return conversation statistics."""
        return {
            "messages": len(self.conversation_history),
            "total_tokens": self.total_tokens_used,
            "estimated_cost_usd": self.total_tokens_used * 0.00000015,  # gpt-4o-mini
            "duration_minutes": (datetime.now() - self.created_at).seconds / 60
        }


# Usage Example — Build a customer service bot
bot = Chatbot(
    system_prompt="""You are Aria, a helpful customer service agent for TechShop.
    Help customers with orders, returns, and product questions.
    Be friendly, concise, and always ask for the order number when processing requests.""",
    temperature=0.5
)

# Simulate a conversation
print("TechShop Customer Service")
print("=" * 40)

responses = [
    "Hi! I have a problem with my order.",
    "My name is Maria and order number is TS-9876.",
    "The laptop I received has a cracked screen.",
    "How long will the replacement take?"
]

for user_input in responses:
    print(f"\nUser: {user_input}")
    response = bot.chat(user_input)
    print(f"Aria: {response}")

# Check stats
stats = bot.get_stats()
print(f"\n--- Session Stats ---")
print(f"Messages: {stats['messages']}")
print(f"Tokens used: {stats['total_tokens']}")
print(f"Est. cost: ${stats['estimated_cost_usd']:.4f}")
```

**Interactive Command-Line Chatbot:**

```python
def run_interactive_chatbot():
    """
    Run a chatbot in the terminal.
    Type 'quit' to exit, 'reset' to clear history, 'stats' for usage.
    """
    system_prompt = input("Enter system prompt (or press Enter for default): ").strip()
    if not system_prompt:
        system_prompt = "You are a helpful, friendly assistant."

    bot = Chatbot(system_prompt=system_prompt)

    print("\n=== Chatbot Ready ===")
    print("Commands: 'quit' to exit, 'reset' to clear, 'stats' for usage\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue
        elif user_input.lower() == "quit":
            print("Goodbye!")
            bot.save_conversation("conversation_log.json")
            break
        elif user_input.lower() == "reset":
            bot.reset()
        elif user_input.lower() == "stats":
            stats = bot.get_stats()
            print(f"Stats: {stats}")
        else:
            response = bot.chat(user_input)
            print(f"Bot: {response}\n")

# run_interactive_chatbot()  # Uncomment to run
```

---

## Slide 10: Advanced Chatbot Features

### Streaming, Function Calling & Memory Strategies

**Streaming for Better User Experience:**

```python
from openai import OpenAI

class StreamingChatbot:
    """Chatbot with streaming responses."""

    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.system = system_prompt
        self.history = []

    def stream_chat(self, user_message: str) -> str:
        """Stream the response token by token."""
        self.history.append({"role": "user", "content": user_message})

        messages = [{"role": "system", "content": self.system}] + self.history

        stream = self.client.chat.completions.create(
            model=self.model if hasattr(self, 'model') else "gpt-4o-mini",
            messages=messages,
            stream=True
        )

        full_response = ""
        print("Bot: ", end="", flush=True)

        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                full_response += delta.content

        print()  # Newline after streaming completes
        self.history.append({"role": "assistant", "content": full_response})
        return full_response
```

**Function Calling (Tool Use):**

Let the model call your Python functions to get real-time data!

```python
import json
from openai import OpenAI

client = OpenAI()

# Define tools the model can call
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Get the current status of a customer order",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID, e.g. TS-12345"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Get the current price of a product",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name or ID of the product"
                    }
                },
                "required": ["product_name"]
            }
        }
    }
]

# Your actual functions (would connect to a real database/API)
def get_order_status(order_id: str) -> dict:
    """Simulate order status lookup."""
    # In production, this would query your database
    mock_orders = {
        "TS-12345": {"status": "Shipped", "eta": "March 6, 2026", "carrier": "FedEx"},
        "TS-99999": {"status": "Processing", "eta": "March 8, 2026", "carrier": None}
    }
    return mock_orders.get(order_id, {"status": "Not Found", "eta": None})

def get_product_price(product_name: str) -> dict:
    """Simulate product price lookup."""
    mock_prices = {
        "laptop": {"price": 999.99, "in_stock": True},
        "mouse": {"price": 29.99, "in_stock": True},
        "keyboard": {"price": 79.99, "in_stock": False}
    }
    key = product_name.lower()
    for k, v in mock_prices.items():
        if k in key:
            return v
    return {"price": None, "in_stock": None}

def chat_with_tools(user_message: str, history: list) -> str:
    """Chat with function calling enabled."""
    history.append({"role": "user", "content": user_message})

    messages = [
        {"role": "system", "content": "You are a helpful TechShop assistant. "
         "Use the available tools to look up real order and product information."}
    ] + history

    # First call: model may request a tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    response_message = response.choices[0].message

    # Check if model wants to call a function
    if response_message.tool_calls:
        # Execute each tool call
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Call the actual function
            if function_name == "get_order_status":
                result = get_order_status(**args)
            elif function_name == "get_product_price":
                result = get_product_price(**args)
            else:
                result = {"error": "Unknown function"}

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Second call: model uses tool results to answer
        final_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        answer = final_response.choices[0].message.content
    else:
        answer = response_message.content

    history.append({"role": "assistant", "content": answer})
    return answer

# Test it!
history = []
print(chat_with_tools("What's the status of order TS-12345?", history))
print(chat_with_tools("How much is a laptop?", history))
```

**Memory Strategies:**

```python
# Strategy 1: Full History (Simple, gets expensive with long chats)
# - Send all messages every time
# - Best for: short conversations

# Strategy 2: Windowed History (Most Common)
# - Keep last N messages only
history = history[-20:]  # Keep last 20 messages

# Strategy 3: Summarized Memory
# When history gets long, summarize it
def summarize_and_compress(history: list, keep_last: int = 5) -> list:
    """
    Compress old conversation history into a summary.
    Keeps recent messages verbatim, summarizes the rest.
    """
    if len(history) <= keep_last:
        return history

    old_messages = history[:-keep_last]
    recent_messages = history[-keep_last:]

    # Summarize old messages
    old_text = "\n".join([
        f"{m['role'].title()}: {m['content']}" for m in old_messages
    ])

    summary_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize this conversation in 2-3 sentences."},
            {"role": "user", "content": old_text}
        ],
        max_tokens=150
    )
    summary = summary_response.choices[0].message.content

    # Return: summary as context + recent verbatim messages
    compressed = [{"role": "user", "content": f"[Conversation summary: {summary}]"}]
    compressed += recent_messages
    return compressed

# Strategy 4: Vector Database Memory (RAG — covered next)
# Store conversation as embeddings, retrieve relevant context
```

---

**End of Batch 2 (Slides 6-10)**

*Continue to Batch 3 for RAG, Chatbot Architectures & Business Applications (Slides 11-15)*
