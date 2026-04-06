"""
Week 11: Slide 13 — Business Intelligence Agent

Description:
    A complete, production-patterned AI agent that automates a business
    research task using Claude's tool-use API. The agent is given a set
    of tools (web fetching, data lookup, file saving) and autonomously
    decides how to use them to complete a multi-step business analysis.

    Demonstrates the full agent loop:
        Task → Plan → Call tool → Observe → Repeat → Final answer

Dependencies:
    - anthropic
    - requests
    - beautifulsoup4
    - python-dotenv

Usage:
    python slide13_business_agent.py
"""

import json
import os
import textwrap
from datetime import datetime
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ─── Tool Definitions ──────────────────────────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "fetch_url_text",
        "description": (
            "Fetches the main text content from a URL. "
            "Use this to read web pages, articles, or pricing pages. "
            "Returns the visible text content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL to fetch (must start with http/https)"
                }
            },
            "required": ["url"]
        }
    },
    {
        "name": "get_market_data",
        "description": (
            "Returns simulated market data for a company or industry sector. "
            "Use this to look up market size, growth rates, or key players."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Company name or industry sector to look up"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_report",
        "description": "Saves a completed report to a local file. Use this when the research is complete.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Filename for the report (e.g., 'market_analysis.md')"
                },
                "content": {
                    "type": "string",
                    "description": "Full markdown content of the report"
                }
            },
            "required": ["filename", "content"]
        }
    },
    {
        "name": "calculate_metrics",
        "description": (
            "Performs business calculations such as growth rates, market share, "
            "or ROI given input numbers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["growth_rate", "market_share", "roi", "cagr"],
                    "description": "Type of calculation to perform"
                },
                "values": {
                    "type": "object",
                    "description": "Input values as key-value pairs (e.g., {current: 120, previous: 100})"
                }
            },
            "required": ["operation", "values"]
        }
    }
]


# ─── Tool Implementations ──────────────────────────────────────────────────────

def fetch_url_text(url: str) -> str:
    """
    Fetch readable text from a URL.

    Uses requests + BeautifulSoup to extract visible text.
    Falls back to simulated content if the request fails (for demo purposes).

    Args:
        url: Full URL to fetch.

    Returns:
        Extracted text content, truncated to 3000 chars.
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (research bot)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = soup.get_text(separator="\n", strip=True)
        # Collapse multiple blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        text = "\n".join(lines)
        return text[:3000] + ("..." if len(text) > 3000 else "")

    except ImportError:
        return f"[Simulated] Content from {url}: Product pricing, features, and FAQ information."
    except Exception as e:
        return f"[Fetch failed: {e}] Simulated content for {url}."


def get_market_data(query: str) -> str:
    """
    Return simulated market research data for a company or sector.

    In production, this would call a financial data API (Bloomberg, Refinitiv,
    or a web scraper pointed at market research reports).

    Args:
        query: Company or industry name.

    Returns:
        JSON-formatted market data string.
    """
    simulated_data = {
        "cloud storage": {
            "market_size_2025": "$97B",
            "cagr_2025_2030": "22%",
            "top_players": ["AWS S3", "Google Cloud Storage", "Azure Blob", "Dropbox", "Box"],
            "key_trend": "AI-integrated storage with automatic tagging and search"
        },
        "generative ai": {
            "market_size_2025": "$36B",
            "cagr_2025_2030": "42%",
            "top_players": ["OpenAI", "Anthropic", "Google DeepMind", "Meta AI", "Mistral"],
            "key_trend": "Shift from consumer chatbots to enterprise agent workflows"
        },
        "default": {
            "market_size_2025": "Data not available in this demo",
            "note": "In production, integrate with a financial data provider or web scraper",
            "cagr_estimate": "Industry-average: 8–15% for most tech sectors"
        }
    }

    query_lower = query.lower()
    for key, data in simulated_data.items():
        if key in query_lower:
            return json.dumps(data, indent=2)

    return json.dumps(simulated_data["default"], indent=2)


def save_report(filename: str, content: str) -> str:
    """
    Save a report to the local filesystem.

    Args:
        filename: Target filename.
        content: Markdown content to write.

    Returns:
        Confirmation message with file path.
    """
    # Sanitize filename
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)
    if not safe_name.endswith((".md", ".txt")):
        safe_name += ".md"

    try:
        with open(safe_name, "w", encoding="utf-8") as f:
            f.write(f"*Generated by AI Agent — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
            f.write(content)
        return f"Report saved to: {os.path.abspath(safe_name)}"
    except OSError as e:
        return f"Failed to save report: {e}"


def calculate_metrics(operation: str, values: dict[str, Any]) -> str:
    """
    Perform standard business calculations.

    Args:
        operation: One of 'growth_rate', 'market_share', 'roi', 'cagr'.
        values: Dict of input values specific to the operation.

    Returns:
        Calculation result as a formatted string.

    Example:
        >>> calculate_metrics("growth_rate", {"current": 120, "previous": 100})
        'Year-over-year growth rate: 20.00%'
    """
    try:
        if operation == "growth_rate":
            rate = (values["current"] - values["previous"]) / values["previous"] * 100
            return f"Year-over-year growth rate: {rate:.2f}%"

        elif operation == "market_share":
            share = values["company_revenue"] / values["total_market"] * 100
            return f"Market share: {share:.2f}%"

        elif operation == "roi":
            roi = (values["gain"] - values["cost"]) / values["cost"] * 100
            return f"ROI: {roi:.2f}%"

        elif operation == "cagr":
            n = values.get("years", 5)
            cagr = ((values["end_value"] / values["start_value"]) ** (1 / n) - 1) * 100
            return f"CAGR over {n} years: {cagr:.2f}%"

        else:
            return f"Unknown operation: {operation}"

    except (KeyError, ZeroDivisionError, TypeError) as e:
        return f"Calculation error: {e}. Check that all required values are provided."


# ─── Tool Dispatcher ───────────────────────────────────────────────────────────

def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> str:
    """
    Dispatch a tool call to the correct implementation.

    Args:
        tool_name: Name of the tool to execute.
        tool_input: Input parameters for the tool.

    Returns:
        String result to pass back to the model.
    """
    dispatch = {
        "fetch_url_text": lambda i: fetch_url_text(i["url"]),
        "get_market_data": lambda i: get_market_data(i["query"]),
        "save_report": lambda i: save_report(i["filename"], i["content"]),
        "calculate_metrics": lambda i: calculate_metrics(i["operation"], i["values"]),
    }

    handler = dispatch.get(tool_name)
    if not handler:
        return f"Error: Unknown tool '{tool_name}'"

    try:
        return handler(tool_input)
    except Exception as e:
        return f"Tool execution error: {e}"


# ─── Agent Loop ────────────────────────────────────────────────────────────────

def run_agent(
    task: str,
    model: str = "claude-sonnet-4-6",
    max_iterations: int = 10,
    verbose: bool = True
) -> str:
    """
    Run the core agent loop until the task is complete.

    The loop:
        1. Send messages to Claude with available tools
        2. If stop_reason == "end_turn": return final answer
        3. If stop_reason == "tool_use": execute tools, append results, continue

    Args:
        task: Natural language task description.
        model: Claude model to use.
        max_iterations: Safety limit on loop iterations.
        verbose: Print tool calls and results to stdout.

    Returns:
        Final answer text from the agent.

    Example:
        >>> answer = run_agent("Research the AI market and produce a 1-page brief.")
        >>> print(answer)
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    system_prompt = textwrap.dedent("""
        You are a business research analyst. When given a research task:
        1. Use tools to gather information — don't rely solely on your training data
        2. When you have enough information, synthesize findings into a clear, structured report
        3. Save the final report to a file using the save_report tool
        4. Return a concise executive summary to the user

        Be efficient: gather just enough data to produce a high-quality deliverable.
    """).strip()

    messages = [{"role": "user", "content": task}]
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )

        if verbose:
            print(f"\n[Iteration {iteration}] Stop reason: {response.stop_reason}")

        # Agent is done
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text

        # Process tool calls
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if verbose:
                    print(f"  → Tool: {block.name}({json.dumps(block.input)[:120]}...)")
                result = execute_tool(block.name, block.input)
                if verbose:
                    print(f"  ← Result: {result[:200]}...")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        if not tool_results:
            # No tools called and not end_turn — shouldn't happen, but break to avoid loop
            break

        # Append assistant message and tool results
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

    return "Agent reached maximum iterations without completing the task."


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Business Intelligence Agent — Week 11 Demo")
    print("=" * 60)

    # Task 1: Market research with calculation
    task1 = """
    Research the generative AI market. I need:
    1. Current market size and growth rate
    2. Top 5 players and their estimated positions
    3. The CAGR if the market grows from $36B (2025) to $200B (2030)
    4. A brief strategic recommendation for a business analytics firm

    Save the findings as 'genai_market_brief.md' and give me an executive summary.
    """

    print(f"\nTask: {task1.strip()[:150]}...\n")
    result = run_agent(task1, verbose=True)

    print("\n" + "="*60)
    print("EXECUTIVE SUMMARY")
    print("="*60)
    print(result)

    # Task 2: Simpler calculation demo
    print("\n" + "="*60)
    print("Quick calculation demo (no API call)")
    print("="*60)
    result = calculate_metrics("growth_rate", {"current": 200, "previous": 36})
    print(f"Market growth 2025→2030: {result}")

    result2 = calculate_metrics("cagr", {"start_value": 36, "end_value": 200, "years": 5})
    print(f"5-year CAGR: {result2}")
