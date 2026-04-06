"""
Week 11: Slide 6 — Model Selection Framework

Description:
    Demonstrates how to call multiple LLM providers from Python using
    a unified interface, enabling easy A/B comparisons of model quality,
    speed, and cost. Useful for evaluating which model to use for a
    specific business task before committing to one provider.

Dependencies:
    - anthropic
    - openai
    - python-dotenv

Usage:
    python slide06_model_comparison.py
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

import anthropic
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelResult:
    """Captures the result of one model call for comparison."""
    model_name: str
    provider: str
    response_text: str
    latency_seconds: float
    input_tokens: int
    output_tokens: int

    @property
    def cost_estimate_usd(self) -> float:
        """
        Rough cost estimate based on approximate public pricing (early 2026).
        Prices per million tokens (input/output).
        """
        pricing = {
            "claude-haiku-4-5-20251001":    (0.25, 1.25),
            "claude-sonnet-4-6":          (3.00, 15.00),
            "gpt-4o-mini":            (0.15, 0.60),
        }
        input_price, output_price = pricing.get(self.model_name, (1.0, 5.0))
        cost = (self.input_tokens / 1_000_000 * input_price +
                self.output_tokens / 1_000_000 * output_price)
        return round(cost, 6)

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"Provider: {self.provider} | Model: {self.model_name}\n"
            f"Latency: {self.latency_seconds:.2f}s | "
            f"Tokens: {self.input_tokens} in / {self.output_tokens} out | "
            f"Est. cost: ${self.cost_estimate_usd:.5f}\n"
            f"{'─'*60}\n"
            f"{self.response_text}\n"
        )


def call_claude(
    prompt: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 512
) -> Optional[ModelResult]:
    """
    Call Anthropic Claude and return a ModelResult.

    Args:
        prompt: User message to send.
        model: Anthropic model identifier.
        max_tokens: Maximum tokens to generate.

    Returns:
        ModelResult with response and metadata, or None on error.

    Example:
        >>> result = call_claude("Summarize RAG in one sentence.")
        >>> print(result.response_text)
    """
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    try:
        start = time.time()
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start

        return ModelResult(
            model_name=model,
            provider="Anthropic",
            response_text=response.content[0].text,
            latency_seconds=latency,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
    except anthropic.APIError as e:
        print(f"[Anthropic] API error: {e}")
        return None


def call_openai(
    prompt: str,
    model: str = "gpt-4o-mini",
    max_tokens: int = 512
) -> Optional[ModelResult]:
    """
    Call OpenAI and return a ModelResult.

    Args:
        prompt: User message to send.
        model: OpenAI model identifier.
        max_tokens: Maximum tokens to generate.

    Returns:
        ModelResult with response and metadata, or None on error.
    """
    try:
        from openai import OpenAI, APIError
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        start = time.time()
        response = openai_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = time.time() - start

        return ModelResult(
            model_name=model,
            provider="OpenAI",
            response_text=response.choices[0].message.content,
            latency_seconds=latency,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
    except ImportError:
        print("[OpenAI] openai package not installed. Run: pip install openai")
        return None
    except Exception as e:
        print(f"[OpenAI] Error: {e}")
        return None


def compare_models(
    prompt: str,
    include_openai: bool = True
) -> list[ModelResult]:
    """
    Run the same prompt across multiple models and return all results.

    Args:
        prompt: The prompt to test.
        include_openai: Whether to also call OpenAI (requires API key).

    Returns:
        List of ModelResult objects for comparison.

    Example:
        >>> results = compare_models("What are the top 3 uses of AI in finance?")
        >>> for r in results:
        ...     print(r.summary())
    """
    results = []

    # Test Claude Haiku (fast, cheap)
    print("Calling Claude Haiku...")
    if (r := call_claude(prompt, model="claude-haiku-4-5-20251001")):
        results.append(r)

    # Test Claude Sonnet (high quality)
    print("Calling Claude Sonnet...")
    if (r := call_claude(prompt, model="claude-sonnet-4-6")):
        results.append(r)

    # Optionally test GPT-4o-mini
    if include_openai and os.environ.get("OPENAI_API_KEY"):
        print("Calling GPT-4o-mini...")
        if (r := call_openai(prompt)):
            results.append(r)

    return results


def print_comparison_table(results: list[ModelResult]) -> None:
    """
    Print a formatted side-by-side comparison table.

    Args:
        results: List of ModelResult objects to compare.
    """
    if not results:
        print("No results to compare.")
        return

    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model':<30} {'Latency':>10} {'Tokens Out':>12} {'Est. Cost':>12}")
    print("-"*80)
    for r in results:
        print(f"{r.provider + '/' + r.model_name:<30} "
              f"{r.latency_seconds:>9.2f}s "
              f"{r.output_tokens:>12} "
              f"${r.cost_estimate_usd:>11.5f}")
    print("="*80)

    print("\n" + "="*80)
    print("FULL RESPONSES")
    print("="*80)
    for r in results:
        print(r.summary())


# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Business task — use this as a benchmark
    PROMPT = """You are a business analyst. A retail company's online sales dropped
15% this quarter despite running a larger advertising campaign than last year.
Provide a structured analysis:
1. The 3 most likely root causes
2. The data you would need to confirm each hypothesis
3. Your top recommendation and why

Keep your response under 250 words. Be specific and actionable."""

    print(f"Testing prompt:\n{PROMPT[:150]}...\n")
    results = compare_models(PROMPT, include_openai=False)
    print_comparison_table(results)

    # ── Quick benchmark: just latency and length ──────────────────────────────
    benchmark_prompts = [
        "Summarize the key benefits of RAG in one sentence.",
        "What is the difference between fine-tuning and RAG?",
        "Name the three main AI agent frameworks released in 2025.",
    ]

    print("\n" + "="*80)
    print("QUICK LATENCY BENCHMARK (3 short prompts, Claude Haiku only)")
    print("="*80)
    total_cost = 0.0
    for p in benchmark_prompts:
        r = call_claude(p, model="claude-haiku-4-5-20251001", max_tokens=100)
        if r:
            print(f"[{r.latency_seconds:.2f}s | {r.output_tokens} tok] {p[:60]}")
            total_cost += r.cost_estimate_usd
    print(f"\nTotal estimated cost for 3 short calls: ${total_cost:.5f}")
