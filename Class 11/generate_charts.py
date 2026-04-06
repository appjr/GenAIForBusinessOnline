#!/usr/bin/env python3
"""
Generate chart images for Week 11 slides using OpenAI gpt-image-1.

For each slide that benefits from a data visualization, sends a detailed
chart description to OpenAI and saves the result as a PNG.

Charts are saved to: Class 11/charts/slide_NN_chart.png
Progress is tracked in: Class 11/charts/.progress.json

Usage:
    OPENAI_API_KEY=sk-proj-... python3 generate_charts.py

Cost estimate: 11 charts × ~$0.06 = ~$0.66  (gpt-image-1 high quality)
Resumes from where it left off if interrupted.
"""

import base64
import json
import os
import time
from pathlib import Path

import openai

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY    = os.environ.get("OPENAI_API_KEY", "")
SCRIPT_DIR = Path(__file__).parent
OUT_DIR    = SCRIPT_DIR / "charts"
PROGRESS_FILE = OUT_DIR / ".progress.json"

IMG_MODEL = "gpt-image-1"
IMG_SIZE  = "1536x1024"
IMG_QUAL  = "high"

# Shared style instructions appended to every prompt
IMG_STYLE = (
    "Style: modern professional university course data visualization. "
    "Clean white (#ffffff) background. Bold sans-serif typography (Inter or Calibri). "
    "Color palette: deep navy #16213e for primary bars/headers, vibrant teal #0ea5e9 "
    "for highlights and accent bars, purple #533483 for secondary elements, "
    "warm amber #f59e0b for callout numbers. Sharp edges, clear axis labels, "
    "bold value labels on every data point. McKinsey/HBR infographic quality. "
    "Minimal grid lines (light gray). No clipart. Flat design with subtle drop shadows. "
    "UTDallas graduate course quality. High information density, professional finish."
)

# ── Chart definitions ──────────────────────────────────────────────────────────
# Each entry: slide_number → (filename_prefix, detailed_prompt)

CHARTS = {
    3: (
        "slide_03_chart",
        "Create a vertical five-layer stack architecture diagram for the AI Technology "
        "Ecosystem. Five horizontal bands stacked from bottom to top, each with a colored "
        "left-edge accent and label. "
        "Layer 1 (bottom, navy #16213e): 'Foundation Models' — list GPT-4.1, Claude 3.7, "
        "Gemini 2.5, Llama 4, DeepSeek V3 as chip-style badges. "
        "Layer 2 (teal #0ea5e9): 'APIs & SDKs' — Anthropic API, OpenAI API, Google AI Studio, "
        "Hugging Face. "
        "Layer 3 (purple #533483): 'Orchestration Frameworks' — LangChain, LangGraph, "
        "CrewAI, Claude Agent SDK, OpenAI Agents SDK. "
        "Layer 4 (amber #f59e0b): 'Applications & Products' — RAG Systems, AI Agents, "
        "Workflow Automation, AI Coding Tools. "
        "Layer 5 (top, green #22c55e): 'Business Value' — Cost Reduction, Revenue Growth, "
        "Speed to Market, Competitive Advantage. "
        "Show upward arrows between layers indicating value flow. "
        "Left side: vertical label 'Abstraction Level'. "
        "Title at top: 'The Five-Layer AI Ecosystem'. " + IMG_STYLE
    ),
    4: (
        "slide_04_chart",
        "Create a bubble chart titled 'Frontier AI Models: Cost vs. Capability (2026)'. "
        "X-axis: 'Cost per Million Tokens (USD)' from $0 to $8, log scale. "
        "Y-axis: 'Capability Score' from 50 to 100. "
        "Bubble size represents context window size. "
        "Plot these labeled bubbles: "
        "GPT-4.1 (x=2.0, y=95, large bubble, navy) — label 'GPT-4.1 $2/M'; "
        "Claude 3.7 Sonnet (x=3.0, y=94, largest bubble representing 200K ctx, teal) — "
        "label 'Claude 3.7 $3/M'; "
        "Gemini 2.5 Pro (x=7.0, y=93, extra-large bubble representing 1M ctx, purple) — "
        "label 'Gemini 2.5 Pro $7/M'; "
        "Claude 3.7 Haiku (x=0.25, y=78, medium bubble, teal lighter shade) — "
        "label 'Haiku $0.25/M'; "
        "GPT-4.1 mini (x=0.30, y=75, medium bubble, navy lighter) — "
        "label 'GPT-4.1 mini $0.30/M'; "
        "DeepSeek V3 (x=0.27, y=85, medium bubble, amber, star marker = open source) — "
        "label 'DeepSeek V3* $0.27/M'; "
        "Llama 4 Scout (x=0.05, y=72, small bubble, amber) — "
        "label 'Llama 4 Scout* FREE'. "
        "Add legend: filled circle = proprietary, star = open source. "
        "Annotation box in bottom-right: 'Sweet spot: High capability, low cost'. "
        "Draw a shaded 'sweet spot' zone (green tint) in bottom-right quadrant. " + IMG_STYLE
    ),
    5: (
        "slide_05_chart",
        "Create a grouped bar chart titled 'AI Training Cost Collapse: Open Source vs Proprietary'. "
        "X-axis: years/models from 2020 to 2026. "
        "Y-axis: 'Estimated Training Cost (USD)' on log scale from $100K to $200M. "
        "Show two bar colors: navy for proprietary, amber for open source. "
        "Data — Proprietary: GPT-3 2020=$12M (navy), GPT-4 2023=$100M (navy). "
        "Open source: LLaMA 1 2023=$3M (amber), Mistral 7B 2023=$600K (amber), "
        "LLaMA 2 2023=$5M (amber), Mixtral 2023=$4M (amber), "
        "DeepSeek V3 2024=$6M (amber), DeepSeek R1 2025=$5.6M (amber), "
        "Llama 4 2026=$8M (amber). "
        "Add bold value labels on each bar. "
        "Draw a large teal downward arrow annotation: 'Open source training costs "
        "compressed 95% in 3 years'. "
        "Add callout box: 'GPT-4 cost in 2023 = $100M → equivalent model in 2026 = $6M'. "
        "Legend in top-right. " + IMG_STYLE
    ),
    6: (
        "slide_06_chart",
        "Create a decision flowchart titled 'AI Model Selection Framework'. "
        "Start with a diamond at top: 'Does data need to stay private?'. "
        "YES branch (left, red) → diamond: 'On-premise hardware available?' "
        "  YES → green box: 'Self-hosted open source\\nLlama 4, Phi-4, Mistral' "
        "  NO → blue box: 'Private cloud deployment\\nBedrock, Azure OpenAI, Vertex' "
        "NO branch (right, green) → diamond: 'Need maximum capability?' "
        "  YES → diamond: 'Cost sensitive?' "
        "    YES → teal box: 'GPT-4.1 mini or Claude Haiku\\n$0.25–0.30/M tokens' "
        "    NO → navy box: 'Frontier model\\nClaude 3.7 / GPT-4.1 / Gemini 2.5 Pro' "
        "  NO → diamond: 'Volume > 10M calls/month?' "
        "    YES → amber box: 'Fine-tune a small model\\nLoRA on Llama 4 8B' "
        "    NO → teal box: 'GPT-4.1 nano or Haiku\\n$0.05/M tokens' "
        "Use rounded rectangles for terminal nodes, diamonds for decisions. "
        "Color terminal nodes: navy=frontier, teal=cost-optimized, amber=fine-tuned, "
        "blue=private cloud. " + IMG_STYLE
    ),
    7: (
        "slide_07_chart",
        "Create a two-part infographic titled 'From Chatbot to AI Agent'. "
        "LEFT HALF (40% width): Simple linear diagram labeled 'Chatbot (2022–2024)'. "
        "Show: User input box → single arrow → LLM box → single arrow → Text output box. "
        "Below: label 'One round trip. Human drives every step.' "
        "Color scheme: light gray with navy text. "
        "RIGHT HALF (60% width): Circular agent loop diagram labeled 'AI Agent (2025–2026)'. "
        "Show a clockwise circular flow with 5 nodes connected by curved arrows: "
        "1. 'Receive Task' (navy circle, top) "
        "2. 'Plan Steps' (teal circle, right) "
        "3. 'Call Tool' (purple circle, bottom-right) — show tool icons: web, code, DB "
        "4. 'Observe Result' (amber circle, bottom-left) "
        "5. 'Reason & Adapt' (teal circle, left) "
        "Center of circle: 'Goal: Complete Task' "
        "Below loop: label 'Autonomous. Low supervision. Produces results, not text.' "
        "Add a callout: '4-hour analyst task → 15 minutes'. " + IMG_STYLE
    ),
    14: (
        "slide_14_chart",
        "Create a horizontal bar chart titled 'AI Coding Tools: Developer Productivity Gains'. "
        "Y-axis: development tasks (top to bottom). "
        "X-axis: 'Speedup Factor' from 0 to 95. "
        "Bars (with bold value labels at end of each bar): "
        "Write boilerplate/scaffolding: 90× (amber, longest bar) "
        "Write new function: 7× (teal) "
        "Write unit tests: 7× (teal) "
        "Fix a bug: 4× (navy) "
        "Understand unfamiliar codebase: 5× (navy) "
        "Code review coverage: +35% more PRs reviewed (purple, dashed outline) "
        "Make the 90× bar dramatically longer than the rest to show the scale difference. "
        "Add annotation arrow on 90× bar: 'Biggest gain: zero-value boilerplate eliminated'. "
        "Color bars with navy/teal/amber alternating, bold white value labels inside bars. "
        "Add subtitle below chart: 'Source: GitHub, StackOverflow Developer Surveys 2025–2026'. "
        + IMG_STYLE
    ),
    20: (
        "slide_20_chart",
        "Create a horizontal bar chart titled 'AI Automation: Time Saved by Business Process'. "
        "Show percentage time saved with color-coded intensity. "
        "Y-axis: business processes. X-axis: '% Time Saved' from 0% to 100%. "
        "Bars: "
        "Invoice processing (extract, validate, route): 97% — navy bar, label '5 min → 10 sec' "
        "Contract risk review (50-page): 97% — navy bar, label '2–3 hrs → 3 min' "
        "Competitive intelligence report: 94% — teal bar, label '6 hrs → 20 min' "
        "Meeting transcript → action items: 93% — teal bar, label '30 min → 2 min' "
        "Customer support ticket triage: 91% — purple bar, label '8 min → 45 sec' "
        "Show a vertical dashed line at 90% labeled 'Automation Threshold'. "
        "All bars exceed 90% — use green checkmark icons on the left. "
        "Add callout box at top-right: 'Avg cost per automated task: <$0.10'. "
        "Bold percentage labels at the end of each bar. " + IMG_STYLE
    ),
    25: (
        "slide_25_chart",
        "Create a grouped horizontal bar chart titled 'AI Impact Across Industries: Measured Results'. "
        "Four industry groups, each with 2 key metrics: "
        "FINANCE (navy group): "
        "  Earnings call analysis: 2 hrs → 8 min (−93%) "
        "  Regulatory gap analysis: days → hours (−85%) "
        "HEALTHCARE (teal group): "
        "  Physician documentation: −2 hrs/day (−33%) "
        "  Prior auth letter drafting: 45 min → 4 min (−91%) "
        "RETAIL (purple group): "
        "  Product descriptions: 10,000 SKUs/hr vs 100/day manual (100× faster) "
        "  Customer chat resolution: 70% without human agent "
        "CONSULTING (amber group): "
        "  RFP section drafting: 30 min vs 1–2 days (−96%) "
        "  Contract risk flagging: 3 min vs 2 hrs (−97%) "
        "Use grouped bars with industry-colored pairs. "
        "Show percentage improvement as bold labels. "
        "Add bottom annotation: 'Consistent pattern: AI absorbs 60–80% of information work'. "
        + IMG_STYLE
    ),
    28: (
        "slide_28_chart",
        "Create a three-tier pyramid diagram titled 'AI Adoption Maturity Model'. "
        "The pyramid has three horizontal bands from bottom (widest) to top (narrowest). "
        "BOTTOM TIER (widest, navy): "
        "  Label: 'Tier 1: Individual Productivity' "
        "  Tools: Claude/ChatGPT, Cursor/Copilot, Perplexity "
        "  Cost badge (amber): '$0–$50/person/month' "
        "  Timeline badge: 'Start now' "
        "MIDDLE TIER (medium, teal): "
        "  Label: 'Tier 2: Team Workflows' "
        "  Tools: Zapier/Make/n8n, NotebookLM, RAG chatbot "
        "  Cost badge (amber): '$100–$1,000/month' "
        "  Timeline badge: 'Months 1–6' "
        "TOP TIER (narrowest, purple): "
        "  Label: 'Tier 3: Enterprise AI' "
        "  Tools: Fine-tuned models, multi-agent systems, AI products "
        "  Cost badge (amber): '$10,000+/month' "
        "  Timeline badge: 'Months 6–18' "
        "LEFT SIDE: vertical arrow pointing up labeled 'Complexity & Investment' "
        "RIGHT SIDE: vertical arrow pointing up labeled 'Business Impact & Scale' "
        "Add callout at bottom: 'Start at Tier 1. Prove value. Then invest upward.' "
        + IMG_STYLE
    ),
    29: (
        "slide_29_chart",
        "Create a radar/spider chart titled 'Six AI Mega-Trends: 2026 Impact Assessment'. "
        "Six axes, equally spaced (60° apart), each scored 0–10: "
        "1. Intelligence Cost Collapse (score: 9) "
        "2. Agentic AI Production Readiness (score: 8) "
        "3. Domain Specialization / Moats (score: 7) "
        "4. Multimodal Baseline (score: 8) "
        "5. AI Governance & Compliance (score: 6) "
        "6. Human Role Shift (score: 9) "
        "Show TWO overlapping polygons: "
        "  Navy filled polygon (semi-transparent): '2024 State' "
        "  Teal filled polygon (semi-transparent): '2026 State' "
        "2024 scores (lower): 5, 4, 4, 5, 3, 6 "
        "2026 scores (higher): 9, 8, 7, 8, 6, 9 "
        "Shaded area between polygons represents growth. "
        "Legend: navy='2024', teal='2026'. "
        "Axis labels with bold text outside the radar. "
        "Subtitle: 'Area growth = acceleration of each trend in 2 years'. "
        + IMG_STYLE
    ),
    30: (
        "slide_30_chart",
        "Create a horizontal bar chart titled 'Emerging AI Career Roles: Salary Ranges (US, 2026)'. "
        "Show salary ranges as horizontal bands (min to max), sorted highest to lowest. "
        "Each role shows a floating range bar from minimum to maximum salary. "
        "Roles and ranges (top to bottom): "
        "AI Product Manager: $140K–$210K (navy) "
        "ML Ops / AI Ops Engineer: $130K–$185K (navy) "
        "AI Automation Engineer: $100K–$165K (teal) "
        "Prompt / LLM Engineer: $90K–$155K (teal) "
        "AI Governance Analyst: $90K–$145K (purple) "
        "Business AI Analyst: $80K–$135K (purple) "
        "AI Data Curator: $80K–$135K (amber) "
        "X-axis: salary in $K from $60K to $220K. "
        "Show midpoint dot on each range bar. "
        "Bold labels on right side of each bar showing '$XXK–$XXK'. "
        "Add vertical dashed line at $100K labeled 'Six-figure threshold'. "
        "Callout at top-right: 'All roles above $80K baseline — new field, premium wages'. "
        "Add note at bottom: 'Source: LinkedIn Salary Insights, Levels.fyi, 2026 estimates'. "
        + IMG_STYLE
    ),
}


# ── Progress helpers ───────────────────────────────────────────────────────────

def load_progress():
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("done", []))
    return set()


def save_progress(done):
    PROGRESS_FILE.write_text(json.dumps({"done": sorted(done)}, indent=2))


# ── Main generation ────────────────────────────────────────────────────────────

def generate_charts():
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        raise SystemExit(1)

    client = openai.OpenAI(api_key=API_KEY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    done = load_progress()
    total = len(CHARTS)
    generated = 0
    skipped = 0
    errors = 0

    print(f"\nGenerating {total} charts for Week 11 slides...\n")

    for slide_num in sorted(CHARTS.keys()):
        name_prefix, prompt = CHARTS[slide_num]
        filename = f"{name_prefix}.png"
        out_path = OUT_DIR / filename

        if out_path.exists() and slide_num in done:
            print(f"  ⏭  Slide {slide_num:02d} — {filename} already exists, skipping")
            skipped += 1
            continue

        print(f"  🎨 Slide {slide_num:02d} — {name_prefix}...", flush=True)
        try:
            resp = client.images.generate(
                model=IMG_MODEL,
                prompt=prompt,
                size=IMG_SIZE,
                quality=IMG_QUAL,
                n=1,
            )
            item = resp.data[0]
            if getattr(item, "b64_json", None):
                out_path.write_bytes(base64.b64decode(item.b64_json))
            elif getattr(item, "url", None):
                import urllib.request
                urllib.request.urlretrieve(item.url, out_path)
            else:
                raise ValueError("No image data in response")

            done.add(slide_num)
            save_progress(done)
            size_kb = out_path.stat().st_size // 1024
            print(f"     ✓ saved {filename} ({size_kb} KB)")
            generated += 1

        except Exception as exc:
            print(f"     ✗ ERROR on slide {slide_num}: {exc}")
            errors += 1

        if slide_num != sorted(CHARTS.keys())[-1]:
            time.sleep(3)

    print(f"\n{'─' * 60}")
    print(f"  Done — generated: {generated}, skipped: {skipped}, errors: {errors}")
    print(f"  Charts saved to: {OUT_DIR}")
    if generated + skipped == total:
        print(f"\n  All {total} charts ready. Run build_pptx.py to embed them.")


if __name__ == "__main__":
    generate_charts()
