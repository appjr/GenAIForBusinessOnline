#!/usr/bin/env python3
"""
Generate chart images for Week 11 slides using Python / matplotlib.

Produces clean, professional data visualizations matching the course
color scheme (navy #16213e, teal #0ea5e9, purple #533483, amber #f59e0b).

Saves PNGs to Class 11/charts/slide_NN_chart.png (overwrites AI images).

Usage:
    pip install matplotlib numpy
    python3 generate_charts_python.py
"""

from pathlib import Path
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

OUT_DIR = Path(__file__).parent / "charts"
OUT_DIR.mkdir(exist_ok=True)

# ── Course palette ─────────────────────────────────────────────────────────────
NAVY    = "#16213e"
NAVY2   = "#0f3460"
TEAL    = "#0ea5e9"
PURPLE  = "#533483"
AMBER   = "#f59e0b"
GREEN   = "#22c55e"
WHITE   = "#ffffff"
LGRAY   = "#f1f5f9"
MGRAY   = "#94a3b8"
DGRAY   = "#334155"

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  ✓ {name}  ({path.stat().st_size // 1024} KB)")


# ── Slide 3: Five-Layer AI Ecosystem Stack ─────────────────────────────────────
def slide_03():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    layers = [
        (0, NAVY,   "Layer 1 — Foundation Models",
         "GPT-4.1  ·  Claude 3.7  ·  Gemini 2.5  ·  Llama 4  ·  DeepSeek V3"),
        (1, NAVY2,  "Layer 2 — APIs & SDKs",
         "Anthropic API  ·  OpenAI API  ·  Google AI Studio  ·  Hugging Face"),
        (2, PURPLE, "Layer 3 — Orchestration Frameworks",
         "LangChain  ·  LangGraph  ·  CrewAI  ·  Claude Agent SDK  ·  OpenAI Agents SDK"),
        (3, TEAL,   "Layer 4 — Applications & Products",
         "RAG Systems  ·  AI Agents  ·  Workflow Automation  ·  AI Coding Tools"),
        (4, GREEN,  "Layer 5 — Business Value",
         "Cost Reduction  ·  Revenue Growth  ·  Speed to Market  ·  Competitive Advantage"),
    ]

    for i, (row, color, title, content) in enumerate(layers):
        # Main band
        rect = FancyBboxPatch((0.3, row * 0.92 + 0.08), 9.4, 0.80,
                               boxstyle="round,pad=0.02",
                               facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        # Left accent stripe
        stripe = FancyBboxPatch((0.3, row * 0.92 + 0.08), 0.22, 0.80,
                                 boxstyle="round,pad=0.0",
                                 facecolor=AMBER if i == 4 else WHITE,
                                 edgecolor="none", alpha=0.35)
        ax.add_patch(stripe)
        ax.text(0.72, row * 0.92 + 0.52, title,
                fontsize=11, fontweight="bold", color=WHITE, va="center")
        ax.text(0.72, row * 0.92 + 0.26, content,
                fontsize=8.5, color=WHITE, alpha=0.88, va="center")

    # Upward arrows between layers
    for i in range(4):
        ax.annotate("", xy=(9.85, (i + 1) * 0.92 + 0.08),
                    xytext=(9.85, i * 0.92 + 0.88),
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=2))

    # Title
    ax.text(5, 4.72, "The Five-Layer AI Ecosystem",
            fontsize=16, fontweight="bold", color=NAVY, ha="center", va="center")
    ax.text(0.1, 2.3, "Abstraction\nLevel", fontsize=9, color=MGRAY,
            ha="center", va="center", rotation=90)

    save(fig, "slide_03_chart.png")


# ── Slide 4: Cost vs Capability Bubble Chart ───────────────────────────────────
def slide_04():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    models = [
        ("GPT-4.1",          2.0,  95, 128,  NAVY,   "proprietary"),
        ("Claude 3.7 Sonnet", 3.0, 94, 200,  TEAL,   "proprietary"),
        ("Gemini 2.5 Pro",   7.0,  93, 1000, PURPLE, "proprietary"),
        ("Claude Haiku",     0.25, 78,  32,  "#38bdf8", "proprietary"),
        ("GPT-4.1 mini",     0.30, 75,  64,  "#6366f1", "proprietary"),
        ("DeepSeek V3",      0.27, 85,  64,  AMBER,  "open-source"),
        ("Llama 4 Scout",    0.05, 72,  10,  "#f97316", "open-source"),
    ]

    for name, cost, cap, ctx, color, kind in models:
        size = (ctx ** 0.55) * 18
        marker = "*" if kind == "open-source" else "o"
        ax.scatter(cost, cap, s=size, color=color, alpha=0.82,
                   marker=marker, edgecolors="white", linewidths=1.5, zorder=3)
        offset_x = 0.15 if cost < 1 else 0.25
        offset_y = 1.2 if name not in ("GPT-4.1 mini", "Claude Haiku") else -2.2
        label = f"{name}\n${cost}/M"
        ax.annotate(label, (cost, cap), fontsize=8, color=DGRAY, fontweight="bold",
                    xytext=(cost + offset_x, cap + offset_y),
                    arrowprops=dict(arrowstyle="-", color=MGRAY, lw=0.8))

    # Sweet-spot shading
    sweet = plt.Rectangle((0, 55), 2.5, 35, color=GREEN, alpha=0.08, zorder=0)
    ax.add_patch(sweet)
    ax.text(1.2, 57, "Sweet spot:\nHigh capability, low cost",
            fontsize=8, color=GREEN, alpha=0.9, ha="center")

    ax.set_xscale("log")
    ax.set_xlabel("Cost per Million Tokens (USD) — log scale",
                  fontsize=11, color=DGRAY)
    ax.set_ylabel("Capability Score (benchmark composite)", fontsize=11, color=DGRAY)
    ax.set_title("Frontier AI Models: Cost vs. Capability (2026)",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlim(0.03, 12)
    ax.set_ylim(60, 100)
    ax.grid(True, alpha=0.35, color="white")

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MGRAY,
               markersize=9, label="Proprietary"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=MGRAY,
               markersize=12, label="Open Source"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MGRAY,
               markersize=6, label="Bubble size = context window"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9,
              framealpha=0.9)

    save(fig, "slide_04_chart.png")


# ── Slide 5: Training Cost Collapse ───────────────────────────────────────────
def slide_05():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    data = [
        ("GPT-3\n(2020)",       12_000_000,  NAVY,  "proprietary"),
        ("GPT-4\n(2023)",      100_000_000,  NAVY,  "proprietary"),
        ("LLaMA 1\n(2023)",      3_000_000,  AMBER, "open-source"),
        ("Mistral 7B\n(2023)",     600_000,  AMBER, "open-source"),
        ("LLaMA 2\n(2023)",      5_000_000,  AMBER, "open-source"),
        ("DeepSeek V3\n(2024)",  6_000_000,  AMBER, "open-source"),
        ("DeepSeek R1\n(2025)",  5_600_000,  AMBER, "open-source"),
        ("Llama 4\n(2026)",      8_000_000,  AMBER, "open-source"),
    ]

    labels = [d[0] for d in data]
    values = [d[1] for d in data]
    colors = [d[2] for d in data]
    x = np.arange(len(labels))

    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.5,
                  width=0.65, zorder=3)

    for bar, val in zip(bars, values):
        if val >= 1_000_000:
            label = f"${val/1_000_000:.0f}M"
        else:
            label = f"${val/1_000:.0f}K"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.15,
                label, ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=DGRAY)

    ax.set_yscale("log")
    ax.set_ylim(200_000, 500_000_000)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Estimated Training Cost (USD) — log scale",
                  fontsize=11, color=DGRAY)
    ax.set_title("AI Training Cost Collapse: Open Source vs. Proprietary (2020–2026)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"${v/1e6:.0f}M" if v >= 1e6 else f"${v/1e3:.0f}K"))
    ax.grid(axis="y", alpha=0.35, color="white", zorder=0)

    # Annotation arrow
    ax.annotate("Open-source models: GPT-4 equivalent quality\nat <10% the cost (2023→2026)",
                xy=(5.5, 7_000_000), xytext=(5.5, 50_000_000),
                fontsize=9, color=TEAL, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))

    legend_elements = [
        mpatches.Patch(facecolor=NAVY, label="Proprietary"),
        mpatches.Patch(facecolor=AMBER, label="Open Source"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

    save(fig, "slide_05_chart.png")


# ── Slide 6: Model Selection Flowchart ─────────────────────────────────────────
def slide_06():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    def diamond(cx, cy, w, h, color, label, fontsize=8.5):
        dx, dy = w / 2, h / 2
        xs = [cx, cx + dx, cx, cx - dx, cx]
        ys = [cy + dy, cy, cy - dy, cy, cy + dy]
        ax.fill(xs, ys, color=color, zorder=3, alpha=0.92)
        ax.plot(xs, ys, color="white", lw=1.5, zorder=4)
        for i, line in enumerate(label.split("\n")):
            ax.text(cx, cy + 0.12 * (len(label.split("\n")) - 1) / 2 - i * 0.22,
                    line, ha="center", va="center", fontsize=fontsize,
                    color=WHITE, fontweight="bold", zorder=5)

    def box(cx, cy, w, h, color, label, fontsize=8.5):
        rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                               boxstyle="round,pad=0.08",
                               facecolor=color, edgecolor="white", linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        for i, line in enumerate(label.split("\n")):
            offset = 0.18 * (len(label.split("\n")) - 1) / 2 - i * 0.18
            ax.text(cx, cy + offset, line, ha="center", va="center",
                    fontsize=fontsize, color=WHITE, fontweight="bold", zorder=4)

    def arrow(x1, y1, x2, y2, label="", color=MGRAY):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.12, my, label, fontsize=8, color=color, fontweight="bold")

    # Title
    ax.text(6, 7.7, "AI Model Selection Framework",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)

    # Root diamond
    diamond(6, 6.8, 3.6, 0.9, NAVY2, "Does data need\nto stay private?")

    # YES branch (left)
    arrow(4.2, 6.8, 2.8, 6.8, "YES", "#ef4444")
    diamond(2.0, 6.0, 2.8, 0.85, NAVY2, "On-premise\nhardware available?")
    arrow(2.0, 5.57, 2.0, 4.85, "YES", GREEN)
    box(2.0, 4.45, 3.0, 0.72, GREEN, "Self-hosted open source\nLlama 4 · Phi-4 · Mistral")
    arrow(2.0, 6.0, 0.85, 5.2, "NO", "#ef4444")
    box(0.5, 4.8, 1.6, 0.72, NAVY2, "Private cloud\nBedrock · Azure · Vertex")

    # NO branch (right)
    arrow(7.8, 6.8, 9.2, 6.8, "NO", GREEN)
    diamond(10.0, 6.0, 3.0, 0.85, NAVY2, "Need maximum\ncapability?")
    arrow(10.0, 5.57, 10.0, 4.85, "YES", GREEN)
    diamond(10.0, 4.45, 2.8, 0.78, NAVY2, "Cost sensitive?")
    arrow(10.0, 4.06, 10.0, 3.35, "NO", GREEN)
    box(10.0, 2.95, 3.0, 0.72, PURPLE, "Frontier model\nClaude 3.7 · GPT-4.1 · Gemini 2.5")
    arrow(8.6, 4.45, 7.5, 3.9, "YES", AMBER)
    box(6.8, 3.55, 2.8, 0.68, TEAL, "GPT-4.1 mini\nor Claude Haiku\n$0.25–0.30/M")

    # Volume branch
    arrow(10.0, 6.0, 11.3, 5.2, "NO", "#94a3b8")
    diamond(11.5, 4.75, 2.4, 0.72, NAVY2, "Volume\n>10M/month?")
    arrow(11.5, 4.39, 11.5, 3.7, "YES", AMBER)
    box(11.5, 3.3, 2.2, 0.68, AMBER, "Fine-tune a\nsmall model\nLoRA on Llama 4 8B")
    arrow(10.3, 4.75, 9.2, 4.1, "NO", MGRAY)
    box(8.7, 3.75, 2.4, 0.62, TEAL, "GPT-4.1 nano\nor Haiku\n$0.05/M")

    save(fig, "slide_06_chart.png")


# ── Slide 7: Chatbot vs Agent Diagram ─────────────────────────────────────────
def slide_07():
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # Divider
    ax.axvline(x=6, color=LGRAY, lw=2, ymin=0.05, ymax=0.95)

    # ── LEFT: Chatbot ──────────────────────────────────────────────────────────
    ax.text(3, 6.1, "Chatbot  (2022–2024)", ha="center", fontsize=13,
            fontweight="bold", color=NAVY)

    def simple_box(cx, cy, w, h, color, label, tc=WHITE, fs=10):
        r = FancyBboxPatch((cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor="white", lw=1.5, zorder=3)
        ax.add_patch(r)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold", zorder=4)

    simple_box(3, 5.1, 2.4, 0.62, NAVY2, "User Input")
    ax.annotate("", xy=(3, 4.2), xytext=(3, 4.78),
                arrowprops=dict(arrowstyle="->", color=MGRAY, lw=2))
    simple_box(3, 3.75, 2.4, 0.82, TEAL, "LLM")
    ax.annotate("", xy=(3, 2.88), xytext=(3, 3.34),
                arrowprops=dict(arrowstyle="->", color=MGRAY, lw=2))
    simple_box(3, 2.48, 2.4, 0.62, NAVY2, "Text Output")

    ax.text(3, 1.7, "One round trip.", ha="center", fontsize=9.5, color=DGRAY)
    ax.text(3, 1.35, "Human drives every step.", ha="center", fontsize=9.5, color=DGRAY)

    # ── RIGHT: Agent loop ──────────────────────────────────────────────────────
    ax.text(9, 6.1, "AI Agent  (2025–2026)", ha="center", fontsize=13,
            fontweight="bold", color=NAVY)

    # Circular loop: 5 nodes
    cx, cy, r = 9, 3.5, 2.0
    angles = [90, 18, -54, -126, -198]
    node_labels = ["Receive\nTask", "Plan\nSteps", "Call\nTool",
                   "Observe\nResult", "Reason\n& Adapt"]
    node_colors = [NAVY, TEAL, PURPLE, AMBER, TEAL]

    node_positions = []
    for angle in angles:
        rad = np.radians(angle)
        nx, ny = cx + r * np.cos(rad), cy + r * np.sin(rad)
        node_positions.append((nx, ny))

    # Draw arrows between nodes (clockwise = decreasing angle order)
    for i in range(5):
        x1, y1 = node_positions[i]
        x2, y2 = node_positions[(i + 1) % 5]
        # Midpoint + slight curve via connectionstyle
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=MGRAY, lw=2,
                        connectionstyle="arc3,rad=0.22"
                    ))

    # Tool icons on "Call Tool" node
    ti = node_positions[2]
    ax.text(ti[0] + 0.7, ti[1] - 0.3, "[web · code · DB]", fontsize=8,
            ha="left", va="center", color=PURPLE, style="italic")

    # Draw nodes on top
    for i, (nx, ny) in enumerate(node_positions):
        circle = plt.Circle((nx, ny), 0.55, color=node_colors[i],
                              zorder=5, ec="white", lw=2)
        ax.add_patch(circle)
        for j, line in enumerate(node_labels[i].split("\n")):
            ax.text(nx, ny + 0.11 - j * 0.22, line, ha="center", va="center",
                    fontsize=8.5, color=WHITE, fontweight="bold", zorder=6)

    # Center label
    ax.text(cx, cy, "Goal:\nComplete\nTask", ha="center", va="center",
            fontsize=9, color=NAVY, fontweight="bold")

    # Callout
    cb = FancyBboxPatch((6.4, 0.55), 5.2, 0.7,
                         boxstyle="round,pad=0.1",
                         facecolor=AMBER, edgecolor="none", alpha=0.18, zorder=2)
    ax.add_patch(cb)
    ax.text(9, 0.9, "4-hour analyst task  →  15 minutes",
            ha="center", va="center", fontsize=11, fontweight="bold",
            color=AMBER)

    ax.text(9, 0.35, "Autonomous · Low supervision · Returns results, not text",
            ha="center", fontsize=9, color=DGRAY)

    save(fig, "slide_07_chart.png")


# ── Slide 14: Developer Productivity ──────────────────────────────────────────
def slide_14():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    tasks = [
        ("Write boilerplate / scaffolding", 90, AMBER),
        ("Write new function", 7, TEAL),
        ("Write unit tests", 7, TEAL),
        ("Understand unfamiliar codebase", 5, NAVY2),
        ("Fix a bug", 4, NAVY2),
        ("Code review coverage +35%", 3.5, PURPLE),
    ]

    labels = [t[0] for t in tasks]
    values = [t[1] for t in tasks]
    colors = [t[2] for t in tasks]
    y = np.arange(len(labels))

    bars = ax.barh(y, values, color=colors, edgecolor="white",
                   linewidth=1.5, height=0.62, zorder=3)

    for i, (bar, val, task) in enumerate(zip(bars, values, tasks)):
        suffix = "×" if task[0] != "Code review coverage +35%" else ""
        ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{val}{suffix}", va="center", fontsize=11,
                fontweight="bold", color=tasks[i][2])

    # Special annotation on 90x bar
    ax.annotate("Biggest gain:\nboilerplate eliminated",
                xy=(90, 5), xytext=(60, 4.6),
                fontsize=8.5, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Speedup Factor (×)", fontsize=11, color=DGRAY)
    ax.set_title("AI Coding Tools: Developer Productivity Gains",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlim(0, 102)
    ax.grid(axis="x", alpha=0.35, color="white", zorder=0)
    ax.text(0.98, -0.08, "Source: GitHub & StackOverflow Developer Surveys 2025–2026",
            transform=ax.transAxes, fontsize=7.5, color=MGRAY, ha="right")

    save(fig, "slide_14_chart.png")


# ── Slide 20: AI Automation Time Saved ────────────────────────────────────────
def slide_20():
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    processes = [
        ("Invoice processing\n(extract, validate, route)", 97, "5 min → 10 sec",   NAVY),
        ("Contract risk review\n(50-page document)",        97, "2–3 hrs → 3 min",  NAVY),
        ("Competitive intelligence\nreport (weekly)",       94, "6 hrs → 20 min",   TEAL),
        ("Meeting transcript\n→ action items",              93, "30 min → 2 min",   TEAL),
        ("Customer support ticket\ntriage + draft reply",   91, "8 min → 45 sec",   PURPLE),
    ]

    labels = [p[0] for p in processes]
    values = [p[1] for p in processes]
    detail = [p[2] for p in processes]
    colors = [p[3] for p in processes]
    y = np.arange(len(labels))

    bars = ax.barh(y, values, color=colors, edgecolor="white",
                   linewidth=1.5, height=0.65, zorder=3)

    for bar, val, det in zip(bars, values, detail):
        ax.text(val - 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val}%", va="center", ha="right", fontsize=12,
                fontweight="bold", color=WHITE, zorder=4)
        ax.text(101, bar.get_y() + bar.get_height() / 2,
                det, va="center", fontsize=9, color=DGRAY)

    # 90% threshold line
    ax.axvline(x=90, color=GREEN, lw=2, linestyle="--", alpha=0.8, zorder=2)
    ax.text(90.5, -0.55, "90%\nthreshold", fontsize=8, color=GREEN,
            va="bottom", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("% Time Saved", fontsize=11, color=DGRAY)
    ax.set_title("AI Automation: Time Saved by Business Process",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlim(0, 145)
    ax.set_xticks(range(0, 101, 10))
    ax.grid(axis="x", alpha=0.35, color="white", zorder=0)

    # Callout
    ax.text(72, -0.9, "Avg. cost per automated task: <$0.10",
            fontsize=10, color=AMBER, fontweight="bold", ha="center")

    save(fig, "slide_20_chart.png")


# ── Slide 25: Industry Impact ─────────────────────────────────────────────────
def slide_25():
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    categories = {
        "Finance": (NAVY,
            [("Earnings call analysis\n2 hrs → 8 min", 93),
             ("Regulatory gap analysis\ndays → hours", 85)]),
        "Healthcare": (TEAL,
            [("Physician documentation\n−2 hrs/day", 33),
             ("Prior auth letter\n45 min → 4 min", 91)]),
        "Retail": (PURPLE,
            [("Product descriptions\n10,000 SKUs/hr vs 100/day", 99),
             ("Chat resolution w/o\nhuman agent", 70)]),
        "Consulting": (AMBER,
            [("RFP section drafting\n30 min vs 1–2 days", 96),
             ("Contract risk flagging\n3 min vs 2 hrs", 97)]),
    }

    n_industries = len(categories)
    n_metrics = 2
    group_w = 0.7
    bar_w = group_w / n_metrics
    x_positions = np.arange(n_industries)

    for i, (industry, (color, metrics)) in enumerate(categories.items()):
        for j, (label, pct) in enumerate(metrics):
            x = i + (j - 0.5) * bar_w * 1.1
            bar = ax.bar(x, pct, width=bar_w * 0.88,
                         color=color, alpha=0.75 + j * 0.2,
                         edgecolor="white", linewidth=1.5, zorder=3)
            ax.text(x, pct + 1.5, f"{pct}%",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold", color=color)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(categories.keys()), fontsize=13, fontweight="bold")
    ax.set_ylabel("% Improvement / Time Reduction", fontsize=11, color=DGRAY)
    ax.set_title("AI Impact Across Industries: Measured Results (2025–2026)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=12)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.35, color="white", zorder=0)

    # Industry labels with metric names below chart
    for i, (industry, (color, metrics)) in enumerate(categories.items()):
        for j, (label, _) in enumerate(metrics):
            x = i + (j - 0.5) * bar_w * 1.1
            ax.text(x, -8, label, ha="center", va="top", fontsize=7,
                    color=DGRAY, wrap=True)

    ax.text(0.5, -0.17,
            "Consistent pattern: AI absorbs 60–80% of information-gathering and drafting work",
            transform=ax.transAxes, fontsize=10, color=DGRAY,
            ha="center", style="italic")

    save(fig, "slide_25_chart.png")


# ── Slide 28: Three-Tier Adoption Pyramid ─────────────────────────────────────
def slide_28():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(5, 6.65, "AI Adoption Maturity Model",
            ha="center", fontsize=16, fontweight="bold", color=NAVY)

    tiers = [
        # (y_center, half_width, color, tier_label, tools, cost, timeline)
        (1.2, 4.5, NAVY,   "Tier 1: Individual Productivity",
         "Claude / ChatGPT · Cursor / Copilot · Perplexity",
         "$0–$50/person/month", "Start now"),
        (3.1, 3.2, NAVY2,  "Tier 2: Team Workflows",
         "Zapier / Make / n8n · NotebookLM · RAG chatbot",
         "$100–$1,000/month", "Months 1–6"),
        (5.0, 1.8, PURPLE, "Tier 3: Enterprise AI",
         "Fine-tuned models · Multi-agent systems",
         "$10,000+/month", "Months 6–18"),
    ]

    for y, hw, color, title, tools, cost, timeline in tiers:
        xs = [5 - hw, 5 + hw, 5 + hw, 5 - hw]
        ys = [y - 0.78, y - 0.78, y + 0.78, y + 0.78]
        ax.fill(xs, ys, color=color, zorder=3, alpha=0.92)
        ax.plot(xs + [xs[0]], ys + [ys[0]], color="white", lw=2, zorder=4)

        ax.text(5, y + 0.38, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=WHITE, zorder=5)
        ax.text(5, y + 0.0, tools, ha="center", va="center",
                fontsize=8.5, color=WHITE, alpha=0.9, zorder=5)

        # Cost badge
        cb = FancyBboxPatch((5 + hw + 0.15, y - 0.28), 2.5, 0.52,
                             boxstyle="round,pad=0.06",
                             facecolor=AMBER, edgecolor="none", alpha=0.9, zorder=5)
        ax.add_patch(cb)
        ax.text(5 + hw + 1.4, y, cost, ha="center", va="center",
                fontsize=8, fontweight="bold", color=NAVY, zorder=6)

        # Timeline badge
        tb = FancyBboxPatch((5 - hw - 2.65, y - 0.28), 2.45, 0.52,
                             boxstyle="round,pad=0.06",
                             facecolor=TEAL, edgecolor="none", alpha=0.9, zorder=5)
        ax.add_patch(tb)
        ax.text(5 - hw - 1.42, y, timeline, ha="center", va="center",
                fontsize=8, fontweight="bold", color=WHITE, zorder=6)

    # Arrows on sides
    ax.annotate("", xy=(0.2, 5.7), xytext=(0.2, 0.4),
                arrowprops=dict(arrowstyle="->", color=MGRAY, lw=2))
    ax.text(0.08, 3.1, "Complexity &\nInvestment", ha="center", va="center",
            fontsize=8.5, color=MGRAY, rotation=90)

    ax.annotate("", xy=(9.8, 5.7), xytext=(9.8, 0.4),
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=2))
    ax.text(9.92, 3.1, "Business\nImpact & Scale", ha="center", va="center",
            fontsize=8.5, color=TEAL, rotation=90)

    # Bottom callout
    ax.text(5, 0.22, "Start at Tier 1. Prove value. Then invest upward.",
            ha="center", fontsize=10.5, color=AMBER, fontweight="bold")

    save(fig, "slide_28_chart.png")


# ── Slide 29: Six Trends Radar Chart ──────────────────────────────────────────
def slide_29():
    categories = [
        "Intelligence\nCost Collapse",
        "Agentic AI\nProduction",
        "Domain\nSpecialization",
        "Multimodal\nBaseline",
        "AI Governance\n& Compliance",
        "Human Role\nShift",
    ]
    scores_2024 = [5, 4, 4, 5, 3, 6]
    scores_2026 = [9, 8, 7, 8, 6, 9]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    s24 = scores_2024 + scores_2024[:1]
    s26 = scores_2026 + scores_2026[:1]

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 10)
    ax.set_rticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=8, color=MGRAY)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10, color=NAVY, fontweight="bold")
    ax.tick_params(pad=12)
    ax.grid(color="white", lw=1.2, alpha=0.6)
    ax.spines["polar"].set_visible(False)

    ax.plot(angles, s24, color=NAVY, lw=2.5, linestyle="--", zorder=3)
    ax.fill(angles, s24, color=NAVY, alpha=0.15, zorder=2)

    ax.plot(angles, s26, color=TEAL, lw=2.5, zorder=4)
    ax.fill(angles, s26, color=TEAL, alpha=0.30, zorder=3)

    for angle, val in zip(angles[:-1], scores_2026):
        x = angle
        ax.plot(x, val, "o", color=TEAL, ms=7, zorder=5)

    ax.set_title("Six AI Mega-Trends: Impact Assessment (2024 vs 2026)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=22, y=1.08)

    legend_elements = [
        Line2D([0], [0], color=NAVY, lw=2.5, linestyle="--", label="2024 State"),
        Line2D([0], [0], color=TEAL, lw=2.5, label="2026 State"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              bbox_to_anchor=(1.28, -0.05), fontsize=10, framealpha=0.9)
    fig.text(0.5, 0.01, "Shaded area growth = acceleration of each trend 2024 → 2026",
             ha="center", fontsize=9, color=MGRAY, style="italic")

    save(fig, "slide_29_chart.png")


# ── Slide 30: Career Salary Ranges ────────────────────────────────────────────
def slide_30():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    roles = [
        ("AI Product Manager",          140, 210, NAVY),
        ("ML Ops / AI Ops Engineer",    130, 185, NAVY2),
        ("AI Automation Engineer",      100, 165, TEAL),
        ("Prompt / LLM Engineer",        90, 155, TEAL),
        ("AI Governance Analyst",        90, 145, PURPLE),
        ("Business AI Analyst",          80, 135, PURPLE),
        ("AI Data Curator",              80, 135, AMBER),
    ]

    y = np.arange(len(roles))

    for i, (role, lo, hi, color) in enumerate(roles):
        # Range bar
        ax.barh(i, hi - lo, left=lo, height=0.55, color=color,
                alpha=0.82, edgecolor="white", linewidth=1.5, zorder=3)
        # Midpoint dot
        mid = (lo + hi) / 2
        ax.plot(mid, i, "o", color="white", ms=8, zorder=5)
        ax.plot(mid, i, "o", color=color, ms=5, zorder=6)
        # Label
        ax.text(hi + 2, i, f"${lo}K–${hi}K",
                va="center", fontsize=10, fontweight="bold", color=color)

    # $100K threshold
    ax.axvline(x=100, color=GREEN, lw=2, linestyle="--", alpha=0.9, zorder=2)
    ax.text(100.5, -0.65, "$100K\nthreshold", fontsize=8,
            color=GREEN, fontweight="bold", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in roles], fontsize=10.5)
    ax.set_xlabel("Annual Salary (USD thousands)", fontsize=11, color=DGRAY)
    ax.set_title("Emerging AI Career Roles: Salary Ranges (US, 2026)",
                 fontsize=14, fontweight="bold", color=NAVY, pad=12)
    ax.set_xlim(60, 240)
    ax.set_xticks(range(60, 231, 20))
    ax.set_xticklabels([f"${v}K" for v in range(60, 231, 20)], fontsize=9)
    ax.grid(axis="x", alpha=0.35, color="white", zorder=0)

    ax.text(0.98, -0.10,
            "Source: LinkedIn Salary Insights · Levels.fyi · 2026 estimates",
            transform=ax.transAxes, fontsize=7.5, color=MGRAY, ha="right")
    ax.text(150, -0.82,
            "All roles above $80K baseline — new field, premium compensation",
            fontsize=9.5, color=AMBER, fontweight="bold", ha="center")

    save(fig, "slide_30_chart.png")


# ── Run all ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating 11 Python charts → {OUT_DIR}\n")
    slide_03()
    slide_04()
    slide_05()
    slide_06()
    slide_07()
    slide_14()
    slide_20()
    slide_25()
    slide_28()
    slide_29()
    slide_30()
    print("\n✅  All charts generated.")
