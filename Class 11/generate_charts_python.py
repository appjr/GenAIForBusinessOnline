#!/usr/bin/env python3
"""
Generate chart images for Week 11 slides — clean, no-overlap versions.
Saves PNGs to Class 11/charts/slide_NN_chart.png

Usage:
    pip install matplotlib numpy
    python3 generate_charts_python.py
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker

OUT_DIR = Path(__file__).parent / "charts"
OUT_DIR.mkdir(exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
NAVY   = "#16213e"
NAVY2  = "#0f3460"
TEAL   = "#0ea5e9"
PURPLE = "#533483"
AMBER  = "#f59e0b"
GREEN  = "#22c55e"
RED    = "#ef4444"
WHITE  = "#ffffff"
LGRAY  = "#f1f5f9"
MGRAY  = "#94a3b8"
DGRAY  = "#334155"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    print(f"  ✓ {name}  ({p.stat().st_size // 1024} KB)")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 3: Five-Layer AI Ecosystem Stack
# ─────────────────────────────────────────────────────────────────────────────
def slide_03():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.2, 5.6)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    layers = [
        (NAVY,   "Layer 1  —  Foundation Models",
         "GPT-4.1  ·  Claude 3.7  ·  Gemini 2.5  ·  Llama 4  ·  DeepSeek V3"),
        (NAVY2,  "Layer 2  —  APIs & SDKs",
         "Anthropic API  ·  OpenAI API  ·  Google AI Studio  ·  Hugging Face"),
        (PURPLE, "Layer 3  —  Orchestration Frameworks",
         "LangChain  ·  LangGraph  ·  CrewAI  ·  Claude Agent SDK"),
        (TEAL,   "Layer 4  —  Applications",
         "RAG Systems  ·  AI Agents  ·  Workflow Automation  ·  AI Coding Tools"),
        (GREEN,  "Layer 5  —  Business Value",
         "Cost Reduction  ·  Revenue Growth  ·  Competitive Advantage"),
    ]

    for i, (color, title, content) in enumerate(layers):
        y = i * 1.02
        rect = FancyBboxPatch((0.3, y + 0.06), 8.4, 0.88,
                               boxstyle="round,pad=0.04",
                               facecolor=color, edgecolor=WHITE, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.75, y + 0.62, title, fontsize=11.5, fontweight="bold",
                color=WHITE, va="center")
        ax.text(0.75, y + 0.28, content, fontsize=9, color=WHITE,
                alpha=0.9, va="center")

    # Upward arrows on right
    for i in range(4):
        ax.annotate("", xy=(8.85, (i + 1) * 1.02 + 0.1),
                    xytext=(8.85, i * 1.02 + 0.94),
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.2))

    ax.text(4.5, 5.35, "The Five-Layer AI Ecosystem",
            fontsize=15, fontweight="bold", color=NAVY, ha="center")
    ax.text(0.15, 2.55, "Abstraction Level  ↑", fontsize=8.5,
            color=MGRAY, ha="center", va="center", rotation=90)

    save(fig, "slide_03_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 4: Cost vs Capability — clean scatter, no overlapping labels
# ─────────────────────────────────────────────────────────────────────────────
def slide_04():
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    # (name, cost, capability, color, open_source, offset_pts_x, offset_pts_y)
    # Offsets are in POINTS (textcoords="offset points") — safe on log scale
    models = [
        ("GPT-4.1",           2.0,  95, NAVY,      False,   8,   8),
        ("Claude 3.7",        3.0,  94, TEAL,      False,  -70, -14),
        ("Gemini 2.5 Pro",    7.0,  93, PURPLE,    False,   8,   8),
        ("DeepSeek V3 *",     0.27, 85, AMBER,     True,    8,   8),
        ("Claude Haiku",      0.25, 78, "#38bdf8",  False,  8,  -16),
        ("GPT-4.1 mini",      0.30, 75, "#6366f1",  False,  8,   8),
        ("Llama 4 Scout *",   0.05, 72, "#f97316",  True,   8,   8),
    ]

    for name, cost, cap, color, oss, opx, opy in models:
        marker = "D" if oss else "o"
        ms = 160 if oss else 120
        ax.scatter(cost, cap, s=ms, color=color, marker=marker,
                   edgecolors=WHITE, linewidths=1.5, zorder=4)
        ax.annotate(name, (cost, cap),
                    xytext=(opx, opy), textcoords="offset points",
                    fontsize=8.5, fontweight="bold", color=color, zorder=5)

    # Sweet-spot box
    ax.axvspan(0.01, 1.0, alpha=0.06, color=GREEN, zorder=0)
    ax.axhspan(80, 100, alpha=0.06, color=GREEN, zorder=0)
    ax.text(0.55, 96.5, "Best value zone", fontsize=8, color=GREEN,
            fontweight="bold", ha="center")

    ax.set_xscale("log")
    ax.set_xlim(0.02, 15)
    ax.set_ylim(65, 100)
    ax.set_xlabel("Cost per Million Tokens (USD) — log scale", fontsize=11, color=DGRAY)
    ax.set_ylabel("Capability Score", fontsize=11, color=DGRAY)
    ax.set_title("Frontier AI Models: Cost vs. Capability (2026)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax.grid(True, alpha=0.3, color=WHITE)

    legend = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=DGRAY,
               markersize=9, label="Proprietary"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor=DGRAY,
               markersize=8, label="Open Source  (*)"),
    ]
    ax.legend(handles=legend, fontsize=9, loc="lower right", framealpha=0.9)
    ax.text(0.02, -0.1, "* Open-source models — free to self-host",
            transform=ax.transAxes, fontsize=8, color=MGRAY)

    save(fig, "slide_04_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 5: Training Cost Collapse
# ─────────────────────────────────────────────────────────────────────────────
def slide_05():
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    data = [
        ("GPT-3\n(2020)",      12e6,  NAVY,  "Prop."),
        ("GPT-4\n(2023)",     100e6,  NAVY,  "Prop."),
        ("LLaMA 1\n(2023)",    3e6,   AMBER, "OSS"),
        ("Mistral 7B\n(2023)", 0.6e6, AMBER, "OSS"),
        ("DeepSeek V3\n(2024)",6e6,   AMBER, "OSS"),
        ("DeepSeek R1\n(2025)",5.6e6, AMBER, "OSS"),
        ("Llama 4\n(2026)",    8e6,   AMBER, "OSS"),
    ]

    x = np.arange(len(data))
    bars = ax.bar(x, [d[1] for d in data],
                  color=[d[2] for d in data],
                  edgecolor=WHITE, linewidth=1.5, width=0.62, zorder=3)

    for bar, (label, val, *_) in zip(bars, data):
        lbl = f"${val/1e6:.0f}M" if val >= 1e6 else f"${val/1e3:.0f}K"
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() * 1.25, lbl,
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=DGRAY)

    ax.set_yscale("log")
    ax.set_ylim(3e5, 8e8)
    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data], fontsize=9)
    ax.set_ylabel("Training Cost (USD) — log scale", fontsize=11, color=DGRAY)
    ax.set_title("AI Training Cost Collapse (2020–2026)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda v, _: f"${v/1e6:.0f}M" if v >= 1e6 else f"${v/1e3:.0f}K"))
    ax.grid(axis="y", alpha=0.3, color=WHITE, zorder=0)

    ax.annotate("Open-source cost\n≈ 6% of GPT-4",
                xy=(4, 6e6), xytext=(4.4, 2e7),
                fontsize=9, color=TEAL, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=1.5))

    ax.legend(handles=[
        mpatches.Patch(facecolor=NAVY,  label="Proprietary"),
        mpatches.Patch(facecolor=AMBER, label="Open Source"),
    ], loc="upper left", fontsize=10)

    save(fig, "slide_05_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 6: Model Selection — simplified decision tree
# ─────────────────────────────────────────────────────────────────────────────
def slide_06():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    def diamond(cx, cy, w, h, color, lines):
        dx, dy = w/2, h/2
        xs = [cx, cx+dx, cx, cx-dx, cx]
        ys = [cy+dy, cy, cy-dy, cy, cy+dy]
        ax.fill(xs, ys, color=color, zorder=3, alpha=0.93)
        ax.plot(xs, ys, color=WHITE, lw=2, zorder=4)
        for j, ln in enumerate(lines):
            off = 0.17 * (len(lines)-1)/2 - j*0.17
            ax.text(cx, cy+off, ln, ha="center", va="center",
                    fontsize=9, color=WHITE, fontweight="bold", zorder=5)

    def box(cx, cy, w, h, color, lines):
        r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor=WHITE, lw=2, zorder=3)
        ax.add_patch(r)
        for j, ln in enumerate(lines):
            off = 0.18*(len(lines)-1)/2 - j*0.18
            ax.text(cx, cy+off, ln, ha="center", va="center",
                    fontsize=9, color=WHITE, fontweight="bold", zorder=4)

    def arrow(x1, y1, x2, y2, lbl="", lbl_color=MGRAY):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=MGRAY, lw=2))
        if lbl:
            mx, my = (x1+x2)/2 + 0.12, (y1+y2)/2
            ax.text(mx, my, lbl, fontsize=9, fontweight="bold", color=lbl_color)

    ax.text(5, 6.65, "AI Model Selection Framework",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)

    # Root
    diamond(5, 5.7, 3.8, 0.95, NAVY2, ["Data must stay private?"])

    # LEFT: YES → private
    arrow(3.1, 5.7, 1.8, 5.7, "YES", RED)
    box(1.3, 5.7, 2.2, 0.7, NAVY2, ["Private / On-Prem", "Bedrock · Azure · Vertex"])

    # RIGHT: NO → capability check
    arrow(6.9, 5.7, 8.2, 5.7, "NO", GREEN)
    diamond(8.5, 5.7, 2.5, 0.88, NAVY2, ["Max capability", "needed?"])

    # YES → frontier
    arrow(8.5, 5.26, 8.5, 4.55, "YES", GREEN)
    box(8.5, 4.15, 2.6, 0.72, PURPLE, ["Frontier model", "Claude 3.7 · GPT-4.1"])

    # NO → cost check
    arrow(7.25, 5.7, 6.1, 4.9, "NO", MGRAY)
    diamond(5.5, 4.55, 2.8, 0.88, NAVY2, ["Cost sensitive?"])

    # YES (cost) → cheap
    arrow(5.5, 4.11, 5.5, 3.4, "YES", AMBER)
    box(5.5, 3.0, 2.8, 0.72, TEAL, ["Cost-optimized", "Haiku · GPT-4.1 mini"])

    # NO (cost) → volume check
    arrow(4.1, 4.55, 2.8, 3.9, "NO", MGRAY)
    diamond(2.3, 3.55, 2.6, 0.85, NAVY2, ["Volume > 10M", "calls/month?"])

    arrow(2.3, 3.12, 2.3, 2.45, "YES", AMBER)
    box(2.3, 2.05, 2.6, 0.72, AMBER, ["Fine-tune small model", "LoRA on Llama 4 8B"])

    arrow(3.6, 3.55, 5.0, 2.85, "NO", MGRAY)
    box(5.5, 2.65, 2.4, 0.65, GREEN, ["Open-source", "Llama 4 Scout — Free"])

    # Cost key
    for color, lbl in [(PURPLE, "Frontier"), (TEAL, "Cost-Opt."),
                       (AMBER, "Fine-tuned"), (GREEN, "Open-source"), (NAVY2, "Private")]:
        pass  # handled by colors

    save(fig, "slide_06_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 7: Chatbot vs Agent
# ─────────────────────────────────────────────────────────────────────────────
def slide_07():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    # Divider
    ax.axvline(5.5, color=LGRAY, lw=2, ymin=0.04, ymax=0.96)

    # ── LEFT: Chatbot ─────────────────────────────────────────────────────────
    ax.text(2.7, 6.1, "Chatbot  (2022–2024)",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)

    def sbox(cx, cy, w, h, fc, text, fs=10):
        r = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                            boxstyle="round,pad=0.1",
                            facecolor=fc, edgecolor=WHITE, lw=2, zorder=3)
        ax.add_patch(r)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, color=WHITE, fontweight="bold", zorder=4)

    sbox(2.7, 5.1, 2.2, 0.6, NAVY2, "User Input")
    ax.annotate("", xy=(2.7, 4.3), xytext=(2.7, 4.8),
                arrowprops=dict(arrowstyle="->", color=MGRAY, lw=2.2))
    sbox(2.7, 3.9, 2.2, 0.75, TEAL, "LLM")
    ax.annotate("", xy=(2.7, 3.1), xytext=(2.7, 3.52),
                arrowprops=dict(arrowstyle="->", color=MGRAY, lw=2.2))
    sbox(2.7, 2.7, 2.2, 0.6, NAVY2, "Text Output")

    ax.text(2.7, 1.9, "One round-trip.", ha="center", fontsize=10, color=DGRAY)
    ax.text(2.7, 1.5, "Human drives every step.", ha="center", fontsize=10, color=DGRAY)

    # ── RIGHT: Agent loop ─────────────────────────────────────────────────────
    ax.text(8.2, 6.1, "AI Agent  (2025–2026)",
            ha="center", fontsize=13, fontweight="bold", color=NAVY)

    cx, cy, r = 8.2, 3.5, 1.8
    n_nodes = 5
    angles = np.linspace(np.pi/2, np.pi/2 + 2*np.pi, n_nodes, endpoint=False)
    node_labels = ["Receive\nTask", "Plan\nSteps", "Call\nTools",
                   "Observe\nResult", "Reason &\nAdapt"]
    node_colors = [NAVY, TEAL, PURPLE, AMBER, TEAL]

    pts = [(cx + r*np.cos(a), cy + r*np.sin(a)) for a in angles]

    # Draw arcs between adjacent nodes
    for i in range(n_nodes):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % n_nodes]
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=MGRAY, lw=2,
                                    connectionstyle="arc3,rad=0.18"))

    # Draw nodes
    for i, (nx, ny) in enumerate(pts):
        c = plt.Circle((nx, ny), 0.52, color=node_colors[i],
                        zorder=5, ec=WHITE, lw=2)
        ax.add_patch(c)
        lines = node_labels[i].split("\n")
        for j, ln in enumerate(lines):
            off = 0.12*(len(lines)-1)/2 - j*0.12
            ax.text(nx, ny+off, ln, ha="center", va="center",
                    fontsize=8.5, color=WHITE, fontweight="bold", zorder=6)

    ax.text(cx, cy, "GOAL", ha="center", va="center",
            fontsize=11, color=NAVY, fontweight="bold")

    # Callout
    r2 = FancyBboxPatch((5.8, 0.4), 4.8, 0.75,
                          boxstyle="round,pad=0.1",
                          facecolor=AMBER, edgecolor="none", alpha=0.18)
    ax.add_patch(r2)
    ax.text(8.2, 0.77, "4-hour analyst task  →  15 minutes",
            ha="center", va="center", fontsize=11, fontweight="bold", color=AMBER)

    save(fig, "slide_07_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 14: Developer Productivity
# ─────────────────────────────────────────────────────────────────────────────
def slide_14():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    tasks = [
        ("Write boilerplate / scaffolding", 90, AMBER),
        ("Write new function",              7,  TEAL),
        ("Write unit tests",                7,  TEAL),
        ("Understand unfamiliar codebase",  5,  NAVY2),
        ("Fix a bug",                       4,  NAVY2),
    ]

    y = np.arange(len(tasks))
    bars = ax.barh(y, [t[1] for t in tasks],
                   color=[t[2] for t in tasks],
                   edgecolor=WHITE, linewidth=1.5, height=0.58, zorder=3)

    for i, (bar, (_, val, color)) in enumerate(zip(bars, tasks)):
        suffix = "×"
        # Value label inside or outside bar
        if val >= 20:
            ax.text(val - 2, bar.get_y() + bar.get_height()/2,
                    f"{val}{suffix}", va="center", ha="right",
                    fontsize=13, fontweight="bold", color=WHITE, zorder=4)
        else:
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val}{suffix}", va="center", ha="left",
                    fontsize=12, fontweight="bold", color=color, zorder=4)

    ax.annotate("Biggest win: boilerplate\neliminated entirely",
                xy=(90, 4), xytext=(60, 3.6),
                fontsize=9, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.5))

    ax.set_yticks(y)
    ax.set_yticklabels([t[0] for t in tasks], fontsize=10.5)
    ax.set_xlabel("Speedup Factor (×)", fontsize=11, color=DGRAY)
    ax.set_title("AI Coding Tools: Developer Productivity Gains (2025–2026)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlim(0, 104)
    ax.grid(axis="x", alpha=0.3, color=WHITE, zorder=0)
    ax.text(0.99, -0.11, "Source: GitHub & StackOverflow Surveys 2025–2026",
            transform=ax.transAxes, fontsize=7.5, color=MGRAY, ha="right")

    save(fig, "slide_14_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 20: AI Automation Time Saved
# ─────────────────────────────────────────────────────────────────────────────
def slide_20():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    processes = [
        ("Invoice processing",       97, "5 min → 10 sec",   NAVY),
        ("Contract risk review",     97, "2–3 hrs → 3 min",  NAVY),
        ("Competitive intel report", 94, "6 hrs → 20 min",   TEAL),
        ("Meeting → action items",   93, "30 min → 2 min",   TEAL),
        ("Support ticket triage",    91, "8 min → 45 sec",   PURPLE),
    ]

    y = np.arange(len(processes))
    bars = ax.barh(y, [p[1] for p in processes],
                   color=[p[3] for p in processes],
                   edgecolor=WHITE, linewidth=1.5, height=0.6, zorder=3)

    for bar, (_, val, detail, color) in zip(bars, processes):
        ax.text(val - 1.5, bar.get_y() + bar.get_height()/2,
                f"{val}%", va="center", ha="right",
                fontsize=12, fontweight="bold", color=WHITE, zorder=4)
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                detail, va="center", fontsize=9, color=DGRAY)

    ax.axvline(90, color=GREEN, lw=2.2, linestyle="--", alpha=0.85, zorder=2)
    ax.text(90.4, -0.62, "90% threshold", fontsize=8.5,
            color=GREEN, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([p[0] for p in processes], fontsize=11)
    ax.set_xlabel("% Time Saved", fontsize=11, color=DGRAY)
    ax.set_title("AI Automation: Time Saved by Business Process",
                 fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlim(0, 140)
    ax.set_xticks(range(0, 101, 20))
    ax.grid(axis="x", alpha=0.3, color=WHITE, zorder=0)
    ax.text(0.5, -0.12, "Avg. cost per automated task: <$0.10",
            transform=ax.transAxes, fontsize=10,
            color=AMBER, fontweight="bold", ha="center")

    save(fig, "slide_20_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 25: Industry Impact — horizontal grouped bars
# ─────────────────────────────────────────────────────────────────────────────
def slide_25():
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    # (industry, metric, pct_saved, color)
    rows = [
        ("Finance",     "Earnings call analysis",   93, NAVY),
        ("Finance",     "Regulatory gap analysis",  85, NAVY),
        ("Healthcare",  "Physician docs",            33, TEAL),
        ("Healthcare",  "Prior auth letters",        91, TEAL),
        ("Retail",      "Product descriptions",      99, PURPLE),
        ("Retail",      "Chat auto-resolution",      70, PURPLE),
        ("Consulting",  "RFP section drafting",      96, AMBER),
        ("Consulting",  "Contract risk flagging",    97, AMBER),
    ]

    y = np.arange(len(rows))
    colors = [r[3] for r in rows]
    values = [r[2] for r in rows]
    labels = [f"{r[0]}:  {r[1]}" for r in rows]

    bars = ax.barh(y, values, color=colors, edgecolor=WHITE,
                   linewidth=1.5, height=0.62, zorder=3)

    for bar, val, color in zip(bars, values, colors):
        if val >= 50:
            ax.text(val - 1.5, bar.get_y() + bar.get_height()/2,
                    f"{val}%", va="center", ha="right",
                    fontsize=10, fontweight="bold", color=WHITE, zorder=4)
        else:
            ax.text(val + 1, bar.get_y() + bar.get_height()/2,
                    f"{val}%", va="center", ha="left",
                    fontsize=10, fontweight="bold", color=color, zorder=4)

    # Industry separators
    ax.axhline(1.5, color=WHITE, lw=1.5, alpha=0.7)
    ax.axhline(3.5, color=WHITE, lw=1.5, alpha=0.7)
    ax.axhline(5.5, color=WHITE, lw=1.5, alpha=0.7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xlabel("% Time / Effort Saved", fontsize=11, color=DGRAY)
    ax.set_title("AI Impact Across Industries (2025–2026)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlim(0, 120)
    ax.grid(axis="x", alpha=0.3, color=WHITE, zorder=0)

    ax.legend(handles=[
        mpatches.Patch(facecolor=NAVY,   label="Finance"),
        mpatches.Patch(facecolor=TEAL,   label="Healthcare"),
        mpatches.Patch(facecolor=PURPLE, label="Retail"),
        mpatches.Patch(facecolor=AMBER,  label="Consulting"),
    ], loc="lower right", fontsize=9, framealpha=0.9)

    ax.text(0.5, -0.1, "AI absorbs 60–80% of information-gathering and drafting work",
            transform=ax.transAxes, fontsize=9, color=DGRAY,
            ha="center", style="italic")

    save(fig, "slide_25_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 28: Three-Tier Adoption Pyramid
# ─────────────────────────────────────────────────────────────────────────────
def slide_28():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    ax.text(4.5, 6.2, "AI Adoption Maturity Model",
            ha="center", fontsize=15, fontweight="bold", color=NAVY)

    # Tier bands (y_bottom, height, half_width, color, title, tools, cost, time)
    tiers = [
        (0.3, 1.6, 4.0, NAVY,   "Tier 1  —  Individual Productivity",
         "Claude / ChatGPT  ·  Cursor / Copilot  ·  Perplexity",
         "$0–$50/person", "Start now"),
        (2.1, 1.5, 2.8, NAVY2,  "Tier 2  —  Team Workflows",
         "Zapier / Make / n8n  ·  NotebookLM  ·  RAG chatbot",
         "$100–$1,000/mo", "Months 1–6"),
        (3.8, 1.5, 1.6, PURPLE, "Tier 3  —  Enterprise AI",
         "Fine-tuned models  ·  Multi-agent systems",
         "$10,000+/mo", "Months 6–18"),
    ]

    for (yb, ht, hw, color, title, tools, cost, timeline) in tiers:
        mid = 4.5
        xs = [mid - hw, mid + hw, mid + hw, mid - hw]
        ys = [yb, yb, yb + ht, yb + ht]
        ax.fill(xs, ys, color=color, alpha=0.92, zorder=3)
        ax.plot(xs + [xs[0]], ys + [ys[0]], color=WHITE, lw=2, zorder=4)

        ax.text(mid, yb + ht*0.67, title,
                ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=WHITE, zorder=5)
        ax.text(mid, yb + ht*0.32, tools,
                ha="center", va="center", fontsize=8.5,
                color=WHITE, alpha=0.9, zorder=5)

        # Cost badge (right)
        bx = mid + hw + 0.18
        by = yb + ht/2
        rb = FancyBboxPatch((bx, by - 0.22), 1.9, 0.44,
                             boxstyle="round,pad=0.06",
                             facecolor=AMBER, edgecolor="none", zorder=5)
        ax.add_patch(rb)
        ax.text(bx + 0.95, by, cost, ha="center", va="center",
                fontsize=8, fontweight="bold", color=NAVY, zorder=6)

        # Timeline badge (left)
        lx = mid - hw - 2.1
        lb = FancyBboxPatch((lx, by - 0.22), 1.88, 0.44,
                             boxstyle="round,pad=0.06",
                             facecolor=TEAL, edgecolor="none", zorder=5)
        ax.add_patch(lb)
        ax.text(lx + 0.94, by, timeline, ha="center", va="center",
                fontsize=8, fontweight="bold", color=WHITE, zorder=6)

    ax.annotate("", xy=(0.35, 5.3), xytext=(0.35, 0.5),
                arrowprops=dict(arrowstyle="->", color=MGRAY, lw=2))
    ax.text(0.22, 2.9, "Complexity\n& Cost", ha="center", va="center",
            fontsize=8.5, color=MGRAY, rotation=90)

    ax.text(4.5, 0.1, "Start at Tier 1. Prove value. Then invest upward.",
            ha="center", fontsize=10, color=AMBER, fontweight="bold")

    save(fig, "slide_28_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 29: Six Trends Radar
# ─────────────────────────────────────────────────────────────────────────────
def slide_29():
    cats = [
        "Intelligence\nCost Collapse",
        "Agentic AI\nProduction",
        "Domain\nSpecialization",
        "Multimodal\nBaseline",
        "AI Governance\n& Compliance",
        "Human Role\nShift",
    ]
    s24 = [5, 4, 4, 5, 3, 6]
    s26 = [9, 8, 7, 8, 6, 9]

    N = len(cats)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 6.5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_rlim(0, 10)
    ax.set_rticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2","4","6","8","10"], fontsize=8, color=MGRAY)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10, color=NAVY, fontweight="bold")
    ax.tick_params(pad=14)
    ax.grid(color=WHITE, lw=1.2, alpha=0.7)
    ax.spines["polar"].set_visible(False)

    data24 = s24 + s24[:1]
    data26 = s26 + s26[:1]

    ax.plot(angles, data24, color=NAVY, lw=2.5, linestyle="--", zorder=3)
    ax.fill(angles, data24, color=NAVY, alpha=0.12, zorder=2)
    ax.plot(angles, data26, color=TEAL, lw=2.5, zorder=4)
    ax.fill(angles, data26, color=TEAL, alpha=0.28, zorder=3)

    for a, v in zip(angles[:-1], s26):
        ax.plot(a, v, "o", color=TEAL, ms=7, zorder=5)

    ax.set_title("Six AI Mega-Trends: 2024 vs. 2026",
                 fontsize=13, fontweight="bold", color=NAVY, pad=22, y=1.1)
    ax.legend(handles=[
        Line2D([0],[0], color=NAVY, lw=2.5, linestyle="--", label="2024"),
        Line2D([0],[0], color=TEAL, lw=2.5, label="2026"),
    ], loc="lower right", bbox_to_anchor=(1.3, -0.05), fontsize=10)

    save(fig, "slide_29_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# Slide 30: Career Salary Ranges
# ─────────────────────────────────────────────────────────────────────────────
def slide_30():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(LGRAY)

    roles = [
        ("AI Product Manager",       140, 210, NAVY),
        ("ML Ops / AI Ops Engineer", 130, 185, NAVY2),
        ("AI Automation Engineer",   100, 165, TEAL),
        ("Prompt / LLM Engineer",     90, 155, TEAL),
        ("AI Governance Analyst",     90, 145, PURPLE),
        ("Business AI Analyst",       80, 135, PURPLE),
        ("AI Data Curator",           80, 135, AMBER),
    ]

    y = np.arange(len(roles))
    for i, (role, lo, hi, color) in enumerate(roles):
        ax.barh(i, hi - lo, left=lo, height=0.54,
                color=color, alpha=0.85,
                edgecolor=WHITE, linewidth=1.5, zorder=3)
        mid = (lo + hi) / 2
        ax.plot(mid, i, "o", color=WHITE, ms=8, zorder=5)
        ax.plot(mid, i, "o", color=color, ms=5, zorder=6)
        ax.text(hi + 2, i, f"${lo}K–${hi}K",
                va="center", fontsize=9.5, fontweight="bold", color=color)

    ax.axvline(100, color=GREEN, lw=2, linestyle="--", alpha=0.9, zorder=2)
    ax.text(100.5, -0.7, "$100K", fontsize=8.5, color=GREEN,
            fontweight="bold", va="bottom")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in roles], fontsize=10.5)
    ax.set_xlabel("Annual Salary (USD thousands)", fontsize=11, color=DGRAY)
    ax.set_title("Emerging AI Career Roles: Salary Ranges (US, 2026)",
                 fontsize=13, fontweight="bold", color=NAVY, pad=10)
    ax.set_xlim(60, 250)
    ax.set_xticks(range(60, 231, 30))
    ax.set_xticklabels([f"${v}K" for v in range(60, 231, 30)], fontsize=9)
    ax.grid(axis="x", alpha=0.3, color=WHITE, zorder=0)
    ax.text(0.99, -0.1, "Source: LinkedIn Salary Insights · Levels.fyi · 2026",
            transform=ax.transAxes, fontsize=7.5, color=MGRAY, ha="right")

    save(fig, "slide_30_chart.png")


# ── Run all ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\nGenerating 11 charts → {OUT_DIR}\n")
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
