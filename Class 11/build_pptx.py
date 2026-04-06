#!/usr/bin/env python3
"""
Build Week11_AI_Trends_Slides.pptx directly from the week11-slides-batch*.md files.

Parses each markdown batch, extracts slides (## Slide N: Title), and renders
them as styled python-pptx slides using the course color scheme.

Handles:
  - Slide titles and subtitles (##, ###)
  - Bullet points (-, *, numbered lists) with sub-bullets
  - Tables  → native PPTX tables
  - Code blocks → styled text box with monospace font
  - Blockquotes → callout text box
  - Bold / italic inline formatting
  - Chart images → split layout (text left 58%, chart right 40%) when
    charts/slide_NN_chart.png exists for a given slide number

Usage:
    pip install python-pptx
    # Optional: generate charts first
    OPENAI_API_KEY=sk-... python3 generate_charts.py
    python3 build_pptx.py

Output:
    Week11_AI_Trends_Slides.pptx  (same directory)
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt, Emu

# ── Chart image lookup ─────────────────────────────────────────────────────────
# Slides that have a generated chart image in charts/slide_NN_chart.png
# The key is the 1-based content slide number (not PPTX slide index).
CHART_SLIDES = {3, 4, 5, 6, 7, 14, 20, 25, 28, 29, 30}

# ── Course colors ──────────────────────────────────────────────────────────────
NAVY        = RGBColor(0x16, 0x21, 0x3E)
NAVY_MID    = RGBColor(0x0F, 0x34, 0x60)
PURPLE      = RGBColor(0x53, 0x34, 0x83)
TEAL        = RGBColor(0x0E, 0xA5, 0xE9)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY  = RGBColor(0xF8, 0xF9, 0xFF)
DARK_TEXT   = RGBColor(0x1A, 0x1A, 0x2E)
MID_TEXT    = RGBColor(0x2D, 0x37, 0x48)
CODE_BG     = RGBColor(0x1A, 0x1A, 0x2E)
CODE_TEXT   = RGBColor(0xE2, 0xE8, 0xF0)
TEAL_ACCENT = RGBColor(0x0E, 0xA5, 0xE9)

# ── Slide geometry (16:9) ──────────────────────────────────────────────────────
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

TITLE_TOP    = Inches(0)
TITLE_H      = Inches(1.35)
CONTENT_TOP  = Inches(1.45)
CONTENT_H    = Inches(5.85)
MARGIN_L     = Inches(0.5)
MARGIN_R     = Inches(0.5)
CONTENT_W    = SLIDE_W - MARGIN_L - MARGIN_R


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class SlideContent:
    """Parsed representation of one slide."""
    title: str = ""
    subtitle: str = ""
    blocks: list = field(default_factory=list)   # list of Block objects


@dataclass
class TextBlock:
    kind: str          # "para", "bullet", "subbullet", "h3", "h4"
    text: str


@dataclass
class CodeBlock:
    kind: str = "code"
    language: str = ""
    lines: list = field(default_factory=list)


@dataclass
class QuoteBlock:
    kind: str = "quote"
    text: str = ""


@dataclass
class TableBlock:
    kind: str = "table"
    headers: list = field(default_factory=list)
    rows: list = field(default_factory=list)


# ── Markdown parser ────────────────────────────────────────────────────────────

def strip_inline(text: str) -> str:
    """Remove markdown bold/italic markers, leaving plain text."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*',     r'\1', text)
    text = re.sub(r'`(.+?)`',       r'\1', text)
    return text.strip()


def parse_batches(batch_files: list[Path]) -> list[SlideContent]:
    """
    Parse all batch markdown files and return a list of SlideContent objects.

    Each ## Slide N: Title line starts a new slide.
    Content between slide headings is parsed into typed blocks.
    """
    slides: list[SlideContent] = []
    current: Optional[SlideContent] = None

    in_code = False
    code_lang = ""
    code_lines: list[str] = []

    in_table = False
    table_headers: list[str] = []
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal in_table, table_headers, table_rows
        if in_table and current is not None:
            current.blocks.append(TableBlock(headers=table_headers, rows=table_rows))
        in_table = False
        table_headers = []
        table_rows = []

    def flush_code():
        nonlocal in_code, code_lang, code_lines
        if in_code and current is not None:
            current.blocks.append(CodeBlock(language=code_lang, lines=list(code_lines)))
        in_code = False
        code_lang = ""
        code_lines = []

    for path in batch_files:
        if not path.exists():
            print(f"  ✗ Missing: {path.name}")
            continue
        text = path.read_text(encoding="utf-8")

        for raw_line in text.splitlines():
            line = raw_line.rstrip()

            # ── Code fence toggle ──────────────────────────────────────────────
            if line.startswith("```"):
                if not in_code:
                    flush_table()
                    in_code = True
                    code_lang = line[3:].strip() or "text"
                    code_lines = []
                else:
                    flush_code()
                continue

            if in_code:
                code_lines.append(line)
                continue

            # ── Table rows ─────────────────────────────────────────────────────
            if "|" in line and line.strip().startswith("|"):
                # Skip separator rows like |---|---|
                if re.match(r"^[\|\s\-:]+$", line):
                    continue
                cells = [strip_inline(c) for c in line.strip().strip("|").split("|")]
                if not in_table:
                    flush_table()
                    in_table = True
                    table_headers = cells
                else:
                    table_rows.append(cells)
                continue
            else:
                if in_table:
                    flush_table()

            # ── Slide heading: ## Slide N: Title ──────────────────────────────
            if line.startswith("## Slide "):
                flush_code()
                flush_table()
                # Strip "Slide N: " prefix → keep just the descriptive title
                raw_title = line[3:].strip()
                m_title = re.match(r"Slide\s+\d+:\s+(.*)", raw_title)
                clean_title = m_title.group(1) if m_title else raw_title
                current = SlideContent(title=clean_title)
                slides.append(current)
                continue

            # Skip batch header lines (# Week 11 ...) and dividers
            if line.startswith("# ") or line.strip() == "---":
                continue

            # Nothing to add if no slide started yet
            if current is None:
                continue

            # ── Subtitle (### line immediately after slide heading) ────────────
            if line.startswith("### "):
                content = strip_inline(line[4:])
                if not current.subtitle:
                    current.subtitle = content
                else:
                    current.blocks.append(TextBlock(kind="h3", text=content))
                continue

            if line.startswith("#### "):
                current.blocks.append(TextBlock(kind="h4", text=strip_inline(line[5:])))
                continue

            # ── Blockquote ────────────────────────────────────────────────────
            if line.startswith("> "):
                current.blocks.append(QuoteBlock(text=strip_inline(line[2:])))
                continue

            # ── Bullets ───────────────────────────────────────────────────────
            m = re.match(r"^([ \t]*)[-\*] (.+)$", line)
            if m:
                indent = len(m.group(1).expandtabs(4))
                kind = "subbullet" if indent >= 2 else "bullet"
                current.blocks.append(TextBlock(kind=kind, text=strip_inline(m.group(2))))
                continue

            # ── Numbered list ─────────────────────────────────────────────────
            m2 = re.match(r"^\d+\. (.+)$", line)
            if m2:
                current.blocks.append(TextBlock(kind="bullet",
                                                 text=strip_inline(m2.group(1))))
                continue

            # ── Plain paragraph (non-empty) ───────────────────────────────────
            stripped = line.strip()
            if stripped:
                current.blocks.append(TextBlock(kind="para", text=strip_inline(stripped)))

    # Flush anything open at EOF
    flush_code()
    flush_table()

    return slides


# ── python-pptx helpers ────────────────────────────────────────────────────────

def solid_fill(shape, color: RGBColor):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color


def add_title_bar(slide, title_text: str, subtitle_text: str = ""):
    """Draw the navy title bar at the top of the slide."""
    # Background bar
    bar = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(0), Inches(0), SLIDE_W, TITLE_H
    )
    solid_fill(bar, NAVY)
    bar.line.fill.background()

    # Title text frame
    txBox = slide.shapes.add_textbox(
        Inches(0.45), Inches(0.08),
        SLIDE_W - Inches(0.9), Inches(0.75)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = title_text
    run.font.color.rgb = WHITE
    run.font.size = Pt(28)
    run.font.bold = True
    run.font.name = "Calibri"

    # Subtitle text (smaller, lighter)
    if subtitle_text:
        sub_box = slide.shapes.add_textbox(
            Inches(0.45), Inches(0.82),
            SLIDE_W - Inches(0.9), Inches(0.45)
        )
        stf = sub_box.text_frame
        sp = stf.paragraphs[0]
        srun = sp.add_run()
        srun.text = subtitle_text
        srun.font.color.rgb = TEAL
        srun.font.size = Pt(16)
        srun.font.name = "Calibri"

    # Thin teal accent line below title bar
    line = slide.shapes.add_shape(
        1, Inches(0), TITLE_H, SLIDE_W, Inches(0.04)
    )
    solid_fill(line, TEAL)
    line.line.fill.background()


def add_bg(slide):
    """White slide background."""
    bg = slide.shapes.add_shape(
        1, Inches(0), Inches(0), SLIDE_W, SLIDE_H
    )
    solid_fill(bg, WHITE)
    bg.line.fill.background()
    # Send to back
    slide.shapes._spTree.remove(bg._element)
    slide.shapes._spTree.insert(2, bg._element)


def set_para_font(para, size, bold=False, color=None, italic=False, name="Calibri"):
    for run in para.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        run.font.name = name
        if color:
            run.font.color.rgb = color


def render_table(slide, tbl: TableBlock,
                 left: Emu, top: Emu, width: Emu, height: Emu):
    """Render a TableBlock as a native PPTX table."""
    n_cols = max(len(tbl.headers), max((len(r) for r in tbl.rows), default=0))
    n_rows = 1 + len(tbl.rows)
    if n_cols == 0 or n_rows == 0:
        return

    # Clamp height
    row_h = min(Inches(0.38), height // n_rows)
    tbl_h = row_h * n_rows

    table = slide.shapes.add_table(n_rows, n_cols, left, top, width, tbl_h).table

    # Header row
    for ci, hdr in enumerate(tbl.headers[:n_cols]):
        cell = table.cell(0, ci)
        cell.text = hdr
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
        para = cell.text_frame.paragraphs[0]
        para.alignment = PP_ALIGN.LEFT
        for run in para.runs:
            run.font.color.rgb = WHITE
            run.font.size = Pt(11)
            run.font.bold = True
            run.font.name = "Calibri"

    # Data rows
    for ri, row in enumerate(tbl.rows, start=1):
        row_color = LIGHT_GRAY if ri % 2 == 0 else WHITE
        for ci in range(n_cols):
            cell = table.cell(ri, ci)
            cell.text = row[ci] if ci < len(row) else ""
            cell.fill.solid()
            cell.fill.fore_color.rgb = row_color
            para = cell.text_frame.paragraphs[0]
            for run in para.runs:
                run.font.size = Pt(10)
                run.font.name = "Calibri"
                run.font.color.rgb = DARK_TEXT


def render_code(slide, blk: CodeBlock,
                left: Emu, top: Emu, width: Emu):
    """Render a CodeBlock as a dark-background text box."""
    # Limit to first 18 lines so it fits
    lines = blk.lines[:18]
    n_lines = max(len(lines), 1)
    box_h = Inches(0.28) * n_lines + Inches(0.25)

    box = slide.shapes.add_shape(1, left, top, width, box_h)
    solid_fill(box, CODE_BG)
    box.line.fill.background()

    tb = slide.shapes.add_textbox(
        left + Inches(0.15), top + Inches(0.1),
        width - Inches(0.3), box_h - Inches(0.2)
    )
    tf = tb.text_frame
    tf.word_wrap = False

    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = ln
        run.font.color.rgb = CODE_TEXT
        run.font.size = Pt(9)
        run.font.name = "Courier New"

    return box_h


# ── Split-layout geometry ──────────────────────────────────────────────────────
# When a chart image is available, the slide is divided:
#   Left pane  (text/tables): 0.5" to 7.6"  → ~7.1" wide
#   Right pane (chart image): 7.7" to 12.8" → ~5.1" wide
TEXT_PANE_W  = Inches(7.1)
CHART_PANE_L = Inches(7.65)
CHART_PANE_W = Inches(5.15)
DIVIDER_X    = Inches(7.55)


def _chart_path(script_dir: Path, slide_num: int) -> Optional[Path]:
    """Return chart image path if it exists for this slide number."""
    p = script_dir / "charts" / f"slide_{slide_num:02d}_chart.png"
    return p if p.exists() else None


def render_slide_with_chart(prs: Presentation, sc: SlideContent,
                             slide_num: int, chart_img: Path):
    """
    Two-column slide layout:
      Left 58%: text blocks and tables
      Right 40%: AI-generated chart image
    """
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    add_bg(slide)
    add_title_bar(slide, sc.title, sc.subtitle)

    # ── Thin vertical divider ─────────────────────────────────────────────────
    div = slide.shapes.add_shape(
        1, DIVIDER_X, CONTENT_TOP, Inches(0.025), CONTENT_H
    )
    solid_fill(div, RGBColor(0xDD, 0xDD, 0xEE))
    div.line.fill.background()

    # ── Chart image (right pane) ──────────────────────────────────────────────
    chart_top  = CONTENT_TOP + Inches(0.1)
    chart_h    = CONTENT_H - Inches(0.2)
    slide.shapes.add_picture(
        str(chart_img),
        CHART_PANE_L, chart_top, CHART_PANE_W, chart_h
    )

    # ── Slide number badge ────────────────────────────────────────────────────
    num_box = slide.shapes.add_textbox(
        SLIDE_W - Inches(0.7), SLIDE_H - Inches(0.35),
        Inches(0.55), Inches(0.28)
    )
    np_ = num_box.text_frame.paragraphs[0]
    np_.alignment = PP_ALIGN.CENTER
    nr = np_.add_run()
    nr.text = str(slide_num)
    nr.font.size = Pt(10)
    nr.font.color.rgb = RGBColor(0xAA, 0xAA, 0xBB)
    nr.font.name = "Calibri"

    # ── Left pane: text blocks ────────────────────────────────────────────────
    left     = MARGIN_L
    top      = CONTENT_TOP
    avail_h  = CONTENT_H
    pane_w   = TEXT_PANE_W

    content_tf_box = None
    content_tf = None

    def ensure_text_box():
        nonlocal content_tf_box, content_tf, top
        if content_tf_box is None:
            content_tf_box = slide.shapes.add_textbox(
                left, top, pane_w, avail_h
            )
            content_tf = content_tf_box.text_frame
            content_tf.word_wrap = True

    def flush_text_box():
        nonlocal content_tf_box, content_tf, top
        if content_tf_box is not None:
            n_paras = len(content_tf.paragraphs)
            est_h = Inches(0.30) * n_paras
            top = top + max(est_h, Inches(0.3))
            content_tf_box = None
            content_tf = None

    def add_text_para(text, size, bold=False, color=None, indent=0, bullet_char=""):
        ensure_text_box()
        if len(content_tf.paragraphs) == 1 and not content_tf.paragraphs[0].runs:
            p = content_tf.paragraphs[0]
        else:
            p = content_tf.add_paragraph()
        p.level = indent
        p.space_before = Pt(2)
        p.space_after  = Pt(1)
        if bullet_char:
            rb = p.add_run()
            rb.text = bullet_char + " "
            rb.font.size = Pt(size)
            rb.font.color.rgb = TEAL if bullet_char in ("•", "›") else NAVY_MID
            rb.font.name = "Calibri"
        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = "Calibri"
        run.font.color.rgb = color or DARK_TEXT

    for blk in sc.blocks:
        if isinstance(blk, TableBlock):
            flush_text_box()
            if top + Inches(0.4) > SLIDE_H - Inches(0.3):
                continue
            avail = SLIDE_H - Inches(0.3) - top
            render_table(slide, blk, left, top, pane_w, avail)
            n_rows = 1 + len(blk.rows)
            top += Inches(0.38) * n_rows + Inches(0.12)
            continue

        if isinstance(blk, CodeBlock):
            flush_text_box()
            if top + Inches(0.5) > SLIDE_H - Inches(0.3):
                continue
            used = render_code(slide, blk, left, top, pane_w)
            top += used + Inches(0.12)
            continue

        if isinstance(blk, QuoteBlock):
            flush_text_box()
            if top + Inches(0.5) > SLIDE_H - Inches(0.3):
                continue
            bar = slide.shapes.add_shape(
                1, left, top, Inches(0.06), Inches(0.55)
            )
            solid_fill(bar, PURPLE)
            bar.line.fill.background()
            qtb = slide.shapes.add_textbox(
                left + Inches(0.18), top, pane_w - Inches(0.18), Inches(0.6)
            )
            qtf = qtb.text_frame
            qtf.word_wrap = True
            qp = qtf.paragraphs[0]
            qrun = qp.add_run()
            qrun.text = blk.text
            qrun.font.size = Pt(13)
            qrun.font.italic = True
            qrun.font.color.rgb = NAVY_MID
            qrun.font.name = "Calibri"
            top += Inches(0.65)
            continue

        if isinstance(blk, TextBlock):
            if blk.kind == "h3":
                flush_text_box()
                add_text_para(blk.text, 15, bold=True, color=PURPLE)
            elif blk.kind == "h4":
                add_text_para(blk.text, 12, bold=True, color=NAVY_MID)
            elif blk.kind == "bullet":
                add_text_para(blk.text, 12, bullet_char="•")
            elif blk.kind == "subbullet":
                add_text_para(blk.text, 10, bullet_char="›", indent=1, color=MID_TEXT)
            elif blk.kind == "para":
                if len(blk.text) < 80 and blk.text.endswith(":"):
                    add_text_para(blk.text, 12, bold=True, color=NAVY_MID)
                else:
                    add_text_para(blk.text, 11, color=MID_TEXT)

    flush_text_box()


def render_slide(prs: Presentation, sc: SlideContent, slide_num: int):
    """Create one PPTX slide from a SlideContent object."""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)

    add_bg(slide)
    add_title_bar(slide, sc.title, sc.subtitle)

    # ── Slide number badge (bottom-right) ─────────────────────────────────────
    num_box = slide.shapes.add_textbox(
        SLIDE_W - Inches(0.7), SLIDE_H - Inches(0.35),
        Inches(0.55), Inches(0.28)
    )
    np = num_box.text_frame.paragraphs[0]
    np.alignment = PP_ALIGN.CENTER
    nr = np.add_run()
    nr.text = str(slide_num)
    nr.font.size = Pt(10)
    nr.font.color.rgb = RGBColor(0xAA, 0xAA, 0xBB)
    nr.font.name = "Calibri"

    # ── Render content blocks ─────────────────────────────────────────────────
    # We manage a single content text box for text blocks,
    # and break out to separate shapes for tables and code.
    left = MARGIN_L
    top  = CONTENT_TOP
    avail_h = CONTENT_H

    # Separate blocks into segments: runs of text vs. table/code
    # We collect text runs and render them, then handle special blocks inline.
    content_tf_box = None
    content_tf = None

    def ensure_text_box():
        nonlocal content_tf_box, content_tf, top
        if content_tf_box is None:
            content_tf_box = slide.shapes.add_textbox(
                left, top, CONTENT_W, avail_h
            )
            content_tf = content_tf_box.text_frame
            content_tf.word_wrap = True

    def flush_text_box():
        nonlocal content_tf_box, content_tf, top
        if content_tf_box is not None:
            # Estimate rendered height — rough heuristic
            n_paras = len(content_tf.paragraphs)
            est_h = Inches(0.30) * n_paras
            top = top + max(est_h, Inches(0.3))
            content_tf_box = None
            content_tf = None

    def add_text_para(text: str, size: int, bold: bool = False,
                      color: RGBColor = None, indent: int = 0,
                      bullet_char: str = ""):
        ensure_text_box()
        if len(content_tf.paragraphs) == 1 and not content_tf.paragraphs[0].runs:
            p = content_tf.paragraphs[0]
        else:
            p = content_tf.add_paragraph()

        p.level = indent
        p.space_before = Pt(2)
        p.space_after  = Pt(1)

        if bullet_char:
            run_bullet = p.add_run()
            run_bullet.text = bullet_char + " "
            run_bullet.font.size = Pt(size)
            run_bullet.font.color.rgb = TEAL if bullet_char in ("•", "›") else NAVY_MID
            run_bullet.font.name = "Calibri"

        run = p.add_run()
        run.text = text
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.name = "Calibri"
        run.font.color.rgb = color or DARK_TEXT

    # ── Process blocks ─────────────────────────────────────────────────────────
    for blk in sc.blocks:

        if isinstance(blk, TableBlock):
            flush_text_box()
            if top + Inches(0.4) > SLIDE_H - Inches(0.3):
                continue  # no room
            avail = SLIDE_H - Inches(0.3) - top
            render_table(slide, blk, left, top, CONTENT_W, avail)
            n_rows = 1 + len(blk.rows)
            top += Inches(0.38) * n_rows + Inches(0.12)
            continue

        if isinstance(blk, CodeBlock):
            flush_text_box()
            if top + Inches(0.5) > SLIDE_H - Inches(0.3):
                continue
            used = render_code(slide, blk, left, top, CONTENT_W)
            top += used + Inches(0.12)
            continue

        if isinstance(blk, QuoteBlock):
            flush_text_box()
            if top + Inches(0.5) > SLIDE_H - Inches(0.3):
                continue
            # Purple left-border callout box
            bar = slide.shapes.add_shape(
                1, left, top, Inches(0.06), Inches(0.55)
            )
            solid_fill(bar, PURPLE)
            bar.line.fill.background()

            qtb = slide.shapes.add_textbox(
                left + Inches(0.18), top, CONTENT_W - Inches(0.18), Inches(0.6)
            )
            qtf = qtb.text_frame
            qtf.word_wrap = True
            qp = qtf.paragraphs[0]
            qrun = qp.add_run()
            qrun.text = blk.text
            qrun.font.size = Pt(13)
            qrun.font.italic = True
            qrun.font.color.rgb = NAVY_MID
            qrun.font.name = "Calibri"
            top += Inches(0.65)
            continue

        # ── Text blocks ────────────────────────────────────────────────────────
        if isinstance(blk, TextBlock):
            if blk.kind == "h3":
                flush_text_box()
                add_text_para(blk.text, 16, bold=True, color=PURPLE)
            elif blk.kind == "h4":
                add_text_para(blk.text, 13, bold=True, color=NAVY_MID)
            elif blk.kind == "bullet":
                add_text_para(blk.text, 13, bullet_char="•")
            elif blk.kind == "subbullet":
                add_text_para(blk.text, 11, bullet_char="›", indent=1,
                              color=MID_TEXT)
            elif blk.kind == "para":
                # Short paragraphs that look like section headers
                if len(blk.text) < 80 and blk.text.endswith(":"):
                    add_text_para(blk.text, 13, bold=True, color=NAVY_MID)
                else:
                    add_text_para(blk.text, 12, color=MID_TEXT)

    flush_text_box()


# ── Title slide ────────────────────────────────────────────────────────────────

def render_title_slide(prs: Presentation):
    """Create the decorative title/cover slide."""
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    # Full-bleed gradient background (simulated with layered rectangles)
    bg1 = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    solid_fill(bg1, NAVY)
    bg1.line.fill.background()

    accent = slide.shapes.add_shape(1, Inches(0), Inches(4.5), SLIDE_W, Inches(3.0))
    solid_fill(accent, NAVY_MID)
    accent.line.fill.background()

    teal_bar = slide.shapes.add_shape(1, Inches(0), Inches(3.8), SLIDE_W, Inches(0.08))
    solid_fill(teal_bar, TEAL)
    teal_bar.line.fill.background()

    # Main title
    ttb = slide.shapes.add_textbox(Inches(0.8), Inches(1.3), Inches(11.8), Inches(1.6))
    ttf = ttb.text_frame
    tp = ttf.paragraphs[0]
    tr = tp.add_run()
    tr.text = "Week 11: New AI Tools and Trends"
    tr.font.size = Pt(42)
    tr.font.bold = True
    tr.font.color.rgb = WHITE
    tr.font.name = "Calibri"

    # Subtitle
    stb = slide.shapes.add_textbox(Inches(0.8), Inches(2.95), Inches(11.8), Inches(0.8))
    stf = stb.text_frame
    sp = stf.paragraphs[0]
    sr = sp.add_run()
    sr.text = "The Full 2026 AI Landscape — Models · Agents · Coding Tools · Automation · Strategy"
    sr.font.size = Pt(19)
    sr.font.color.rgb = TEAL
    sr.font.name = "Calibri"

    # Meta info
    mtb = slide.shapes.add_textbox(Inches(0.8), Inches(5.0), Inches(11.8), Inches(1.8))
    mtf = mtb.text_frame
    for line, size, bold, color in [
        ("BUAN 6v99 — Generative AI for Business", 16, False, WHITE),
        ("University of Texas at Dallas  |  Spring 2026", 14, False,
         RGBColor(0xAA, 0xBB, 0xDD)),
        ("April 8, 2026  |  Professor Antonio de Pádua Paes Jr.", 14, False,
         RGBColor(0xAA, 0xBB, 0xDD)),
    ]:
        if mtf.paragraphs[0].runs:
            mp = mtf.add_paragraph()
        else:
            mp = mtf.paragraphs[0]
        mr = mp.add_run()
        mr.text = line
        mr.font.size = Pt(size)
        mr.font.bold = bold
        mr.font.color.rgb = color
        mr.font.name = "Calibri"
        mp.space_before = Pt(4)


# ── Section divider slide ──────────────────────────────────────────────────────

SECTION_TITLES = {
    7:  ("Section 2", "AI Agent Frameworks & MCP"),
    14: ("Section 3", "AI Coding Tools"),
    20: ("Section 4", "Business Automation & Multimodal AI"),
    26: ("Section 5", "Key Trends & Strategic Outlook"),
}


def render_section_divider(prs: Presentation, section_num: str, section_title: str):
    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    bg = slide.shapes.add_shape(1, Inches(0), Inches(0), SLIDE_W, SLIDE_H)
    solid_fill(bg, NAVY_MID)
    bg.line.fill.background()

    bar = slide.shapes.add_shape(1, Inches(0), Inches(3.2), SLIDE_W, Inches(0.08))
    solid_fill(bar, TEAL)
    bar.line.fill.background()

    nb = slide.shapes.add_textbox(Inches(1.0), Inches(2.1), Inches(11.0), Inches(0.7))
    np = nb.text_frame.paragraphs[0]
    nr = np.add_run()
    nr.text = section_num
    nr.font.size = Pt(22)
    nr.font.color.rgb = TEAL
    nr.font.name = "Calibri"
    nr.font.bold = True

    tb = slide.shapes.add_textbox(Inches(1.0), Inches(2.85), Inches(11.0), Inches(1.2))
    tp = tb.text_frame.paragraphs[0]
    trun = tp.add_run()
    trun.text = section_title
    trun.font.size = Pt(36)
    trun.font.bold = True
    trun.font.color.rgb = WHITE
    trun.font.name = "Calibri"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    script_dir = Path(__file__).parent

    batch_files = [
        script_dir / f"week11-slides-batch{i}.md"
        for i in range(1, 6)
    ]

    print("Parsing markdown batches...")
    slides = parse_batches(batch_files)
    print(f"  Found {len(slides)} slides")

    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    # Cover slide
    render_title_slide(prs)
    print("  ✓ Cover slide")

    # Content slides
    charts_used = 0
    for i, sc in enumerate(slides, start=1):
        # Insert section divider before certain slide numbers
        if i in SECTION_TITLES:
            sec_num, sec_title = SECTION_TITLES[i]
            render_section_divider(prs, sec_num, sec_title)
            print(f"  ✓ Section divider: {sec_title}")

        # Use split layout if chart image exists for this slide
        chart_img = _chart_path(script_dir, i)
        if chart_img is not None:
            render_slide_with_chart(prs, sc, i, chart_img)
            print(f"  ✓ Slide {i} [+chart]: {sc.title[:55]}")
            charts_used += 1
        else:
            render_slide(prs, sc, i)
            print(f"  ✓ Slide {i}: {sc.title[:60]}")

    output = script_dir / "Week11_AI_Trends_Slides.pptx"
    prs.save(str(output))

    size_mb = output.stat().st_size / (1024 * 1024)
    total_slides = len(prs.slides)
    print(f"\n✅  Saved: {output.name}")
    print(f"   Total slides (incl. cover + dividers): {total_slides}")
    print(f"   Slides with embedded charts: {charts_used}")
    print(f"   File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
