#!/usr/bin/env python3
"""
Generate slide images for Week 7 using OpenAI gpt-image-1.
Adapted from course_agent.py — reads actual slide content to build
content-accurate prompts, exactly matching what each slide teaches.

Usage:
    OPENAI_API_KEY=sk-proj-... python3 generate_slide_images.py

Images saved to:  Class 7/images/slide_NNN_final.png
Cost:             28 images × ~$0.06 = ~$1.68  (gpt-image-1 high quality)
Resumes from where it left off if interrupted.
"""

import base64
import json
import os
import re
import time
from pathlib import Path

import openai

# ── Config ────────────────────────────────────────────────────────────────────
API_KEY   = os.environ.get("OPENAI_API_KEY", "")
SCRIPT_DIR = Path(__file__).parent
OUT_DIR    = SCRIPT_DIR / "images"
PROGRESS_FILE = OUT_DIR / ".progress.json"

IMG_MODEL = "gpt-image-1"
IMG_SIZE  = "1536x1024"
IMG_QUAL  = "high"

CLASS_TITLE = "Week 7: Text Generation & Chatbots"

IMG_STYLE = (
    "Style: modern professional university course slide. Clean white background "
    "with very subtle light gray (#f8f9fa) sections. Bold sans-serif typography "
    "(Inter or Montserrat). Accent colors: deep navy #1a2744 for headers, vibrant "
    "teal #0ea5e9 for highlights and data elements, warm amber #f59e0b for callouts "
    "and key numbers. Data visualization elements rendered with sharp edges, clear "
    "labels, and strong hierarchy. McKinsey/HBR deck quality — minimal, elegant, "
    "high-information-density. No clipart. Flat design with subtle drop shadows. "
    "Clean grid layout. Professional infographic. UTDallas graduate course quality."
)

BATCH_FILES = [
    SCRIPT_DIR / "week07-slides-batch1.md",
    SCRIPT_DIR / "week07-slides-batch2.md",
    SCRIPT_DIR / "week07-slides-batch3.md",
    SCRIPT_DIR / "week07-slides-batch4.md",
    SCRIPT_DIR / "week07-slides-batch5.md",
]


# ── Slide extraction ──────────────────────────────────────────────────────────

def extract_all_slides() -> list[dict]:
    """Read all batch files and return a flat list of slide dicts."""
    slides = []
    for batch_file in BATCH_FILES:
        if not batch_file.exists():
            print(f"  ⚠  Missing: {batch_file.name}")
            continue
        text = batch_file.read_text(encoding="utf-8", errors="ignore")
        # Split on ## Slide headings
        sections = re.split(r"\n(?=## Slide)", text)
        for section in sections:
            section = section.strip()
            if not section.startswith("## Slide"):
                continue
            # Extract title from first line
            first_line = section.splitlines()[0]
            title = first_line[3:].strip()  # strip "## "
            # Extract clean text excerpt (strip markdown, code blocks, tables)
            body = section[len(first_line):]
            body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
            body = re.sub(r"[#*`|>\-]", "", body)
            body = re.sub(r"\n+", " ", body)
            excerpt = " ".join(body.split())[:350]
            slides.append({
                "seq": len(slides) + 1,
                "title": title,
                "excerpt": excerpt,
                "filename": f"slide_{len(slides)+1:03d}_final.png",
            })
    return slides


# ── Progress helpers ──────────────────────────────────────────────────────────

def load_progress() -> set:
    if PROGRESS_FILE.exists():
        data = json.loads(PROGRESS_FILE.read_text())
        return set(data.get("done", []))
    return set()


def save_progress(done: set):
    PROGRESS_FILE.write_text(json.dumps({"done": sorted(done)}, indent=2))


# ── Image generation ──────────────────────────────────────────────────────────

def generate_images():
    client = openai.OpenAI(api_key=API_KEY)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    slides   = extract_all_slides()
    done     = load_progress()
    total    = len(slides)
    generated = 0
    skipped   = 0
    errors    = 0

    print(f"\n  {total} slides found across {len(BATCH_FILES)} batch files.\n")

    for slide in slides:
        seq      = slide["seq"]
        title    = slide["title"]
        excerpt  = slide["excerpt"]
        filename = slide["filename"]
        out_path = OUT_DIR / filename

        if out_path.exists() and seq in done:
            print(f"  ⏭  [{seq:02d}/{total}] {filename} — skipping")
            skipped += 1
            continue

        prompt = (
            f'Slide {seq} of {total} for "{CLASS_TITLE}": "{title}". '
            f'Key content: {excerpt} '
            f'{IMG_STYLE} '
            f'Show slide number {seq} as a small badge in the bottom-right corner.'
        )

        print(f"  🎨 [{seq:02d}/{total}] {title[:65]}...", flush=True)
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

            done.add(seq)
            save_progress(done)
            size_kb = out_path.stat().st_size // 1024
            print(f"     ✓ saved ({size_kb} KB)")
            generated += 1

        except Exception as exc:
            print(f"     ✗ ERROR: {exc}")
            errors += 1

        if seq < total:
            time.sleep(3)   # brief pause between requests

    print(f"\n{'─'*60}")
    print(f"  Done — generated: {generated}, skipped: {skipped}, errors: {errors}")
    print(f"  Images saved to: {OUT_DIR}")


if __name__ == "__main__":
    if not API_KEY:
        print("ERROR: Set OPENAI_API_KEY environment variable first.")
        raise SystemExit(1)
    generate_images()
