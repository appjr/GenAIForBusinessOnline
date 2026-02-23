#!/usr/bin/env python3
"""
Generate slide images for Week 6: Coding with AI
Uses OpenAI's DALL-E 3 to create playful academic cartoon-style images
"""

import os
import sys
import requests
from pathlib import Path

# Image generation configuration
STYLE = "playful academic cartoon style, vibrant colors, friendly and approachable, professional but fun, suitable for business education"
SIZE = "1024x1024"
QUALITY = "standard"
MODEL = "gpt-image-1"

# Slide prompts for Week 6: Coding with AI (27 slides)
SLIDE_PROMPTS = {
    # Batch 1: Introduction & AI Coding Landscape (Slides 1-5)
    1: f"Week 6 title illustration: A cheerful developer sitting at a modern desk with an AI assistant hologram appearing beside them, helping write code on a glowing screen. Binary code and neural network patterns float in the background. The scene shows collaboration between human and AI. {STYLE}",
    
    2: f"Agenda overview: A colorful roadmap or timeline showing different stages of learning, with icons representing coding tools, chat bubbles for AI assistants, checkmarks for testing, and a trophy for ROI. The path winds through playful cloud-like sections. {STYLE}",
    
    3: f"Learning objectives: A cheerful student character climbing a ladder toward floating knowledge bubbles, each bubble containing icons: a coding symbol, AI brain, GitHub logo, testing beaker, and dollar sign for ROI. Stars and achievement badges surround them. {STYLE}",
    
    4: f"Evolution timeline: A playful transformation showing four stages from left to right - an old typewriter (1990s), a simple computer with basic autocomplete (2000s), a laptop with smart suggestions (2010s), and a modern setup with a friendly AI robot assistant helping a developer (2020s). Progressive improvement visualization. {STYLE}",
    
    5: f"AI tools comparison: A friendly marketplace scene with different AI tool characters at colorful booths - GitHub Copilot as a helpful co-pilot with wings, ChatGPT as a wise advisor, Claude as a thoughtful scholar, and CodeWhisperer as an AWS cloud character. Developers happily comparing options. {STYLE}",
    
    # Batch 2: GitHub Copilot Deep Dive (Slides 6-10)
    6: f"GitHub Copilot setup: A cheerful developer installing a plugin, with the GitHub Copilot logo appearing as a friendly co-pilot character helping them. Installation steps float around them like achievement unlocks in a game. VS Code interface shown in a friendly, simplified way. {STYLE}",
    
    7: f"Effective prompting: A developer writing a comment, and magical code appearing from the comment like a genie from a lamp. The AI assistant shows the connection between clear instructions and quality code output. Light bulbs and sparkles indicate good ideas. {STYLE}",
    
    8: f"Real-world examples: Split scene showing different coding scenarios - one section with data analysis charts and graphs, another with API connections illustrated as friendly robots shaking hands, and a third showing automated reports flying out of a printer. All interconnected with flowing code. {STYLE}",
    
    9: f"Code refactoring: A before/after comparison showing messy, tangled code on the left transforming into clean, organized code on the right. An AI character acts as a friendly organizer, tidying up the code like Marie Kondo organizing a closet. Sparkles show the improvement. {STYLE}",
    
    10: f"GitHub Copilot Chat: A friendly chat interface with a developer asking questions and an AI assistant providing helpful answers. Speech bubbles contain code snippets, explanations, and friendly emojis. The interaction feels like a helpful pair programming session. {STYLE}",
    
    # Batch 3: Conversational AI for Coding (Slides 11-15)
    11: f"ChatGPT vs Claude: Two friendly AI characters side by side - ChatGPT as an energetic helper with the OpenAI logo, and Claude as a thoughtful scholar with books. They're both helping different developers, showing their unique strengths. No competition, just collaboration. {STYLE}",
    
    12: f"Complex problems visualization: A developer facing a large, intimidating puzzle or maze representing a complex coding problem. An AI assistant appears as a friendly guide with a glowing map, helping navigate through the complexity to find the solution. {STYLE}",
    
    13: f"Prompting strategies: A playful infographic showing the 4-part comment structure as building blocks - What, Input, Output, Special considerations - stacking together to form a strong prompt foundation. An AI character helps assemble them like LEGO blocks. {STYLE}",
    
    14: f"Data pipeline example: A cheerful factory-style illustration showing data flowing through different processing stages - incoming transactions as colorful packages on a conveyor belt, AI workers processing them, and clean organized outputs. Shows transformation from raw data to insights. {STYLE}",
    
    15: f"AI debugging: A detective-style scene with a developer and AI assistant examining code with magnifying glasses, finding bugs (shown as cute little bug characters). The AI highlights errors with spotlights, and provides fixes shown as bandages or repairs. Problem-solving collaboration. {STYLE}",
    
    # Batch 4: Code Review, Testing & Documentation (Slides 16-20)
    16: f"AI code review: A friendly AI reviewer character examining code on a screen with a clipboard, highlighting issues with gentle markers - security vulnerabilities shown as small warning signs, performance bottlenecks as speed bumps, and best practices shown as gold stars. {STYLE}",
    
    17: f"Security review example: A shield-wielding AI guardian protecting code from security threats. SQL injection attacks shown as blocked arrows, password vulnerabilities as locked treasure chests, and security fixes as protective barriers. Hero defending the codebase. {STYLE}",
    
    18: f"Test generation: An AI factory worker creating test cases like widgets on an assembly line. Each test is a colorful building block - happy path tests in green, edge cases in yellow, error cases in orange. 100% coverage shown as a complete puzzle. {STYLE}",
    
    19: f"Comprehensive testing: A pyramid of tests with AI helpers at each level - unit tests at the bottom as small building blocks, integration tests in the middle as connected pieces, and performance tests at the top as a stopwatch. All pieces fit together perfectly. {STYLE}",
    
    20: f"Documentation automation: An AI scribe character writing beautiful documentation in a magical book. Docstrings appear as organized scrolls, README files as welcoming signs, and API docs as helpful guidebooks. Everything well-organized and inviting. {STYLE}",
    
    # Batch 5: Business ROI & Implementation (Slides 21-27)
    21: f"Documentation examples: Side-by-side comparison of before (blank or messy notes) and after (beautifully formatted documentation with clear structure). An AI character transforms chaos into order with a magic wand. Professional yet friendly documentation. {STYLE}",
    
    22: f"Business impact metrics: A dashboard showing upward trending graphs, happy developers, faster delivery trucks, and quality badges. KPIs displayed as achievement medals - velocity increase, quality improvement, satisfaction scores. Success visualization. {STYLE}",
    
    23: f"ROI calculation: A friendly calculator or spreadsheet character showing impressive ROI numbers. Dollar signs and percentage symbols float around. Small investment on one side, massive value on the other, balanced on a scale that tips heavily toward value. {STYLE}",
    
    24: f"Case studies montage: Four different company scenarios - a startup team high-fiving, a mid-size company dashboard showing growth, an enterprise migration scene, and a financial services modernization. Success stories from different perspectives. {STYLE}",
    
    25: f"Implementation roadmap: A three-phase journey map showing Pilot (small team testing), Measurement (collecting metrics with charts), and Rollout (full team adoption). Each phase has cheerful milestone markers and celebration moments. Clear path to success. {STYLE}",
    
    26: f"Hands-on exercises: Developers actively coding with AI assistance, screens showing live code generation, pair programming sessions, and practical challenges. Energetic learning environment with people engaged and enjoying the process. {STYLE}",
    
    27: f"Key takeaways and next steps: A checklist or roadmap with completed items marked with checkmarks, future goals with sparkles, and a graduate character holding a diploma. Resources, community connections, and continued learning paths illustrated. Inspiring conclusion. {STYLE}",
}

def generate_slide_image(slide_number: int, output_dir: Path):
    """
    Generate a slide image using OpenAI's DALL-E 3
    
    Args:
        slide_number: The slide number to generate (1-27)
        output_dir: Directory to save the image
    """
    if slide_number not in SLIDE_PROMPTS:
        print(f"❌ Error: No prompt defined for slide {slide_number}")
        print(f"Valid slide numbers: 1-27")
        return False
    
    # Get API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Run: export OPENAI_API_KEY='your-key-here'")
        return False
    
    prompt = SLIDE_PROMPTS[slide_number]
    
    print(f"\n{'='*60}")
    print(f"Generating Slide {slide_number}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"Model: {MODEL}")
    print(f"Size: {SIZE}")
    print(f"Quality: {QUALITY}")
    print()
    
    # Call OpenAI API
    try:
        response = requests.post(
            'https://api.openai.com/v1/images/generations',
            headers={
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            },
            json={
                'model': MODEL,
                'prompt': prompt,
                'n': 1,
                'size': SIZE,
                'quality': QUALITY
            },
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        
        if 'data' not in result or len(result['data']) == 0:
            print("❌ Error: No image data in response")
            return False
        
        image_url = result['data'][0]['url']
        print(f"✅ Image generated successfully")
        print(f"URL: {image_url}")
        
        # Download image
        print(f"Downloading image...")
        img_response = requests.get(image_url, timeout=30)
        img_response.raise_for_status()
        
        # Save image
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"slide{slide_number:02d}.png"
        
        with open(image_path, 'wb') as f:
            f.write(img_response.content)
        
        print(f"✅ Image saved to: {image_path}")
        
        # Save URL for reference
        url_file = output_dir.parent / f"slide_{slide_number}_image_url.txt"
        with open(url_file, 'w') as f:
            f.write(f"Slide {slide_number}\n")
            f.write(f"URL: {image_url}\n")
            f.write(f"Prompt: {prompt}\n")
        
        print(f"✅ URL saved to: {url_file}")
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error calling OpenAI API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python generate_any_slide.py <slide_number>")
        print("Example: python generate_any_slide.py 1")
        print(f"Valid slide numbers: 1-27")
        sys.exit(1)
    
    try:
        slide_number = int(sys.argv[1])
    except ValueError:
        print(f"❌ Error: '{sys.argv[1]}' is not a valid number")
        sys.exit(1)
    
    if slide_number < 1 or slide_number > 27:
        print(f"❌ Error: Slide number must be between 1 and 27")
        print(f"Week 6 has 27 slides total across 5 batches")
        sys.exit(1)
    
    # Set up paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "slide_images"
    
    # Generate image
    success = generate_slide_image(slide_number, output_dir)
    
    if success:
        print(f"\n{'='*60}")
        print(f"✨ SUCCESS!")
        print(f"{'='*60}")
        print(f"Slide {slide_number} image generated and saved")
        print(f"Location: {output_dir / f'slide{slide_number:02d}.png'}")
        print()
        sys.exit(0)
    else:
        print(f"\n{'='*60}")
        print(f"❌ FAILED")
        print(f"{'='*60}")
        print(f"Could not generate image for slide {slide_number}")
        print()
        sys.exit(1)

if __name__ == "__main__":
    main()
