"""
Generic Slide Generator - Works for any slide number
Generates playful academic cartoon style slides using gpt-image-1
"""

import os
import sys
import re
import base64
import subprocess
from openai import OpenAI

def extract_slide_content(slide_number):
    """Extract slide content from markdown files"""
    
    # Determine which batch file to read and offset
    # Batch 1: Slides 1-5, Batch 2: Slides 6-10, Batch 3: Slides 11-15
    # Batch 4: Slides 16-20, Batch 5: Slides 21-27
    if slide_number <= 5:
        markdown_file = 'Class 6/week06-slides-batch1.md'
        slide_offset = slide_number - 1  # Slides 1-5 in Batch 1
    elif slide_number <= 10:
        markdown_file = 'Class 6/week06-slides-batch2.md'
        slide_offset = slide_number - 6  # Slides 6-10 in Batch 2
    elif slide_number <= 15:
        markdown_file = 'Class 6/week06-slides-batch3.md'
        slide_offset = slide_number - 11  # Slides 11-15 in Batch 3
    elif slide_number <= 20:
        markdown_file = 'Class 6/week06-slides-batch4.md'
        slide_offset = slide_number - 16  # Slides 16-20 in Batch 4
    elif slide_number <= 27:
        markdown_file = 'Class 6/week06-slides-batch5.md'
        slide_offset = slide_number - 21  # Slides 21-27 in Batch 5
    else:
        raise ValueError(f"Slide number {slide_number} is out of range (1-27)")
    
    # Read the file
    with open(markdown_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by ## headers to get sections
    sections = re.split(r'\n## ', content)
    sections = [s.strip() for s in sections if s.strip() and not s.startswith('#')]
    
    # Get the appropriate section
    if slide_offset < len(sections):
        slide_content = "## " + sections[slide_offset]
    else:
        print(f"⚠️  Slide {slide_number} index {slide_offset} out of range (found {len(sections)} sections)")
        # Try to get any content
        slide_content = sections[0] if sections else ""
    
    # Clean content - remove emojis
    slide_content = re.sub(r'[^\x00-\x7F]+', '', slide_content)
    
    # Keep structure but clean
    slide_content = slide_content.strip()
    
    return slide_content

def generate_slide_image(slide_number, slide_content):
    """Generate slide image using gpt-image-1"""
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Create comprehensive image generation prompt
    image_prompt = f"""Create a professional university presentation slide image with the following specifications:

SLIDE CONTENT:
{slide_content}

DESIGN THEME: Playful academic cartoon + clean university style

COLOR PALETTE:
- Soft blues (primary)
- Oranges (accents)
- Greens (secondary)
- White background with subtle texture

TYPOGRAPHY:
- Bold headers in rounded friendly font (like Montserrat or Poppins)
- Body text in clean sans-serif (like Open Sans)
- Title at top - large, bold, centered
- Section headers clearly differentiated

VISUAL STYLE:
- Illustrated cartoon-style icons (simple, friendly, hand-drawn feel)
- Simple diagrams where appropriate
- Light shadows for depth
- Subtle paper/canvas texture background
- Clean layout with good spacing
- Playful but professional
- Icons should be colorful and engaging

LAYOUT:
- 16:9 aspect ratio landscape (1536x1024)
- Title at top center
- Content well-organized with clear hierarchy
- Bullet points clean and readable
- Good white space
- Professional university quality

REQUIREMENTS:
- All text from content must be included and readable
- Professional enough for graduate business school
- Playful enough to be engaging and memorable
- Cartoon-style illustrations but sophisticated
- Clean, organized, easy to read at presentation size

Style reference: Coursera, Khan Academy, or Duolingo - educational, friendly, professional
"""
    
    print(f"🎨 Generating slide {slide_number} with playful academic cartoon style...")
    print("Using gpt-image-1 model")
    print("\nDesign specs:")
    print("  - Palette: Soft blues, oranges, greens")
    print("  - Style: Playful academic cartoon")
    print("  - Typography: Bold + rounded friendly")
    print("  - Visuals: Illustrated icons, clean diagrams")
    print()
    
    response = client.images.generate(
        model="gpt-image-1",
        prompt=image_prompt,
        size="1536x1024",  # 16:9 landscape format
        quality="high",
        n=1
    )
    
    print(f"✅ Image generated successfully!")
    
    # Save image
    output_file = f'Class 6/slide_images/slide_{slide_number:03d}_final.png'
    os.makedirs('Class 6/slide_images', exist_ok=True)
    
    print(f"\n📥 Saving image...")
    
    # Check if we have URL or b64_json
    if hasattr(response.data[0], 'url') and response.data[0].url:
        image_url = response.data[0].url
        print(f"🖼️  URL: {image_url}")
        subprocess.run(['curl', '-s', '-o', output_file, image_url], check=True)
    elif hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
        # Decode base64 image
        image_data = base64.b64decode(response.data[0].b64_json)
        with open(output_file, 'wb') as f:
            f.write(image_data)
        image_url = "Generated as base64 (no URL)"
    else:
        raise ValueError("No image data in response")
    
    print(f"✅ Saved to: {output_file}")
    
    # Save reference
    ref_file = f'Class 6/slide_{slide_number:03d}_image_url.txt'
    with open(ref_file, 'w') as f:
        f.write(f"Slide {slide_number} - Playful Academic Cartoon Style\n")
        f.write(f"Model: gpt-image-1\n")
        f.write(f"URL: {image_url}\n")
        f.write(f"\nPrompt used:\n{image_prompt}\n")
    
    return output_file

if __name__ == "__main__":
    print("="*80)
    print("🎓 GENERIC SLIDE GENERATOR - Playful Academic Cartoon Style")
    print("="*80)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set")
        exit(1)
    
    # Get slide number from command line or prompt
    if len(sys.argv) > 1:
        slide_number = int(sys.argv[1])
    else:
        slide_number = int(input("Enter slide number to generate: "))
    
    print(f"\n📝 Extracting content for slide {slide_number}...")
    slide_content = extract_slide_content(slide_number)
    
    if not slide_content or len(slide_content) < 50:
        print(f"❌ Could not extract content for slide {slide_number}")
        print("Please check the markdown files")
        exit(1)
    
    print(f"✅ Content extracted ({len(slide_content)} characters)")
    
    output_file = generate_slide_image(slide_number, slide_content)
    
    print("\n" + "="*80)
    print(f"✨ Slide {slide_number} image generated!")
    print(f"📄 File: {output_file}")
    print("\n🎨 Design features:")
    print("   - Playful academic cartoon style")
    print("   - Soft blues, oranges, greens palette")
    print("   - Bold headers + friendly rounded font")
    print("   - Illustrated icons and diagrams")
    print("   - Clean, professional university quality")
    print(f"\n🚀 To generate another slide, run:")
    print(f"   python Class6/generate_any_slide.py <slide_number>")
    print(f"\nExample: python Class6/generate_any_slide.py 5")
