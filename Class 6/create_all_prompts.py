"""
Create text files with prompts for all 27 slides
"""
import os
import re

def extract_slide_content(slide_number):
    """Extract slide content from markdown files"""
    
    # Determine which batch file to read and offset
    if slide_number <= 5:
        markdown_file = 'week06-slides-batch1.md'
        slide_offset = slide_number - 1
    elif slide_number <= 10:
        markdown_file = 'week06-slides-batch2.md'
        slide_offset = slide_number - 6
    elif slide_number <= 15:
        markdown_file = 'week06-slides-batch3.md'
        slide_offset = slide_number - 11
    elif slide_number <= 20:
        markdown_file = 'week06-slides-batch4.md'
        slide_offset = slide_number - 16
    elif slide_number <= 27:
        markdown_file = 'week06-slides-batch5.md'
        slide_offset = slide_number - 21
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
        slide_content = sections[0] if sections else ""
    
    # Clean content - remove emojis
    slide_content = re.sub(r'[^\x00-\x7F]+', '', slide_content)
    slide_content = slide_content.strip()
    
    return slide_content

# Create slide_prompts directory
os.makedirs('slide_prompts', exist_ok=True)

print("="*80)
print("📝 CREATING PROMPT TEXT FILES FOR ALL 27 SLIDES")
print("="*80)
print()

successful = 0
failed = 0

for slide_num in range(1, 28):
    try:
        # Extract content
        content = extract_slide_content(slide_num)
        
        # Create prompt
        prompt = f"""Slide {slide_num} - Playful Academic Cartoon Style

SLIDE CONTENT:
{content}

---

IMAGE GENERATION PROMPT:

Create a professional university presentation slide image with the following specifications:

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
        
        # Save to file
        filename = f'slide_prompts/slide_{slide_num:03d}_prompt.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        successful += 1
        print(f"✅ Slide {slide_num:2d}: Saved ({len(content):5d} chars)")
        
    except Exception as e:
        failed += 1
        print(f"❌ Slide {slide_num:2d}: Error - {str(e)[:80]}")

print()
print("="*80)
print(f"✅ Successfully created: {successful}/27 prompt files")
print(f"❌ Failed: {failed}/27")
print(f"📁 Files saved to: slide_prompts/")
print("="*80)
