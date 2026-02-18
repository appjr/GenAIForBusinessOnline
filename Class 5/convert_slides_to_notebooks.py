"""
Convert Week 5 Markdown Slides to Jupyter Notebooks
Each markdown file becomes an interactive notebook with executable code cells.
"""

import re
import nbformat as nbf
from pathlib import Path

def parse_markdown_to_notebook(md_content, title):
    """
    Convert markdown slide content to Jupyter notebook.
    
    Args:
        md_content: Markdown content
        title: Notebook title
    
    Returns:
        Jupyter notebook object
    """
    nb = nbf.v4.new_notebook()
    
    # Add title
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n**Interactive Jupyter Notebook Version**"))
    
    # Split content by slide markers (## Slide)
    slides = re.split(r'^## Slide \d+:', md_content, flags=re.MULTILINE)
    
    for slide_content in slides:
        if not slide_content.strip():
            continue
        
        # Split slide into sections
        parts = []
        current_text = []
        in_code_block = False
        code_block = []
        code_language = 'python'
        
        lines = slide_content.split('\n')
        
        for line in lines:
            # Check for code block markers
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block - save it
                    if code_block:
                        # Save any accumulated text first
                        if current_text:
                            parts.append(('markdown', '\n'.join(current_text)))
                            current_text = []
                        
                        # Save code block
                        code = '\n'.join(code_block)
                        if code.strip():
                            parts.append((code_language, code))
                        code_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    # Save accumulated text
                    if current_text:
                        parts.append(('markdown', '\n'.join(current_text)))
                        current_text = []
                    
                    # Determine language
                    lang_match = re.search(r'```(\w+)', line)
                    code_language = lang_match.group(1) if lang_match else 'python'
                    in_code_block = True
            elif in_code_block:
                code_block.append(line)
            else:
                current_text.append(line)
        
        # Add any remaining text
        if current_text:
            text = '\n'.join(current_text)
            if text.strip():
                parts.append(('markdown', text))
        
        # Convert parts to cells
        for part_type, content in parts:
            if part_type == 'markdown':
                # Clean up markdown
                content = content.strip()
                if content:
                    nb.cells.append(nbf.v4.new_markdown_cell(content))
            elif part_type == 'python':
                # Add as code cell
                content = content.strip()
                if content and not content.startswith('#'):
                    # Check if it's actual executable code
                    if any(keyword in content for keyword in ['import ', 'def ', 'class ', 'print(', '=']):
                        nb.cells.append(nbf.v4.new_code_cell(content))
                    else:
                        # Treat as markdown if not executable
                        nb.cells.append(nbf.v4.new_markdown_cell(f"```python\n{content}\n```"))
            elif part_type == 'bash':
                # Add bash as markdown code block
                nb.cells.append(nbf.v4.new_markdown_cell(f"```bash\n{content}\n```"))
    
    return nb


def convert_all_slides():
    """Convert all markdown slides to notebooks"""
    base_dir = Path(".")
    
    # Find all markdown slide files
    slide_files = sorted(base_dir.glob("week05-slides-batch*.md"))
    
    if not slide_files:
        print("No slide files found in current directory!")
        return
    
    # Create output directory
    output_dir = base_dir / "week05-slide-notebooks"
    output_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print("CONVERTING MARKDOWN SLIDES TO JUPYTER NOTEBOOKS")
    print("="*70)
    print(f"\nFound {len(slide_files)} slide files")
    print(f"Output directory: {output_dir}/\n")
    
    for slide_file in slide_files:
        print(f"Converting {slide_file.name}...")
        
        # Read markdown
        with open(slide_file, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to notebook
        title = slide_file.stem.replace('-', ' ').replace('_', ' ').title()
        nb = parse_markdown_to_notebook(md_content, title)
        
        # Save notebook
        nb_file = output_dir / f"{slide_file.stem}.ipynb"
        with open(nb_file, 'w', encoding='utf-8') as f:
            nbf.write(nb, f)
        
        print(f"  ✓ Created {nb_file.name} ({len(nb.cells)} cells)")
    
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nCreated {len(slide_files)} interactive notebooks")
    print(f"Location: {output_dir}/")
    print("\nStudents can now:")
    print("  - Read slides in notebook format")
    print("  - Run code examples interactively")
    print("  - Modify and experiment with code")
    print("  - Add their own notes")


if __name__ == "__main__":
    convert_all_slides()
