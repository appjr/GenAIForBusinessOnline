"""
Convert Week 5 materials to HTML and Jupyter notebooks
"""

import os
import re
import nbformat as nbf
from pathlib import Path

def markdown_to_html(md_content, title):
    """Convert markdown to styled HTML"""
    # Simple markdown to HTML conversion
    html = md_content
    
    # Code blocks
    html = re.sub(r'```python\n(.*?)```', r'<pre><code class="language-python">\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```bash\n(.*?)```', r'<pre><code class="language-bash">\1</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'```\n(.*?)```', r'<pre><code>\1</code></pre>', html, flags=re.DOTALL)
    
    # Headers
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    
    # Bold and italic
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)
    
    # Lists
    html = re.sub(r'^\- (.*?)$', r'<li>\1</li>', html, flags=re.MULTILINE)
    
    # Links
    html = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2">\1</a>', html)
    
    # Wrap in HTML template
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
        li {{
            margin: 5px 0;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        {html}
    </div>
</body>
</html>"""
    
    return full_html


def python_to_notebook(py_content, title):
    """Convert Python script to Jupyter notebook"""
    nb = nbf.v4.new_notebook()
    
    # Add title cell
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}\n\n**From Week 5 Code Examples**"))
    
    # Split by docstring and functions
    lines = py_content.split('\n')
    
    current_block = []
    in_docstring = False
    
    for line in lines:
        if line.strip().startswith('"""') or line.strip().startswith("'''"):
            if in_docstring:
                # End of docstring - add as markdown
                docstring_text = '\n'.join(current_block)
                if docstring_text.strip():
                    nb.cells.append(nbf.v4.new_markdown_cell(docstring_text))
                current_block = []
                in_docstring = False
            else:
                # Start of docstring
                in_docstring = True
                current_block = []
        elif in_docstring:
            current_block.append(line.strip('"').strip("'"))
        else:
            current_block.append(line)
            
            # Add code cell when we hit a function or class definition end
            if line.strip() and not line.startswith(' ') and current_block:
                if line.startswith('if __name__'):
                    # Split main section
                    code = '\n'.join(current_block[:-1])
                    if code.strip():
                        nb.cells.append(nbf.v4.new_code_cell(code))
                    current_block = [line]
    
    # Add remaining code
    if current_block:
        code = '\n'.join(current_block)
        if code.strip():
            nb.cells.append(nbf.v4.new_code_cell(code))
    
    return nb


def convert_week05_materials():
    """Convert all Week 5 materials"""
    base_dir = Path(".")
    
    print("="*70)
    print("CONVERTING WEEK 5 MATERIALS")
    print("="*70)
    
    # Convert markdown slides to HTML
    slide_files = list(base_dir.glob("week05-slides-batch*.md"))
    html_dir = base_dir / "week05-html-slides"
    html_dir.mkdir(exist_ok=True)
    
    print(f"\n📄 Converting {len(slide_files)} slide files to HTML...")
    for md_file in sorted(slide_files):
        print(f"  Converting {md_file.name}...")
        with open(md_file, 'r') as f:
            md_content = f.read()
        
        html_content = markdown_to_html(md_content, md_file.stem)
        html_file = html_dir / f"{md_file.stem}.html"
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"    ✓ Saved to {html_file}")
    
    # Convert Python exercises to notebooks
    exercise_dir = base_dir / "week05-code-examples"
    notebook_dir = base_dir / "week05-notebooks"
    notebook_dir.mkdir(exist_ok=True)
    
    py_files = list(exercise_dir.rglob("*.py"))
    print(f"\n📓 Converting {len(py_files)} Python files to Jupyter notebooks...")
    
    for py_file in sorted(py_files):
        if py_file.name == 'convert_week05_materials.py':
            continue
            
        print(f"  Converting {py_file.name}...")
        with open(py_file, 'r') as f:
            py_content = f.read()
        
        nb = python_to_notebook(py_content, py_file.stem.replace('_', ' ').title())
        
        # Create subdirectory structure
        rel_path = py_file.relative_to(exercise_dir)
        nb_file = notebook_dir / rel_path.parent / f"{py_file.stem}.ipynb"
        nb_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(nb_file, 'w') as f:
            nbf.write(nb, f)
        
        print(f"    ✓ Saved to {nb_file}")
    
    print("\n" + "="*70)
    print("✅ CONVERSION COMPLETE!")
    print("="*70)
    print(f"\nHTML Slides: {html_dir}/")
    print(f"Notebooks: {notebook_dir}/")
    print(f"\nTotal files created:")
    print(f"  - {len(slide_files)} HTML slide files")
    print(f"  - {len(py_files)-1} Jupyter notebooks")


if __name__ == "__main__":
    convert_week05_materials()
