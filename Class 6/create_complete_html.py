#!/usr/bin/env python3
"""
Create a complete self-contained HTML file from all markdown batches.
"""

import markdown
from pathlib import Path

def create_html_document(batch_contents):
    """Create complete HTML document."""
    
    # Build content sections
    content_sections = []
    for i, (title, content) in enumerate(batch_contents, 1):
        batch_html = f'''
        <div class="batch" id="batch{i}">
            <div class="batch-header">
                <h2>{title}</h2>
            </div>
            {content}
        </div>
        '''
        content_sections.append(batch_html)
    
    full_content = '\n'.join(content_sections)
    
    # Create HTML with proper CSS (single braces)
    html_doc = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Week 6: Coding with AI - Complete Course Materials</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{ font-size: 2.5em; margin-bottom: 10px; }}
        header p {{ font-size: 1.2em; opacity: 0.9; }}
        
        nav {{
            background: #2d3748;
            padding: 15px 40px;
            position: sticky;
            top: 0;
            z-index: 100;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        nav ul {{
            list-style: none;
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 20px;
        }}
        
        nav a {{
            color: white;
            text-decoration: none;
            padding: 8px 15px;
            border-radius: 5px;
            transition: background 0.3s;
        }}
        
        nav a:hover {{ background: rgba(255,255,255,0.1); }}
        
        .content {{ padding: 40px; }}
        
        .batch {{
            margin-bottom: 60px;
            padding: 30px;
            background: #f7fafc;
            border-radius: 8px;
        }}
        
        .batch-header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        
        .batch-header h2 {{ font-size: 2em; margin-bottom: 5px; }}
        
        .batch h1 {{ color: #667eea; font-size: 1.8em; margin: 25px 0 15px 0; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .batch h2 {{ color: #764ba2; font-size: 1.5em; margin: 20px 0 12px 0; }}
        .batch h3 {{ color: #667eea; font-size: 1.3em; margin: 18px 0 10px 0; }}
        .batch h4 {{ color: #764ba2; font-size: 1.15em; margin: 15px 0 8px 0; }}
        .batch h5 {{ color: #555; font-size: 1.05em; margin: 12px 0 6px 0; font-weight: 600; }}
        
        pre {{
            background: #2d3748;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 5px;
            overflow-x: auto;
            margin: 15px 0;
            border-left: 4px solid #667eea;
        }}
        
        code {{
            font-family: 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }}
        
        p code, li code {{
            background: #edf2f7;
            color: #e53e3e;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.9em;
        }}
        
        p {{ margin: 12px 0; line-height: 1.8; }}
        
        ul, ol {{ margin: 15px 0; padding-left: 30px; }}
        li {{ margin: 8px 0; }}
        
        ul ul, ol ul, ul ol, ol ol {{ margin: 8px 0; }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        tr:hover {{ background: #f7fafc; }}
        
        strong {{ color: #2d3748; font-weight: 600; }}
        em {{ font-style: italic; }}
        
        a {{ color: #667eea; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        
        hr {{
            border: none;
            border-top: 2px solid #e2e8f0;
            margin: 30px 0;
        }}
        
        blockquote {{
            border-left: 4px solid #667eea;
            padding-left: 20px;
            margin: 20px 0;
            font-style: italic;
            color: #555;
            background: #f7fafc;
            padding: 15px 20px;
            border-radius: 4px;
        }}
        
        footer {{
            background: #2d3748;
            color: white;
            text-align: center;
            padding: 30px;
        }}
        
        footer p {{
            color: white;
        }}
        
        footer strong {{
            color: white;
            font-weight: 600;
        }}
        
        @media print {{
            body {{ background: white; }}
            nav {{ display: none; }}
            .container {{ box-shadow: none; }}
        }}
        
        @media (max-width: 768px) {{
            .content {{ padding: 20px; }}
            header h1 {{ font-size: 1.8em; }}
            nav ul {{ flex-direction: column; gap: 10px; }}
            .batch {{ padding: 15px; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Week 6: Coding with AI</h1>
            <p>The Future of Software Development</p>
            <p><strong>BUAN 6v99.SW2 - Generative AI for Business</strong></p>
            <p>Spring 2026 | Complete Course Materials</p>
        </header>
        
        <nav>
            <ul>
                <li><a href="#batch1">Batch 1: Introduction</a></li>
                <li><a href="#batch2">Batch 2: Code Generation</a></li>
                <li><a href="#batch3">Batch 3: Debugging</a></li>
                <li><a href="#batch4">Batch 4: Testing & Review</a></li>
                <li><a href="#batch5">Batch 5: ROI & Best Practices</a></li>
            </ul>
        </nav>
        
        <div class="content">
            {full_content}
        </div>
        
        <footer>
            <p><strong>Week 6: Coding with AI - Complete Course Materials</strong></p>
            <p>BUAN 6v99.SW2 - Generative AI for Business | Spring 2026</p>
            <p style="margin-top: 20px;">
                <strong>Professor Antonio Paes</strong><br>
                University of Texas at Dallas
            </p>
        </footer>
    </div>
</body>
</html>'''
    
    return html_doc

def main():
    """Main function."""
    script_dir = Path(__file__).parent
    
    # Read all batch files
    batch_files = [
        ('week06-slides-batch1.md', '📚 Batch 1: Introduction & AI Coding Tools Landscape'),
        ('week06-slides-batch2.md', '⚡ Batch 2: Code Generation & GitHub Copilot Deep Dive'),
        ('week06-slides-batch3.md', '🐛 Batch 3: Conversational Coding & Debugging'),
        ('week06-slides-batch4.md', '✅ Batch 4: Code Review, Testing & Documentation'),
        ('week06-slides-batch5.md', '💼 Batch 5: Business Applications, ROI & Best Practices')
    ]
    
    batch_contents = []
    
    print("=" * 60)
    print("Creating Complete HTML Document")
    print("=" * 60)
    
    # Configure markdown with extensions
    md = markdown.Markdown(extensions=[
        'extra',        # Tables, fenced code blocks, etc.
        'codehilite',   # Syntax highlighting
        'nl2br',        # Newline to <br>
        'sane_lists'    # Better list handling
    ])
    
    for filename, title in batch_files:
        filepath = script_dir / filename
        if filepath.exists():
            print(f"Reading {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                md_content = f.read()
                html_content = md.convert(md_content)
                batch_contents.append((title, html_content))
                md.reset()  # Reset for next file
        else:
            print(f"⚠️  Warning: {filename} not found")
    
    # Create HTML document
    print("\nGenerating HTML...")
    html_doc = create_html_document(batch_contents)
    
    # Save to file
    output_file = script_dir / 'Week06_Coding_with_AI_Complete.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_doc)
    
    print(f"\n✅ Success!")
    print(f"   Created: {output_file}")
    print(f"   Batches included: {len(batch_contents)}")
    print(f"   File size: {output_file.stat().st_size / 1024:.1f} KB")
    print("\n" + "=" * 60)
    print("Open the file in your browser:")
    print(f"   open '{output_file}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
