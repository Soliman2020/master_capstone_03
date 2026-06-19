from markdown_pdf import MarkdownPdf, Section
from pathlib import Path

files = ["reports/Machine_Learning_Analysis_Report.md"]

for file in files:
    # Read markdown with UTF-8 encoding
    with open(file, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    # Get the directory where the markdown file is located
    markdown_dir = Path(file).parent
    # Create PDF with root set to the markdown file's directory
    pdf = MarkdownPdf(toc_level=2)
    pdf.meta["title"] = file.replace(".md", "")
    
    # Pass root=markdown_dir so images can be found
    pdf.add_section(Section(markdown_content, root=str(markdown_dir)))
    
    output_pdf = file.replace(".md", ".pdf")
    pdf.save(output_pdf)