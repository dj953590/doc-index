"""
Simple PDF to Markdown Converter for doc-index

Converts PDF documents to Markdown format using PyMuPDF's built-in markdown extraction.
The generated Markdown can then be indexed using the doc-index Markdown indexing pipeline.

Usage:
    python md_generator.py --pdf "path/to/document.pdf" --output "output.md"

Example:
    from md_generator import pdf_to_markdown
    md_text = pdf_to_markdown("document.pdf")
"""

import sys
import argparse
from pathlib import Path

import pymupdf4llm

try:
    import fitz  # PyMuPDF
except ImportError:
    print("ERROR: PyMuPDF not installed.")
    print("Install with: pip install pymupdf")
    sys.exit(1)


def pdf_to_markdown(pdf_path: str, verbose: bool = False) -> str:
    """
    Convert a PDF file to Markdown format.

    Args:
        pdf_path: Path to the PDF file.
        verbose: Print progress information.

    Returns:
        Markdown content as string.

    Example:
        md = pdf_to_markdown("document.pdf")
        print(md)
    """

    pdf_file = Path(pdf_path).expanduser()



    if verbose:
        print(f"Converting: {pdf_file.name}")


    # Extract markdown from each page
    md_text = pymupdf4llm.to_markdown(pdf_file)


    return md_text





def save_markdown(pdf_path: str, output_path: str, verbose: bool = False) -> str:
    """
    Convert PDF to Markdown and save to file.

    Args:
        pdf_path: Path to the PDF file.
        output_path: Path to save the Markdown file.
        verbose: Print progress information.

    Returns:
        Path to the output file.

    Example:
        output = save_markdown("document.pdf", "output.md", verbose=True)
        print(f"Saved to: {output}")
    """
    md_text = pdf_to_markdown(pdf_path, verbose=verbose)

    output_file = Path(output_path).expanduser()


    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_text)

    if verbose:
        print(f"✓ Saved to: {output_file}")

    return str(output_file)


