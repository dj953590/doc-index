"""
Simple Example: PDF to Markdown Conversion

This example demonstrates how to convert PDF documents to Markdown format
using PyMuPDF's built-in markdown extraction.

The generated Markdown can then be indexed using the doc-index pipeline.
"""

import asyncio
from pathlib import Path

from docindex.utils import resolve_path
# Import the PDF to Markdown converter
from md_generator import pdf_to_markdown, save_markdown

# Import the doc-index pipeline
from docindex.doc_index import MarkdownDocumentIndexer



async def example_pdf_to_markdown_to_index():
    """
    Complete workflow: PDF → Markdown → Index → Retrieve
    """

    print("\n" + "="*70)
    print("PDF to Markdown to Index Workflow")
    print("="*70)

    # Step 1: Convert PDF to Markdown

    pdf_path = resolve_path("examples/documents/q1-fy25-earnings.pdf")
    markdown_output = resolve_path("examples/documents/q1-fy25-earnings.md")

    print(f"\n[STEP 1] Converting PDF to Markdown")
    print(f"  Input:  {pdf_path}")

    try:
        output_file = save_markdown(
            pdf_path=str(pdf_path),
            output_path=str(markdown_output),
            verbose=True
        )
        print(f"✓ Conversion complete")
    except FileNotFoundError as e:
        print(f"⚠️  PDF not found: {pdf_path}")
        raise e


     # Step 2: Index the Markdown
    print(f"[STEP 2] Indexing the Markdown")

    indexer = MarkdownDocumentIndexer(model="gemini-2.5-flash")

    result = await indexer.index(
        md_path=str(markdown_output),
        if_thinning=False,
        if_add_node_summary="no",
        if_add_doc_description="no",
        if_add_node_text="yes",
        if_add_node_id="yes",
    )

    print(f"✓ Indexed successfully")
    print(f"  Document: {result['doc_name']}")
    print(f"  Lines: {result['line_count']}")
    print(f"  Sections: {len(result['structure'])}")

    # Step 3: Display the structure
    print(f"\n[STEP 3] Document Structure")

    def print_tree(nodes, indent=0, max_depth=2):
        if indent >= max_depth:
            return
        for node in nodes:
            title = node['title'][:50]  # Truncate long titles
            print("  " * indent + f"• {title} (Line {node['line_num']})")
            if node.get('nodes'):
                print_tree(node['nodes'], indent + 1, max_depth)

    print_tree(result['structure'])

    return result


def example_direct_conversion():
    """
    Simple direct conversion without indexing.
    """

    print("\n" + "="*70)
    print("Simple Direct Conversion")
    print("="*70)

    pdf_path = "examples/documents/2023-annual-report.pdf"

    print(f"\nConverting: {pdf_path}")

    try:
        md_text = pdf_to_markdown(pdf_path, verbose=True)
        print(f"\n✓ Generated {len(md_text)} characters of Markdown")
        print(f"\nFirst 500 characters:\n{'-'*60}")
        print(md_text[:500])
        print(f"{'-'*60}\n")
    except FileNotFoundError:
        print(f"PDF not found: {pdf_path}")
        print("To use this example, place a PDF in examples/documents/")




async def main():
    """Run examples."""

    print("\n" + "="*70)
    print("PDF to Markdown Example")
    print("="*70)

    # Run direct conversion example
    #example_direct_conversion()

    # Run full workflow example
    try:
        await example_pdf_to_markdown_to_index()
    except Exception as e:
        print(f"\n⚠️  Workflow example encountered an issue: {e}")
        print("   This is expected if indexing model is not configured.")


if __name__ == "__main__":
    asyncio.run(main())



