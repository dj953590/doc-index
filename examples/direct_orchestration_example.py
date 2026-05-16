"""
Example: Direct Orchestration - Custom Retrieval Workflow

This example demonstrates full programmatic control over the indexing and
retrieval process using MarkdownDocumentIndexer and DocumentRepository.

Use this approach when you need:
- Custom retrieval logic
- Deterministic behavior (no LLM decision-making)
- Performance optimization
- Integration into a service/API

This is the "Approach 2: Direct Orchestration (Programmatic)" from the README.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docindex.doc_index import MarkdownDocumentIndexer
from docindex.retrieve import (
    DocumentRepository,
    PageSelector,
    MarkdownContentRetriever,
)
from docindex.utils import (
    ConfigLoader,
    set_gemini_api_key,
    print_tree,
    structure_to_list,
    format_structure,
)


async def example_full_workflow():
    """
    Complete workflow: Index → Browse → Retrieve → Export
    """

    # ─────────────────────────────────────────────────────────────
    # Initialization
    # ─────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("DIRECT ORCHESTRATION: Full Workflow Example")
    print("="*70)

    config_loader = ConfigLoader()
    config = config_loader.load()

    if config.gemini_api_key:
        set_gemini_api_key(config.gemini_api_key)

    markdown_path = "examples/documents/four-lectures.pdf"

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Create the Indexer
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 1] Creating Indexer")
    print("─" * 70)

    indexer = MarkdownDocumentIndexer(model=config.model)
    print(f"✓ Indexer created with model: {config.model}")
    print(f"  • Thinning: {config.if_thinning}")
    print(f"  • Add summaries: {config.if_add_node_summary}")

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Index the Document
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 2] Indexing Document")
    print("─" * 70)
    print(f"File: {Path(markdown_path).name}")
    print("Processing...\n")

    try:
        result = await indexer.index(
            md_path=markdown_path,
            if_thinning=config.if_thinning,
            min_token_threshold=config.min_token_threshold,
            if_add_node_summary=config.if_add_node_summary,
            summary_token_threshold=config.summary_token_threshold,
            if_add_doc_description=config.if_add_doc_description,
            if_add_node_text=config.if_add_node_text,
            if_add_node_id=config.if_add_node_id,
        )
    except FileNotFoundError:
        print(f"❌ File not found: {markdown_path}")
        print("\nAvailable documents:")
        docs_dir = Path("examples/documents")
        for doc in list(docs_dir.glob("*"))[:5]:
            print(f"  • {doc.name}")
        return

    print(f"\n✓ Indexing Complete")
    print(f"  • Document: {result['doc_name']}")
    print(f"  • Lines: {result['line_count']}")
    print(f"  • Root sections: {len(result['structure'])}")

    if result.get('doc_description'):
        print(f"  • Description: {result['doc_description'][:60]}...")

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Inspect Tree Structure
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 3] Tree Structure")
    print("─" * 70)

    # Count total nodes
    all_nodes = structure_to_list(result['structure'])
    print(f"Total nodes: {len(all_nodes)}")

    # Print first few nodes as tree
    print("\nStructure (first 3 root sections):\n")
    print_tree(result['structure'][:3])

    if len(result['structure']) > 3:
        print(f"\n... and {len(result['structure']) - 3} more sections")

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Create Repository for Retrieval
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 4] Creating Repository")
    print("─" * 70)

    documents = {
        "doc_1": {
            "doc_name": result['doc_name'],
            "line_count": result['line_count'],
            "structure": result['structure'],
        }
    }

    repo = DocumentRepository(documents)
    print("✓ Repository created")
    print(f"  • Documents: 1")
    print(f"  • Document ID: doc_1")

    # ─────────────────────────────────────────────────────────────
    # STEP 5: Retrieve Content by Line Range
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 5] Retrieve by Line Range")
    print("─" * 70)

    try:
        # Get lines 1-50 (typically first few sections)
        print("Retrieving lines 1-50...\n")
        content_json = repo.get_page_content("doc_1", "1-50")
        content_list = json.loads(content_json)

        print(f"✓ Retrieved {len(content_list)} sections:")

        for i, item in enumerate(content_list[:4], 1):
            snippet = item['content'][:70].replace('\n', ' ')
            print(f"  {i}. Line {item['page']}: {snippet}...")

        if len(content_list) > 4:
            print(f"  ... and {len(content_list) - 4} more")

    except Exception as e:
        print(f"⚠️  Could not retrieve content: {e}")

    # ─────────────────────────────────────────────────────────────
    # STEP 6: Advanced Range Selection
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 6] Advanced Range Selection")
    print("─" * 70)

    selector = PageSelector()
    retriever = MarkdownContentRetriever()

    # Example: Get lines 10-15, 25, and 30-35
    range_str = "10-15,25,30-35"
    print(f"Parsing range: '{range_str}'")

    try:
        line_numbers = selector.parse(range_str)
        print(f"✓ Parsed to: {line_numbers}\n")

        custom_content = retriever.get_content(
            documents['doc_1'],
            line_numbers
        )

        print(f"Retrieved {len(custom_content)} nodes from custom range:")

        for item in custom_content:
            title = item.get('title', 'Untitled')
            page = item.get('page', '?')
            print(f"  • [{page:4}] {title}")

    except Exception as e:
        print(f"⚠️  Error: {e}")

    # ─────────────────────────────────────────────────────────────
    # STEP 7: Get Document Metadata
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 7] Document Metadata")
    print("─" * 70)

    try:
        doc_meta = repo.get_document("doc_1")
        print(f"Document: {doc_meta.get('doc_name')}")
        print(f"Lines: {doc_meta.get('line_count')}")
    except Exception as e:
        print(f"⚠️  Could not retrieve metadata: {e}")

    # ─────────────────────────────────────────────────────────────
    # STEP 8: Get Document Structure (without text)
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 8] Get Structure (without text)")
    print("─" * 70)

    try:
        structure_only = repo.get_document_structure("doc_1")

        # Count nodes
        nodes = structure_to_list(json.loads(structure_only))
        print(f"✓ Retrieved structure with {len(nodes)} nodes")
        print(f"  (Structure has no 'text' fields for efficiency)")

    except Exception as e:
        print(f"⚠️  Error: {e}")

    # ─────────────────────────────────────────────────────────────
    # STEP 9: Export Results
    # ─────────────────────────────────────────────────────────────

    print(f"\n[STEP 9] Export Results")
    print("─" * 70)

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Save full structure with summaries
    output_path = output_dir / f"{result['doc_name']}_structure_full.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"✓ Saved: {output_path}")

    # Save structure without text (smaller file)
    structure_only_result = {
        "doc_name": result['doc_name'],
        "line_count": result['line_count'],
        "structure": format_structure(
            result['structure'],
            order=["title", "node_id", "line_num", "summary", "prefix_summary", "nodes"]
        )
    }
    output_path_small = output_dir / f"{result['doc_name']}_structure_compact.json"
    with open(output_path_small, "w") as f:
        json.dump(structure_only_result, f, indent=2)
    print(f"✓ Saved: {output_path_small}")

    print(f"\n  Full size: ~{output_path.stat().st_size / 1024:.1f} KB")
    print(f"  Compact size: ~{output_path_small.stat().st_size / 1024:.1f} KB")

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────

    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    print(f"""
✓ Indexed document: {result['doc_name']}
✓ Created hierarchical tree with {len(all_nodes)} nodes
✓ Built repository for retrieval
✓ Demonstrated various retrieval patterns
✓ Exported results to {output_dir}

Next steps:
  1. Use repo.get_page_content() for semantic retrieval
  2. Implement custom traversal logic
  3. Integrate into your service/API
  4. Optimize based on your access patterns
""")


async def example_batch_indexing():
    """
    Example: Index multiple documents in parallel.
    """

    print("\n" + "="*70)
    print("EXAMPLE: Batch Indexing Multiple Documents")
    print("="*70)

    config_loader = ConfigLoader()
    config = config_loader.load()

    if config.gemini_api_key:
        set_gemini_api_key(config.gemini_api_key)

    docs_dir = Path("examples/documents")
    md_files = list(docs_dir.glob("*.pdf"))[:3]  # First 3 documents

    if not md_files:
        print("No documents found in examples/documents/")
        return

    print(f"\nFound {len(md_files)} documents to index:")
    for f in md_files:
        print(f"  • {f.name}")

    print("\nIndexing in parallel...\n")

    indexer = MarkdownDocumentIndexer(model=config.model)

    # Create tasks for all documents
    tasks = [
        indexer.index(
            str(f),
            if_thinning=config.if_thinning,
            if_add_node_summary="no",  # Skip for speed
            if_add_node_id="yes",
        )
        for f in md_files
    ]

    # Execute all in parallel
    results = await asyncio.gather(*tasks)

    print("\nResults:")
    for i, result in enumerate(results, 1):
        all_nodes = structure_to_list(result['structure'])
        print(f"  {i}. {result['doc_name']}: {len(all_nodes)} nodes, {result['line_count']} lines")

    # Create combined repository
    documents = {
        f"doc_{i}": {
            "doc_name": result['doc_name'],
            "structure": result['structure'],
        }
        for i, result in enumerate(results, 1)
    }

    repo = DocumentRepository(documents)
    print(f"\n✓ Created repository with {len(documents)} documents")


if __name__ == "__main__":
    # Run full workflow example
    asyncio.run(example_full_workflow())

    # Uncomment to run batch indexing example:
    # asyncio.run(example_batch_indexing())

