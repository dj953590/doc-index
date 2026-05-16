import argparse
import asyncio
import json
import os

from docindex.doc_index import md_to_tree
from docindex.utils import ConfigLoader, set_gemini_api_key, resolve_path


def load_options():
    """
    Merge CLI overrides with the package YAML config.


    Returns:
        A `SimpleNamespace` config object accepted by `md_to_tree`.
    """

    return ConfigLoader().load()


async def main():
    """
    Run the command-line Markdown indexing flow.

    The function validates the Markdown path, configures Vertex environment
    variables when supplied, builds the PageIndex JSON through `md_to_tree`, and
    writes the result to `./results/<markdown_stem>_structure.json`.
    """

    md_root = "examples/documents"
    md_path = resolve_path(f"{md_root}/2023-annual-report.md")

    opt = load_options()

    # Set Gemini API key if provided
    if opt.gemini_api_key:
        set_gemini_api_key(opt.gemini_api_key)


    print("Processing markdown file...")
    result = await md_to_tree(
        md_path=str(md_path),
        if_thinning=opt.if_thinning,
        min_token_threshold=opt.min_token_threshold,
        if_add_node_summary=opt.if_add_node_summary,
        summary_token_threshold=opt.summary_token_threshold,
        model=opt.model,
        if_add_doc_description=opt.if_add_doc_description,
        if_add_node_text=opt.if_add_node_text,
        if_add_node_id=opt.if_add_node_id,
    )

    print("Parsing done, saving to file...")
    md_name = os.path.splitext(os.path.basename(md_path))[0]
    output_dir = resolve_path(f"{md_root}/results")
    output_file = f"{str(output_dir)}/{md_name}_structure.json"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Tree structure saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
