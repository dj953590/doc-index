import argparse
import asyncio
import json
import os

from docindex.doc_index import md_to_tree
from docindex.utils import ConfigLoader, set_gemini_api_key


def parse_args():
    """
    Define and parse CLI arguments for Markdown PageIndex generation.

    Returns:
        Parsed `argparse.Namespace` with Markdown path, Gemini/Vertex options,
        and output-shaping switches used by `main`.
    """
    parser = argparse.ArgumentParser(description="Process a Markdown document and generate a PageIndex structure")
    parser.add_argument("--md_path", type=str, required=True, help="Path to the Markdown file")
    return parser.parse_args()


def load_options(args):
    """
    Merge CLI overrides with the package YAML config.

    Args:
        args: Parsed CLI namespace from `parse_args`.

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
    args = parse_args()
    if not args.md_path.lower().endswith((".md", ".markdown")):
        raise ValueError("Markdown file must have .md or .markdown extension")
    if not os.path.isfile(args.md_path):
        raise ValueError(f"Markdown file not found: {args.md_path}")

    opt = load_options(args)

    # Set Gemini API key if provided
    if opt.gemini_api_key:
        set_gemini_api_key(opt.gemini_api_key)

    if opt.google_cloud_project:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = opt.google_cloud_project
        if not opt.google_cloud_location:
            opt.google_cloud_location = "us-central1"
    if opt.google_cloud_location:
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_LOCATION"] = opt.google_cloud_location
    print("Processing markdown file...")
    result = await md_to_tree(
        md_path=args.md_path,
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
    md_name = os.path.splitext(os.path.basename(args.md_path))[0]
    output_dir = "./results"
    output_file = f"{output_dir}/{md_name}_structure.json"
    os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Tree structure saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
