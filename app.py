import argparse
import asyncio
import json
import os

from docindex.doc_index import md_to_tree
from docindex.utils import ConfigLoader


def parse_args():
    """
    Define and parse CLI arguments for Markdown PageIndex generation.

    Returns:
        Parsed `argparse.Namespace` with Markdown path, Gemini/Vertex options,
        and output-shaping switches used by `main`.
    """
    parser = argparse.ArgumentParser(description="Process a Markdown document and generate a PageIndex structure")
    parser.add_argument("--md_path", type=str, required=True, help="Path to the Markdown file")
    parser.add_argument("--model", type=str, default=None, help="Model to use (overrides config.yaml)")
    parser.add_argument("--google-cloud-project", type=str, default=None, help="Vertex AI Google Cloud project")
    parser.add_argument("--google-cloud-location", type=str, default=None, help="Vertex AI Google Cloud location")
    parser.add_argument("--if-add-node-id", type=str, default=None, help="Whether to add node id to each node")
    parser.add_argument("--if-add-node-summary", type=str, default=None, help="Whether to add summary to each node")
    parser.add_argument("--if-add-doc-description", type=str, default=None, help="Whether to add doc description")
    parser.add_argument("--if-add-node-text", type=str, default=None, help="Whether to add text to each node")
    parser.add_argument("--if-thinning", type=str, default=None, help="Whether to apply tree thinning")
    parser.add_argument("--thinning-threshold", type=int, default=None, help="Minimum token threshold for thinning")
    parser.add_argument("--summary-token-threshold", type=int, default=None, help="Token threshold for summaries")
    return parser.parse_args()


def load_options(args):
    """
    Merge CLI overrides with the package YAML config.

    Args:
        args: Parsed CLI namespace from `parse_args`.

    Returns:
        A `SimpleNamespace` config object accepted by `md_to_tree`.
    """
    user_opt = {
        "model": args.model,
        "google_cloud_project": args.google_cloud_project,
        "google_cloud_location": args.google_cloud_location,
        "if_add_node_summary": args.if_add_node_summary,
        "if_add_doc_description": args.if_add_doc_description,
        "if_add_node_text": args.if_add_node_text,
        "if_add_node_id": args.if_add_node_id,
        "if_thinning": args.if_thinning.lower() == "yes" if args.if_thinning else None,
        "min_token_threshold": args.thinning_threshold,
        "summary_token_threshold": args.summary_token_threshold,
    }
    return ConfigLoader().load({key: value for key, value in user_opt.items() if value is not None})


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
