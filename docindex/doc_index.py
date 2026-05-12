import asyncio
import os
import re
from pathlib import Path

try:
    from .utils import (
        count_tokens,
        create_clean_structure_for_description,
        format_structure,
        generate_doc_description,
        generate_node_summary,
        print_json,
        print_toc,
        structure_to_list,
        write_node_id,
    )
except ImportError:
    from utils import (
        count_tokens,
        create_clean_structure_for_description,
        format_structure,
        generate_doc_description,
        generate_node_summary,
        print_json,
        print_toc,
        structure_to_list,
        write_node_id,
    )


HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
HEADER_LEVEL_PATTERN = re.compile(r"^(#{1,6})")
CODE_BLOCK_PATTERN = re.compile(r"^```")


class MarkdownHeaderParser:
    """
    Parse Markdown source into a flat list of heading records.

    This is the first stage of the PageIndex Markdown flow. It scans the raw
    file content, records every ATX heading (`#` through `######`) with its
    source line number, and intentionally ignores headings inside fenced code
    blocks so code samples do not become document sections.
    """

    def extract_nodes(self, markdown_content):
        """
        Return heading metadata and the original line list for a Markdown string.

        Args:
            markdown_content: Complete Markdown source text.

        Returns:
            A tuple of `(node_list, lines)`. `node_list` contains dictionaries
            with `node_title` and `line_num`; `lines` is the original document
            split on newlines for later text extraction.
        """
        node_list = []
        lines = markdown_content.split("\n")
        in_code_block = False

        for line_num, line in enumerate(lines, 1):
            stripped_line = line.strip()

            if CODE_BLOCK_PATTERN.match(stripped_line):
                in_code_block = not in_code_block
                continue

            if not stripped_line or in_code_block:
                continue

            match = HEADER_PATTERN.match(stripped_line)
            if match:
                node_list.append({"node_title": match.group(2).strip(), "line_num": line_num})

        return node_list, lines


class MarkdownContentExtractor:
    """
    Convert heading records into full section records.

    This second PageIndex stage enriches the parser output with heading depth
    and the Markdown text belonging to each section. Each node receives the text
    from its heading line up to the next heading line, preserving the algorithm's
    original section boundary behavior.
    """

    def extract_text_content(self, node_list, markdown_lines):
        """
        Attach heading level and Markdown section text to each parsed heading.

        Args:
            node_list: Flat heading records from `MarkdownHeaderParser`.
            markdown_lines: Full source document split into lines.

        Returns:
            A flat list of dictionaries containing `title`, `line_num`, `level`,
            and `text`, ready for optional thinning and tree construction.
        """
        all_nodes = []

        for node in node_list:
            line_num = node["line_num"]
            line_content = markdown_lines[line_num - 1]
            header_match = HEADER_LEVEL_PATTERN.match(line_content)

            if header_match is None:
                print(f"Warning: Line {line_num} does not contain a valid header: '{line_content}'")
                continue

            all_nodes.append(
                {
                    "title": node["node_title"],
                    "line_num": line_num,
                    "level": len(header_match.group(1)),
                }
            )

        for index, node in enumerate(all_nodes):
            start_line = node["line_num"] - 1
            end_line = all_nodes[index + 1]["line_num"] - 1 if index + 1 < len(all_nodes) else len(markdown_lines)
            node["text"] = "\n".join(markdown_lines[start_line:end_line]).strip()

        return all_nodes


class MarkdownTreeThinner:
    """
    Optionally merge small Markdown sections before building the output tree.

    Thinning is a performance and usability step for very granular Markdown.
    It calculates token counts over each node's full descendant range, then
    merges descendants into a parent when the subtree is below a threshold.
    The class keeps the same algorithmic intent as the original procedural
    implementation while avoiding repeated descendant scans.
    """

    def __init__(self, model=None):
        """
        Store the model name used by token counting.

        Args:
            model: Gemini model id used by token counting, or `None` to use the
            package default.
        """
        self.model = model

    @staticmethod
    def _descendant_end_indices(node_list):
        """
        Compute the exclusive descendant slice end for every flat node.

        Args:
            node_list: Flat section list ordered by source appearance and
            containing a numeric `level` field.

        Returns:
            A list where each index points to the first node after that node's
            descendant subtree. This lets thinning work with slices instead of
            repeatedly walking forward through the list.
        """
        ends = [len(node_list)] * len(node_list)
        stack = []

        for index, node in enumerate(node_list):
            current_level = node["level"]
            while stack and stack[-1][1] >= current_level:
                parent_index, _ = stack.pop()
                ends[parent_index] = index
            stack.append((index, current_level))

        return ends

    def add_token_counts(self, node_list):
        """
        Add `text_token_count` to each node based on its full subtree text.

        Args:
            node_list: Flat Markdown section records containing `text` and
            `level`.

        Returns:
            A copied list of nodes with `text_token_count` added. The input list
            is not mutated.
        """
        result_list = [node.copy() for node in node_list]
        descendant_ends = self._descendant_end_indices(result_list)

        for index, current_node in enumerate(result_list):
            total_text = "\n".join(
                node.get("text", "")
                for node in result_list[index:descendant_ends[index]]
                if node.get("text", "")
            )
            current_node["text_token_count"] = count_tokens(total_text, model=self.model)

        return result_list

    def thin_for_index(self, node_list, min_node_token=None):
        """
        Merge descendants into parents whose subtree token count is too small.

        Args:
            node_list: Flat Markdown section records, preferably already
            annotated by `add_token_counts`.
            min_node_token: Minimum desired token count for a retained subtree.
            When `None`, the function returns a shallow copy unchanged.

        Returns:
            A thinned flat node list that can be passed to `MarkdownTreeBuilder`.
        """
        if min_node_token is None:
            return [node.copy() for node in node_list]

        result_list = [node.copy() for node in node_list]
        descendant_ends = self._descendant_end_indices(result_list)
        nodes_to_remove = set()

        for index in range(len(result_list) - 1, -1, -1):
            if index in nodes_to_remove:
                continue

            current_node = result_list[index]
            total_tokens = current_node.get("text_token_count", 0)

            if total_tokens < min_node_token:
                children_texts = []
                for child_index in range(index + 1, descendant_ends[index]):
                    if child_index not in nodes_to_remove:
                        child_text = result_list[child_index].get("text", "")
                        if child_text.strip():
                            children_texts.append(child_text)
                        nodes_to_remove.add(child_index)

                if children_texts:
                    merged_text = current_node.get("text", "")
                    for child_text in children_texts:
                        if merged_text and not merged_text.endswith("\n"):
                            merged_text += "\n\n"
                        merged_text += child_text
                    current_node["text"] = merged_text
                    current_node["text_token_count"] = count_tokens(merged_text, model=self.model)

        for index in sorted(nodes_to_remove, reverse=True):
            result_list.pop(index)

        return result_list


class MarkdownTreeBuilder:
    """
    Build the hierarchical PageIndex tree from flat Markdown sections.

    This stage turns heading levels into parent/child relationships and assigns
    sequential node ids. It is intentionally narrow: it does not summarize,
    thin, or persist anything; it only constructs tree-shaped dictionaries used
    by retrieval and the ADK tools.
    """

    def build_tree(self, node_list):
        """
        Convert a flat ordered section list into nested PageIndex nodes.

        Args:
            node_list: Flat Markdown section records with `title`, `line_num`,
            `level`, and `text`.

        Returns:
            A list of root tree nodes. Child nodes are stored under `nodes`.
        """
        if not node_list:
            return []

        stack = []
        root_nodes = []
        node_counter = 1

        for node in node_list:
            current_level = node["level"]
            tree_node = {
                "title": node["title"],
                "node_id": str(node_counter).zfill(4),
                "text": node["text"],
                "line_num": node["line_num"],
                "nodes": [],
            }
            node_counter += 1

            while stack and stack[-1][1] >= current_level:
                stack.pop()

            if not stack:
                root_nodes.append(tree_node)
            else:
                parent_node, _ = stack[-1]
                parent_node["nodes"].append(tree_node)

            stack.append((tree_node, current_level))

        return root_nodes

    def clean_tree_for_output(self, tree_nodes):
        """
        Return a cleaned copy of tree nodes with empty child lists omitted.

        Args:
            tree_nodes: Nested tree nodes produced by `build_tree`.

        Returns:
            A recursively cleaned tree containing public output fields.
        """
        cleaned_nodes = []

        for node in tree_nodes:
            cleaned_node = {
                "title": node["title"],
                "node_id": node["node_id"],
                "text": node["text"],
                "line_num": node["line_num"],
            }

            if node["nodes"]:
                cleaned_node["nodes"] = self.clean_tree_for_output(node["nodes"])

            cleaned_nodes.append(cleaned_node)

        return cleaned_nodes


class MarkdownSummaryService:
    """
    Generate Gemini-backed summaries for PageIndex nodes.

    This service is used only when callers request node summaries. It keeps LLM
    concurrency bounded so large Markdown files do not start an unbounded number
    of Gemini requests at once, and it preserves the original shortcut of using
    raw text as the summary when a node is already small.
    """

    def __init__(self, model=None, max_concurrency=8):
        """
        Configure the summary model and concurrency limit.

        Args:
            model: Gemini model id used for summaries.
            max_concurrency: Maximum simultaneous asynchronous summary calls.
        """
        self.model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def get_node_summary(self, node, summary_token_threshold=200):
        """
        Return raw text for small nodes or generate a summary for large nodes.

        Args:
            node: PageIndex node containing `text`.
            summary_token_threshold: Token count below which raw text is reused.

        Returns:
            A string summary suitable for `summary` or `prefix_summary`.
        """
        if summary_token_threshold is None:
            summary_token_threshold = 200
        node_text = node.get("text", "")
        num_tokens = count_tokens(node_text, model=self.model)
        if num_tokens < summary_token_threshold:
            return node_text

        async with self._semaphore:
            return await generate_node_summary(node, model=self.model)

    async def generate_summaries_for_structure(self, structure, summary_token_threshold):
        """
        Populate summaries across an existing PageIndex tree.

        Args:
            structure: Root node list or node dictionary to summarize in place.
            summary_token_threshold: Threshold passed to `get_node_summary`.

        Returns:
            The same structure with leaf `summary` fields and internal
            `prefix_summary` fields populated.
        """
        nodes = structure_to_list(structure)
        tasks = [
            self.get_node_summary(node, summary_token_threshold=summary_token_threshold)
            for node in nodes
        ]
        summaries = await asyncio.gather(*tasks)

        for node, summary in zip(nodes, summaries):
            if not node.get("nodes"):
                node["summary"] = summary
            else:
                node["prefix_summary"] = summary

        return structure


class MarkdownDocumentIndexer:
    """
    Orchestrate the full Markdown-to-PageIndex pipeline.

    This is the main class-level entry point for Markdown indexing. It wires
    together parsing, section text extraction, optional thinning, tree building,
    node id writing, optional Gemini summaries, and optional document
    description generation. The public `md_to_tree` function delegates here for
    backward compatibility.
    """

    def __init__(
        self,
        model=None,
        parser=None,
        extractor=None,
        thinner=None,
        builder=None,
        summary_service=None,
    ):
        """
        Build an indexer from replaceable pipeline components.

        Args:
            model: Gemini model id used for token counting and summaries.
            parser: Optional custom heading parser.
            extractor: Optional custom content extractor.
            thinner: Optional custom thinning implementation.
            builder: Optional custom tree builder.
            summary_service: Optional custom summary service.
        """
        self.model = model
        self.parser = parser or MarkdownHeaderParser()
        self.extractor = extractor or MarkdownContentExtractor()
        self.thinner = thinner or MarkdownTreeThinner(model=model)
        self.builder = builder or MarkdownTreeBuilder()
        self.summary_service = summary_service or MarkdownSummaryService(model=model)

    async def index(
        self,
        md_path,
        if_thinning=False,
        min_token_threshold=None,
        if_add_node_summary="no",
        summary_token_threshold=None,
        if_add_doc_description="no",
        if_add_node_text="no",
        if_add_node_id="yes",
    ):
        """
        Index a Markdown file into the PageIndex JSON-compatible structure.

        Args:
            md_path: Path to the Markdown file.
            if_thinning: Whether to merge small nodes before building the tree.
            min_token_threshold: Token threshold used when thinning is enabled.
            if_add_node_summary: `"yes"` to add summaries to nodes.
            summary_token_threshold: Token threshold for raw-text vs LLM summary.
            if_add_doc_description: `"yes"` to add a one-sentence description.
            if_add_node_text: `"yes"` to keep source text in output nodes.
            if_add_node_id: `"yes"` to rewrite stable zero-padded node ids.

        Returns:
            A dictionary with `doc_name`, `line_count`, and `structure`, plus
            `doc_description` when requested.
        """
        path = Path(md_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Markdown file not found: {path}")

        markdown_content = path.read_text(encoding="utf-8")
        line_count = markdown_content.count("\n") + 1

        print("Extracting nodes from markdown...")
        node_list, markdown_lines = self.parser.extract_nodes(markdown_content)

        print("Extracting text content from nodes...")
        nodes_with_content = self.extractor.extract_text_content(node_list, markdown_lines)

        if if_thinning:
            nodes_with_content = self.thinner.add_token_counts(nodes_with_content)
            print("Thinning nodes...")
            nodes_with_content = self.thinner.thin_for_index(nodes_with_content, min_token_threshold)

        print("Building tree from nodes...")
        tree_structure = self.builder.build_tree(nodes_with_content)

        if if_add_node_id == "yes":
            write_node_id(tree_structure)

        print("Formatting tree structure...")

        if if_add_node_summary == "yes":
            tree_structure = format_structure(
                tree_structure,
                order=["title", "node_id", "line_num", "summary", "prefix_summary", "text", "nodes"],
            )

            print("Generating summaries for each node...")
            tree_structure = await self.summary_service.generate_summaries_for_structure(
                tree_structure,
                summary_token_threshold=summary_token_threshold,
            )

            if if_add_node_text == "no":
                tree_structure = format_structure(
                    tree_structure,
                    order=["title", "node_id", "line_num", "summary", "prefix_summary", "nodes"],
                )

            if if_add_doc_description == "yes":
                print("Generating document description...")
                clean_structure = create_clean_structure_for_description(tree_structure)
                doc_description = generate_doc_description(clean_structure, model=self.model)
                return {
                    "doc_name": path.stem,
                    "doc_description": doc_description,
                    "line_count": line_count,
                    "structure": tree_structure,
                }
        else:
            if if_add_node_text == "yes":
                tree_structure = format_structure(
                    tree_structure,
                    order=["title", "node_id", "line_num", "summary", "prefix_summary", "text", "nodes"],
                )
            else:
                tree_structure = format_structure(
                    tree_structure,
                    order=["title", "node_id", "line_num", "summary", "prefix_summary", "nodes"],
                )

        return {
            "doc_name": path.stem,
            "line_count": line_count,
            "structure": tree_structure,
        }


async def get_node_summary(node, summary_token_threshold=200, model=None):
    """
    Compatibility wrapper for generating a single Markdown node summary.

    Args:
        node: PageIndex node containing `text`.
        summary_token_threshold: Token count below which raw text is reused.
        model: Gemini model id for summary generation.

    Returns:
        A summary string.
    """
    return await MarkdownSummaryService(model=model).get_node_summary(
        node,
        summary_token_threshold=summary_token_threshold,
    )


async def generate_summaries_for_structure_md(structure, summary_token_threshold, model=None):
    """
    Compatibility wrapper for summarizing every node in a PageIndex structure.

    Args:
        structure: Tree structure to mutate with summaries.
        summary_token_threshold: Token threshold for summary generation.
        model: Gemini model id for summary calls.

    Returns:
        The input structure with summary fields populated.
    """
    return await MarkdownSummaryService(model=model).generate_summaries_for_structure(
        structure,
        summary_token_threshold=summary_token_threshold,
    )


def extract_nodes_from_markdown(markdown_content):
    """
    Compatibility wrapper around `MarkdownHeaderParser.extract_nodes`.

    Args:
        markdown_content: Complete Markdown source text.

    Returns:
        `(node_list, lines)` for downstream processing.
    """
    return MarkdownHeaderParser().extract_nodes(markdown_content)


def extract_node_text_content(node_list, markdown_lines):
    """
    Compatibility wrapper around `MarkdownContentExtractor.extract_text_content`.

    Args:
        node_list: Flat heading records.
        markdown_lines: Full document split into lines.

    Returns:
        Flat section records with text and heading levels.
    """
    return MarkdownContentExtractor().extract_text_content(node_list, markdown_lines)


def update_node_list_with_text_token_count(node_list, model=None):
    """
    Compatibility wrapper for adding subtree token counts to Markdown nodes.

    Args:
        node_list: Flat section records.
        model: Gemini model id used for token counting.

    Returns:
        A copied list with `text_token_count` fields.
    """
    return MarkdownTreeThinner(model=model).add_token_counts(node_list)


def tree_thinning_for_index(node_list, min_node_token=None, model=None):
    """
    Compatibility wrapper for the Markdown tree thinning step.

    Args:
        node_list: Flat section records.
        min_node_token: Minimum desired subtree token count.
        model: Gemini model id used for token counting merged text.

    Returns:
        A thinned flat section list.
    """
    return MarkdownTreeThinner(model=model).thin_for_index(node_list, min_node_token)


def build_tree_from_nodes(node_list):
    """
    Compatibility wrapper for building the nested PageIndex tree.

    Args:
        node_list: Flat Markdown section records.

    Returns:
        Nested tree nodes.
    """
    return MarkdownTreeBuilder().build_tree(node_list)


def clean_tree_for_output(tree_nodes):
    """
    Compatibility wrapper for cleaning nested tree output.

    Args:
        tree_nodes: Nested PageIndex nodes.

    Returns:
        Cleaned nested nodes without empty child lists.
    """
    return MarkdownTreeBuilder().clean_tree_for_output(tree_nodes)


async def md_to_tree(
    md_path,
    if_thinning=False,
    min_token_threshold=None,
    if_add_node_summary="no",
    summary_token_threshold=None,
    model=None,
    if_add_doc_description="no",
    if_add_node_text="no",
    if_add_node_id="yes",
):
    """
    Convert a Markdown file into a PageIndex tree dictionary.

    This is the stable public API kept from the original project. It delegates
    to `MarkdownDocumentIndexer` so callers can continue using the function
    while the internals remain organized into classes.

    Args:
        md_path: Path to the Markdown file.
        if_thinning: Whether to merge small nodes before tree construction.
        min_token_threshold: Token threshold for thinning.
        if_add_node_summary: `"yes"` to add node summaries.
        summary_token_threshold: Token threshold for summary generation.
        model: Gemini model id for token counting and summaries.
        if_add_doc_description: `"yes"` to add a document description.
        if_add_node_text: `"yes"` to include node source text in output.
        if_add_node_id: `"yes"` to rewrite node ids.

    Returns:
        A JSON-serializable PageIndex document dictionary.
    """
    return await MarkdownDocumentIndexer(model=model).index(
        md_path=md_path,
        if_thinning=if_thinning,
        min_token_threshold=min_token_threshold,
        if_add_node_summary=if_add_node_summary,
        summary_token_threshold=summary_token_threshold,
        if_add_doc_description=if_add_doc_description,
        if_add_node_text=if_add_node_text,
        if_add_node_id=if_add_node_id,
    )


if __name__ == "__main__":
    import json

    MD_NAME = "cognitive-load"
    MD_PATH = os.path.join(os.path.dirname(__file__), "..", "examples/documents/", f"{MD_NAME}.md")

    MODEL = "gemini-2.5-flash"
    IF_THINNING = False
    THINNING_THRESHOLD = 5000
    SUMMARY_TOKEN_THRESHOLD = 200
    IF_SUMMARY = True

    tree_structure = asyncio.run(
        md_to_tree(
            md_path=MD_PATH,
            if_thinning=IF_THINNING,
            min_token_threshold=THINNING_THRESHOLD,
            if_add_node_summary="yes" if IF_SUMMARY else "no",
            summary_token_threshold=SUMMARY_TOKEN_THRESHOLD,
            model=MODEL,
        )
    )

    print("\n" + "=" * 60)
    print("TREE STRUCTURE")
    print("=" * 60)
    print_json(tree_structure)

    print("\n" + "=" * 60)
    print("TABLE OF CONTENTS")
    print("=" * 60)
    print_toc(tree_structure["structure"])

    output_path = os.path.join(os.path.dirname(__file__), "..", "results", f"{MD_NAME}_structure.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tree_structure, f, indent=2, ensure_ascii=False)

    print(f"\nTree structure saved to: {output_path}")
