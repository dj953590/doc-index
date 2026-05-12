import json

try:
    from .utils import remove_fields
except ImportError:
    from utils import remove_fields


class PageSelector:
    """
    Parse user-facing retrieval selectors into Markdown header line numbers.

    Retrieval treats "pages" as source line numbers because the project is now
    Markdown-only. This small class centralizes validation for selectors used by
    the client, ADK tools, and compatibility wrapper functions.
    """

    def parse(self, pages: str) -> list[int]:
        """
        Convert a selector string into a sorted list of positive integers.

        Args:
            pages: A comma/range selector like `5-7`, `3,8`, or `12`.

        Returns:
            Sorted unique line numbers.

        Raises:
            ValueError: If the selector is empty, malformed, reversed, or
            contains non-positive line numbers.
        """
        if not isinstance(pages, str) or not pages.strip():
            raise ValueError("selector must be a non-empty string")

        result = []
        for part in pages.split(","):
            part = part.strip()
            if not part:
                raise ValueError("empty selector segment")
            if "-" in part:
                start_text, end_text = part.split("-", 1)
                start, end = int(start_text.strip()), int(end_text.strip())
                if start > end:
                    raise ValueError(f"Invalid range '{part}': start must be <= end")
                result.extend(range(start, end + 1))
            else:
                result.append(int(part))

        if any(page < 1 for page in result):
            raise ValueError("line numbers must be positive integers")
        return sorted(set(result))


class MarkdownContentRetriever:
    """
    Retrieve stored Markdown node text from an indexed PageIndex structure.

    The retriever uses line selectors produced by `PageSelector`, walks the
    nested PageIndex tree, and returns nodes whose heading line falls inside the
    requested range. It does not read the source Markdown file; it works from
    the indexed JSON structure.
    """

    def get_content(self, doc_info: dict, page_nums: list[int]) -> list[dict]:
        """
        Return node text for heading lines within the requested range.

        Args:
            doc_info: Document dictionary containing a `structure` field.
            page_nums: Positive line numbers produced by `PageSelector`.

        Returns:
            A list of dictionaries with `page` set to the heading line number
            and `content` set to the stored Markdown node text.
        """
        if not page_nums:
            return []

        min_line, max_line = min(page_nums), max(page_nums)
        results = []
        seen = set()

        def _traverse(nodes):
            """
            Recursively scan tree nodes and collect matching heading lines.

            Args:
                nodes: Current sibling list in the PageIndex tree.
            """
            for node in nodes:
                line_num = node.get("line_num")
                if line_num and min_line <= line_num <= max_line and line_num not in seen:
                    seen.add(line_num)
                    results.append({"page": line_num, "content": node.get("text", "")})
                if node.get("nodes"):
                    _traverse(node["nodes"])

        _traverse(doc_info.get("structure", []))
        results.sort(key=lambda item: item["page"])
        return results


class DocumentRepository:
    """
    Read-only JSON API over in-memory PageIndex documents.

    `PageIndexClient` and ADK tools delegate here so metadata, structure, and
    content retrieval all share consistent validation and JSON formatting.
    """

    def __init__(self, documents):
        """
        Attach a document dictionary and helper services.

        Args:
            documents: Mapping of document ids to indexed document dictionaries.
        """
        self.documents = documents
        self.selector = PageSelector()
        self.content_retriever = MarkdownContentRetriever()

    def get_document(self, doc_id: str) -> str:
        """
        Return compact metadata for one indexed Markdown document.

        Args:
            doc_id: Document id to look up.

        Returns:
            JSON string containing document metadata or an error object.
        """
        doc_info = self.documents.get(doc_id)
        if not doc_info:
            return json.dumps({"error": f"Document {doc_id} not found"})

        return json.dumps(
            {
                "doc_id": doc_id,
                "doc_name": doc_info.get("doc_name", ""),
                "doc_description": doc_info.get("doc_description", ""),
                "type": doc_info.get("type", ""),
                "status": "completed",
                "line_count": doc_info.get("line_count", 0),
            },
            ensure_ascii=False,
        )

    def get_document_structure(self, doc_id: str) -> str:
        """
        Return the document tree with text fields removed.

        Args:
            doc_id: Document id to look up.

        Returns:
            JSON string containing the structure or an error object. Text is
            removed to reduce context size for agents.
        """
        doc_info = self.documents.get(doc_id)
        if not doc_info:
            return json.dumps({"error": f"Document {doc_id} not found"})

        structure = doc_info.get("structure", [])
        structure_no_text = remove_fields(structure, fields=["text"])
        return json.dumps(structure_no_text, ensure_ascii=False)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """
        Return Markdown node text for a line selector.

        Args:
            doc_id: Document id to retrieve from.
            pages: Header line selector such as `5-7`, `3,8`, or `12`.

        Returns:
            JSON string containing content records or an error object.
        """
        doc_info = self.documents.get(doc_id)
        if not doc_info:
            return json.dumps({"error": f"Document {doc_id} not found"})

        try:
            page_nums = self.selector.parse(pages)
        except (ValueError, TypeError) as e:
            return json.dumps(
                {
                    "error": (
                        f"Invalid line selector: {pages!r}. "
                        f'Use "5-7", "3,8", or "12". Error: {e}'
                    )
                }
            )

        try:
            content = self.content_retriever.get_content(doc_info, page_nums)
        except Exception as e:
            return json.dumps({"error": f"Failed to read Markdown content: {e}"})

        return json.dumps(content, ensure_ascii=False)


def _parse_pages(pages: str) -> list[int]:
    """
    Compatibility wrapper for parsing retrieval selectors.

    Args:
        pages: Selector string accepted by `PageSelector`.

    Returns:
        Sorted unique line numbers.
    """
    return PageSelector().parse(pages)


def get_document(documents: dict, doc_id: str) -> str:
    """
    Compatibility wrapper returning metadata JSON for one document.

    Args:
        documents: Mapping of document ids to indexed documents.
        doc_id: Document id to read.

    Returns:
        JSON metadata string or error JSON.
    """
    return DocumentRepository(documents).get_document(doc_id)


def get_document_structure(documents: dict, doc_id: str) -> str:
    """
    Compatibility wrapper returning a compact PageIndex structure JSON string.

    Args:
        documents: Mapping of document ids to indexed documents.
        doc_id: Document id to read.

    Returns:
        JSON structure string with text fields removed, or error JSON.
    """
    return DocumentRepository(documents).get_document_structure(doc_id)


def get_page_content(documents: dict, doc_id: str, pages: str) -> str:
    """
    Compatibility wrapper returning Markdown content for a line selector.

    Args:
        documents: Mapping of document ids to indexed documents.
        doc_id: Document id to retrieve from.
        pages: Header line selector accepted by `PageSelector`.

    Returns:
        JSON content records or error JSON.
    """
    return DocumentRepository(documents).get_page_content(doc_id, pages)
