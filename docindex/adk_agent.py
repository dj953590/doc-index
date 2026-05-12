import asyncio
import json
import os
from pathlib import Path

from .doc_index import md_to_tree
from .retrieve import get_page_content
from .utils import remove_fields


APP_NAME = "pageindex_markdown_agent"
USER_ID = "pageindex_user"
SESSION_ID = "pageindex_session"


def _run_async(coro):
    """
    Execute an async indexing coroutine from synchronous ADK tool functions.

    ADK function tools are plain synchronous Python functions in this module,
    while `md_to_tree` is async. This helper runs the coroutine directly when no
    loop is active and uses a temporary worker thread when called from inside an
    already-running event loop.

    Args:
        coro: Coroutine to execute.

    Returns:
        The coroutine result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _candidate_index_paths(markdown_path):
    """
    Return the standard locations where a PageIndex JSON may live.

    Args:
        markdown_path: Source Markdown file path.

    Returns:
        A prioritized list of candidate JSON paths. The first path is also the
        default creation target when no existing index is found.
    """
    markdown = Path(markdown_path).expanduser().resolve()
    return [
        markdown.parent / "results" / f"{markdown.stem}_structure.json",
        Path.cwd() / "results" / f"{markdown.stem}_structure.json",
        markdown.with_suffix(".docindex.json"),
    ]


def _resolve_index_path(markdown_path, index_path=None):
    """
    Choose the PageIndex JSON path for a Markdown file.

    Args:
        markdown_path: Source Markdown file path.
        index_path: Optional explicit JSON path provided by the caller.

    Returns:
        The explicit path, the first existing conventional path, or the default
        conventional output path.
    """
    if index_path:
        return Path(index_path).expanduser().resolve()

    for candidate in _candidate_index_paths(markdown_path):
        if candidate.is_file():
            return candidate

    return _candidate_index_paths(markdown_path)[0]


def _load_index(index_path):
    """
    Load a PageIndex JSON file from disk.

    Args:
        index_path: Path to the JSON index.

    Returns:
        A tuple of `(resolved_path, parsed_json)`.
    """
    path = Path(index_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return path, data


def ensure_pageindex_json(markdown_path: str, index_path: str = "") -> dict:
    """
    Ensure a Markdown PageIndex JSON file exists.

    Args:
        markdown_path: Path to the source Markdown file.
        index_path: Optional explicit JSON index path. If omitted, existing common
            paths are checked and results/<markdown_stem>_structure.json is used.

    Returns:
        A dictionary with status, index_path, doc_name, line_count, and created.
        ADK agents call this first so later retrieval tools can operate on a
        stable JSON file rather than rebuilding the index each time.
    """
    markdown = Path(markdown_path).expanduser().resolve()
    if not markdown.is_file():
        return {"status": "error", "error": f"Markdown file not found: {markdown}"}
    if markdown.suffix.lower() not in {".md", ".markdown"}:
        return {"status": "error", "error": f"Unsupported Markdown file extension: {markdown.suffix}"}

    resolved_index_path = _resolve_index_path(markdown, index_path or None)
    if resolved_index_path.is_file():
        _, existing = _load_index(resolved_index_path)
        return {
            "status": "success",
            "created": False,
            "index_path": str(resolved_index_path),
            "doc_name": existing.get("doc_name", markdown.stem),
            "line_count": existing.get("line_count", 0),
        }

    result = _run_async(
        md_to_tree(
            md_path=str(markdown),
            if_thinning=False,
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="yes",
            if_add_node_id="yes",
        )
    )
    resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_index_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return {
        "status": "success",
        "created": True,
        "index_path": str(resolved_index_path),
        "doc_name": result.get("doc_name", markdown.stem),
        "line_count": result.get("line_count", 0),
    }


def get_pageindex_structure(index_path: str, include_text: bool = False) -> dict:
    """
    Retrieve the PageIndex document structure.

    Args:
        index_path: Path to a PageIndex JSON file.
        include_text: Include node text when true. Keep false for structure scans.

    Returns:
        A dictionary with status, doc_name, line_count, and structure.
        Agents usually call this with `include_text=False` to select likely
        sections while preserving model context.
    """
    try:
        path, data = _load_index(index_path)
        structure = data.get("structure", [])
        if not include_text:
            structure = remove_fields(structure, fields=["text"])
        return {
            "status": "success",
            "index_path": str(path),
            "doc_name": data.get("doc_name", ""),
            "line_count": data.get("line_count", 0),
            "structure": structure,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


def retrieve_markdown_content(index_path: str, lines: str) -> dict:
    """
    Retrieve Markdown node content by header line selector.

    Args:
        index_path: Path to a PageIndex JSON file.
        lines: Header line selector such as "5-7", "3,8", or "12".

    Returns:
        A dictionary containing retrieved content records.
        Agents call this after identifying promising line numbers from the
        structure or search result.
    """
    try:
        path, data = _load_index(index_path)
        documents = {
            "doc": {
                "type": "md",
                "doc_name": data.get("doc_name", ""),
                "line_count": data.get("line_count", 0),
                "structure": data.get("structure", []),
            }
        }
        content = json.loads(get_page_content(documents, "doc", lines))
        return {"status": "success", "index_path": str(path), "content": content}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def search_pageindex(index_path: str, query: str, max_results: int = 5) -> dict:
    """
    Keyword search over PageIndex nodes for quick candidate retrieval.

    Args:
        index_path: Path to a PageIndex JSON file.
        query: Search text.
        max_results: Maximum number of matching nodes to return.

    Returns:
        A dictionary with matching node titles, line numbers, and snippets.
        This is a lightweight lexical helper for candidate discovery; final
        answers should still use `retrieve_markdown_content` for source text.
    """
    try:
        path, data = _load_index(index_path)
        terms = [term.lower() for term in query.split() if len(term) > 2]
        if not terms:
            return {"status": "success", "index_path": str(path), "matches": []}

        matches = []

        def visit(nodes):
            """
            Recursively score nodes and append lexical matches.

            Args:
                nodes: Current sibling list in the PageIndex tree.
            """
            for node in nodes:
                text = f"{node.get('title', '')}\n{node.get('summary', '')}\n{node.get('prefix_summary', '')}\n{node.get('text', '')}"
                lower_text = text.lower()
                score = sum(lower_text.count(term) for term in terms)
                if score:
                    snippet = node.get("text", "")[:700]
                    matches.append(
                        {
                            "title": node.get("title", ""),
                            "line_num": node.get("line_num"),
                            "node_id": node.get("node_id"),
                            "score": score,
                            "snippet": snippet,
                        }
                    )
                if node.get("nodes"):
                    visit(node["nodes"])

        visit(data.get("structure", []))
        matches.sort(key=lambda item: item["score"], reverse=True)
        return {"status": "success", "index_path": str(path), "matches": matches[:max_results]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def create_pageindex_agent(model: str = "gemini-2.5-flash"):
    """
    Create the Google ADK agent used for Markdown PageIndex question answering.

    The agent is configured with four tools: create/load the JSON index, inspect
    the compact structure, search candidate sections, and retrieve node text.
    It is the programmatic factory used by `root_agent` and by
    `answer_with_pageindex_agent`.

    Args:
        model: Gemini model id passed to ADK.

    Returns:
        A configured `google.adk.agents.llm_agent.Agent` instance.
    """
    try:
        from google.adk.agents.llm_agent import Agent
    except ImportError as exc:
        raise ImportError("google-adk is required to create the PageIndex ADK agent.") from exc

    return Agent(
        name="pageindex_markdown_agent",
        model=model,
        description="Indexes Markdown files into PageIndex JSON and answers questions by retrieving indexed content.",
        instruction=(
            "You answer questions about a Markdown document using PageIndex tools. "
            "First call ensure_pageindex_json with the Markdown path supplied by the user. "
            "Then inspect get_pageindex_structure or search_pageindex to choose relevant sections. "
            "Use retrieve_markdown_content for the selected line numbers before answering. "
            "Cite section titles or line numbers when useful. If the index is missing, create it."
        ),
        tools=[
            ensure_pageindex_json,
            get_pageindex_structure,
            retrieve_markdown_content,
            search_pageindex,
        ],
    )


async def answer_with_pageindex_agent(markdown_path: str, query: str, model: str = "gemini-2.5-flash") -> str:
    """
    Run a one-shot ADK question-answering session over a Markdown file.

    This helper is useful for scripts and examples. It creates an in-memory ADK
    session, sends a prompt containing the Markdown path and user query, lets the
    agent call PageIndex tools as needed, and returns the final text response.

    Args:
        markdown_path: Source Markdown file the agent should index/retrieve.
        query: User question to answer from the Markdown document.
        model: Gemini model id passed to the ADK agent.

    Returns:
        Final text answer produced by the agent.
    """
    try:
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService
        from google.genai import types
    except ImportError as exc:
        raise ImportError("google-adk and google-genai are required to run the PageIndex ADK agent.") from exc

    agent = create_pageindex_agent(model=model)
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)
    prompt = f"Markdown file: {markdown_path}\nUser question: {query}"
    content = types.Content(role="user", parts=[types.Part(text=prompt)])
    events = runner.run_async(user_id=USER_ID, session_id=session.id, new_message=content)

    final_response = ""
    async for event in events:
        if event.is_final_response() and event.content and event.content.parts:
            final_response = event.content.parts[0].text or ""
    return final_response


try:
    root_agent = create_pageindex_agent(model=os.getenv("PAGEINDEX_AGENT_MODEL", "gemini-2.5-flash"))
except ImportError:
    root_agent = None
