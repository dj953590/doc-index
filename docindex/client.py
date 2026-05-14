import asyncio
import concurrent.futures
import json
import os
import uuid
from pathlib import Path

from .doc_index import md_to_tree
from .retrieve import get_document, get_document_structure, get_page_content
from .utils import ConfigLoader, set_gemini_api_key

META_INDEX = "_meta.json"
MARKDOWN_EXTENSIONS = {".md", ".markdown"}


def _normalize_retrieve_model(model: str) -> str:
    """
    Normalize legacy provider-prefixed model names to Gemini model ids.

    Args:
        model: User-supplied model string, possibly prefixed with `google/` or
        `vertex_ai/` from older integrations.

    Returns:
        A model id that Google Gen AI and ADK can pass directly to Gemini.
    """
    if not model:
        return model
    for prefix in ("google/", "vertex_ai/"):
        if model.startswith(prefix):
            return model.removeprefix(prefix)
    return model


def _run_async(coro):
    """
    Run a coroutine from synchronous client methods.

    The public `PageIndexClient.index` method is synchronous, while Markdown
    indexing can call async Gemini summary functions. This helper bridges those
    worlds and uses a worker thread when the caller already has an event loop.

    Args:
        coro: Coroutine to execute.

    Returns:
        The coroutine result.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


class WorkspaceStore:
    """
    Persist indexed Markdown documents on disk for reuse across sessions.

    `PageIndexClient` keeps only lightweight metadata in memory after saving a
    document. This store owns the JSON files, the `_meta.json` index, corruption
    tolerant loading, and lazy loading of full structures when retrieval asks
    for them.
    """

    def __init__(self, workspace):
        """
        Create the workspace directory if needed.

        Args:
            workspace: Directory where document JSON files and `_meta.json`
            should be stored.
        """
        self.workspace = Path(workspace).expanduser()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def load(self):
        """
        Load document metadata from the workspace.

        Returns:
            A dictionary keyed by document id. Each value is a lightweight
            document metadata record suitable for `PageIndexClient.documents`.
        """
        meta = self._read_meta()
        if meta is None:
            meta = self._rebuild_meta()
            if meta:
                print(f"Loaded {len(meta)} document(s) from workspace (legacy mode).")

        documents = {}
        for doc_id, entry in meta.items():
            doc = dict(entry, id=doc_id)
            if doc.get("path") and not os.path.isabs(doc["path"]):
                doc["path"] = str((self.workspace / doc["path"]).resolve())
            documents[doc_id] = doc
        return documents

    def save(self, doc_id, doc):
        """
        Persist one full document and update the workspace metadata index.

        Args:
            doc_id: Stable document identifier assigned by `PageIndexClient`.
            doc: Full document dictionary containing structure and metadata.
        """
        path = self.workspace / f"{doc_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
        self._save_meta(doc_id, self.make_meta_entry(doc))

    def load_full_doc(self, doc_id):
        """
        Load a full stored document by id.

        Args:
            doc_id: Document identifier whose JSON file should be read.

        Returns:
            The full document dictionary, or `None` if the file is missing or
            corrupt.
        """
        return self._read_json(self.workspace / f"{doc_id}.json")

    @staticmethod
    def make_meta_entry(doc):
        """
        Build the lightweight metadata entry saved in `_meta.json`.

        Args:
            doc: Full document dictionary.

        Returns:
            A small dictionary with display metadata and line count.
        """
        return {
            "type": doc.get("type", ""),
            "doc_name": doc.get("doc_name", ""),
            "doc_description": doc.get("doc_description", ""),
            "path": doc.get("path", ""),
            "line_count": doc.get("line_count", 0),
        }

    def _rebuild_meta(self):
        """
        Reconstruct metadata by scanning document JSON files.

        Returns:
            A metadata dictionary keyed by document id. This is used when
            `_meta.json` is missing or invalid.
        """
        meta = {}
        for path in self.workspace.glob("*.json"):
            if path.name == META_INDEX:
                continue
            doc = self._read_json(path)
            if doc and isinstance(doc, dict):
                meta[path.stem] = self.make_meta_entry(doc)
        return meta

    def _read_meta(self):
        """
        Read and validate the workspace metadata file.

        Returns:
            The metadata dictionary, or `None` when missing or invalid.
        """
        meta = self._read_json(self.workspace / META_INDEX)
        if meta is not None and not isinstance(meta, dict):
            print(f"Warning: {META_INDEX} is not a JSON object, ignoring")
            return None
        return meta

    def _save_meta(self, doc_id, entry):
        """
        Merge and write one metadata entry into `_meta.json`.

        Args:
            doc_id: Document identifier to update.
            entry: Lightweight metadata entry from `make_meta_entry`.
        """
        meta = self._read_meta() or self._rebuild_meta()
        meta[doc_id] = entry
        meta_path = self.workspace / META_INDEX
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _read_json(path):
        """
        Read JSON while treating missing/corrupt files as recoverable.

        Args:
            path: JSON file path.

        Returns:
            Parsed JSON data, or `None` if the file cannot be read safely.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: corrupt {Path(path).name}: {e}")
            return None


class MarkdownIndexService:
    """
    Thin service wrapper around the Markdown indexing algorithm.

    `PageIndexClient` owns document ids, persistence, and retrieval; this class
    owns converting a Markdown path plus loaded config into the PageIndex
    structure returned by `md_to_tree`.
    """

    def __init__(self, model, opt):
        """
        Store indexing configuration.

        Args:
            model: Gemini model used by optional summaries/descriptions.
            opt: Loaded configuration namespace from `ConfigLoader`.
        """
        self.model = model
        self.opt = opt

    def index_file(self, file_path):
        """
        Build a PageIndex structure for one Markdown file.

        Args:
            file_path: Absolute or relative path to a Markdown file.

        Returns:
            The dictionary returned by `md_to_tree`.
        """
        result = _run_async(
            md_to_tree(
                md_path=file_path,
                if_thinning=getattr(self.opt, "if_thinning", False),
                min_token_threshold=getattr(self.opt, "min_token_threshold", None),
                if_add_node_summary=getattr(self.opt, "if_add_node_summary", "yes"),
                summary_token_threshold=getattr(self.opt, "summary_token_threshold", 200),
                model=self.model,
                if_add_doc_description=getattr(self.opt, "if_add_doc_description", "yes"),
                if_add_node_text=getattr(self.opt, "if_add_node_text", "yes"),
                if_add_node_id=getattr(self.opt, "if_add_node_id", "yes"),
            )
        )
        return result


class PageIndexClient:
    """
    Main synchronous API for Markdown PageIndex indexing and retrieval.

    The client coordinates configuration, optional Vertex environment setup,
    Markdown indexing, workspace persistence, and retrieval wrappers. Typical
    flow is `index(path)` to create a document id, followed by `get_document`,
    `get_document_structure`, and `get_page_content` for agent/tool use.
    """

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        retrieve_model: str = None,
        workspace: str = None,
        vertex_project: str = None,
        vertex_location: str = None,
    ):
        """
        Configure a client instance.

        Args:
            api_key: Optional Gemini API key for non-Vertex Gen AI calls.
            model: Gemini model used while indexing.
            retrieve_model: Gemini model intended for retrieval agents.
            workspace: Optional directory for persistent JSON indexes.
            vertex_project: Optional Google Cloud project for Vertex AI.
            vertex_location: Optional Vertex AI location; defaults to
            `us-central1` when a project is supplied without a location.
        """
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            set_gemini_api_key(api_key)
        if vertex_project and not vertex_location:
            vertex_location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        if vertex_project:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            os.environ["GOOGLE_CLOUD_PROJECT"] = vertex_project
        if vertex_location:
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            os.environ["GOOGLE_CLOUD_LOCATION"] = vertex_location
        overrides = {}
        if model:
            overrides["model"] = model
        if retrieve_model:
            overrides["retrieve_model"] = retrieve_model
        if vertex_project:
            overrides["google_cloud_project"] = vertex_project
        if vertex_location:
            overrides["google_cloud_location"] = vertex_location

        self.opt = ConfigLoader().load(overrides or None)
        if self.opt.google_cloud_project and not os.getenv("GOOGLE_CLOUD_PROJECT"):
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            os.environ["GOOGLE_CLOUD_PROJECT"] = self.opt.google_cloud_project
        if self.opt.google_cloud_location and not os.getenv("GOOGLE_CLOUD_LOCATION"):
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
            os.environ["GOOGLE_CLOUD_LOCATION"] = self.opt.google_cloud_location
        self.model = self.opt.model
        self.retrieve_model = _normalize_retrieve_model(self.opt.retrieve_model or self.model)
        self.store = WorkspaceStore(workspace) if workspace else None
        self.index_service = MarkdownIndexService(self.model, self.opt)
        self.documents = self.store.load() if self.store else {}

    def index(self, file_path: str, mode: str = "auto") -> str:
        """
        Index a Markdown document and register it in the client.

        Args:
            file_path: Path to a `.md` or `.markdown` file.
            mode: `auto`, `md`, or `markdown`. Other modes are rejected because
            PDF processing has been removed.

        Returns:
            A generated document id used by retrieval methods.
        """
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        if mode not in {"auto", "md", "markdown"}:
            raise ValueError("PDF processing has been removed. Use mode='md' or mode='auto' with a Markdown file.")
        if ext not in MARKDOWN_EXTENSIONS:
            raise ValueError(f"Unsupported file format for Markdown indexing: {file_path}")

        doc_id = str(uuid.uuid4())
        print(f"Indexing Markdown: {file_path}")
        result = self.index_service.index_file(file_path)

        self.documents[doc_id] = {
            "id": doc_id,
            "type": "md",
            "path": file_path,
            "doc_name": result.get("doc_name", ""),
            "doc_description": result.get("doc_description", ""),
            "line_count": result.get("line_count", 0),
            "structure": result["structure"],
        }

        print(f"Indexing complete. Document ID: {doc_id}")
        if self.store:
            self._save_doc(doc_id)
        return doc_id

    def _save_doc(self, doc_id):
        """
        Persist a full document then keep only metadata in memory.

        Args:
            doc_id: Document identifier already present in `self.documents`.
        """
        doc = self.documents[doc_id].copy()
        self.store.save(doc_id, doc)
        self.documents[doc_id].pop("structure", None)

    def _ensure_doc_loaded(self, doc_id: str):
        """
        Lazy-load a document structure from the workspace when needed.

        Args:
            doc_id: Document identifier requested by retrieval methods.
        """
        doc = self.documents.get(doc_id)
        if not self.store or not doc or doc.get("structure") is not None:
            return
        full = self.store.load_full_doc(doc_id)
        if full:
            doc["structure"] = full.get("structure", [])

    def get_document(self, doc_id: str) -> str:
        """
        Return document metadata as a JSON string.

        Args:
            doc_id: Document identifier returned by `index`.

        Returns:
            JSON string containing status, name, description, and line count.
        """
        return get_document(self.documents, doc_id)

    def get_document_structure(self, doc_id: str) -> str:
        """
        Return the PageIndex structure without node text.

        Args:
            doc_id: Document identifier returned by `index`.

        Returns:
            JSON string containing the tree structure with `text` removed to
            keep agent context compact.
        """
        self._ensure_doc_loaded(doc_id)
        return get_document_structure(self.documents, doc_id)

    def get_page_content(self, doc_id: str, pages: str) -> str:
        """
        Return Markdown node content for header line selectors.

        Args:
            doc_id: Document identifier returned by `index`.
            pages: Line selector such as `5-7`, `3,8`, or `12`.

        Returns:
            JSON string of retrieved content records.
        """
        self._ensure_doc_loaded(doc_id)
        return get_page_content(self.documents, doc_id, pages)
