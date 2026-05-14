import logging
import os
import textwrap
from datetime import datetime
import time
import json
import copy
import asyncio
from pathlib import Path
from types import SimpleNamespace as config

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_genai_client = None
_gemini_api_key = None


def set_gemini_api_key(api_key):
    """
    Set the Gemini API key for client initialization.

    When set, this key is used as the preferred authentication method for the
    Gen AI client. The client cache is cleared so the new key will be used on
    the next client creation.

    Args:
        api_key: Gemini API key string, or `None` to clear the key.
    """
    global _genai_client, _gemini_api_key
    _gemini_api_key = api_key
    _genai_client = None  # Clear cache to force re-initialization with new key


def _get_genai_client():
    """
    Lazily create and cache the Google Gen AI client.

    The client is shared by token counting and Gemini generation. It attempts
    authentication in this order:
    1. If `gemini_api_key` was set via `set_gemini_api_key()`, use it directly
    2. If `GOOGLE_CLOUD_PROJECT` and `GOOGLE_CLOUD_LOCATION` are set, connect to Vertex AI
    3. Otherwise, use the standard Gen AI client with default credentials

    Returns:
        A cached `google.genai.Client` instance.
    """
    global _genai_client
    if _genai_client is None:
        try:
            from google import genai
        except ImportError as exc:
            raise ImportError(
                "google-genai is required for Gemini summaries and document descriptions. "
                "Install project dependencies before indexing with LLM-backed options."
            ) from exc

        # Prefer explicitly set API key
        if _gemini_api_key:
            _genai_client = genai.Client(api_key=_gemini_api_key)
        else:
            project = os.getenv("GOOGLE_CLOUD_PROJECT")
            location = os.getenv("GOOGLE_CLOUD_LOCATION")
            if project and location:
                _genai_client = genai.Client(vertexai=True, project=project, location=location)
            else:
                _genai_client = genai.Client()
    return _genai_client


def _get_generate_config():
    """
    Build the deterministic generation config used by PageIndex summaries.

    Returns:
        A `GenerateContentConfig` with temperature set to zero.
    """
    try:
        from google.genai import types
    except ImportError as exc:
        raise ImportError("google-genai is required for Gemini generation.") from exc
    return types.GenerateContentConfig(temperature=0)


def _approximate_token_count(text):
    """
    Estimate token count when the Gemini token counter is unavailable.

    Args:
        text: Text to estimate.

    Returns:
        A conservative character-based token estimate.
    """
    return max(1, (len(text) + 3) // 4)

def count_tokens(text, model=None):
    """
    Count text tokens with Gemini, falling back to a local estimate.

    This function is used by thinning and summary decisions. It attempts the
    model-native token counter first, but remains usable without credentials so
    non-LLM indexing paths do not fail unnecessarily.

    Args:
        text: Text to count.
        model: Optional Gemini model id.

    Returns:
        Integer token count or approximate token count.
    """
    if not text:
        return 0
    model = model or os.getenv("PAGEINDEX_GEMINI_MODEL") or "gemini-2.5-flash"
    try:
        response = _get_genai_client().models.count_tokens(model=model, contents=text)
        return getattr(response, "total_tokens", None) or _approximate_token_count(text)
    except Exception as e:
        logging.debug(f"Falling back to approximate token count: {e}")
        return _approximate_token_count(text)


def llm_completion(model, prompt, chat_history=None, return_finish_reason=False):
    """
    Run a synchronous Gemini completion for document-level prompts.

    The Markdown pipeline uses this for document descriptions. Chat history is
    flattened into text because the prompts are simple and deterministic.

    Args:
        model: Gemini model id, or `None` for the package default.
        prompt: User prompt text.
        chat_history: Optional list of `{role, content}` dictionaries.
        return_finish_reason: When true, return `(content, finish_reason)`.

    Returns:
        Generated text, or a `(text, reason)` tuple when requested.
    """
    model = model or os.getenv("PAGEINDEX_GEMINI_MODEL") or "gemini-2.5-flash"
    max_retries = 10
    if chat_history:
        history_text = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in chat_history
        )
        contents = f"{history_text}\nuser: {prompt}"
    else:
        contents = prompt
    for i in range(max_retries):
        try:
            response = _get_genai_client().models.generate_content(
                model=model,
                contents=contents,
                config=_get_generate_config(),
            )
            content = response.text or ""
            if return_finish_reason:
                return content, "finished"
            return content
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                time.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                if return_finish_reason:
                    return "", "error"
                return ""



async def llm_acompletion(model, prompt):
    """
    Run an asynchronous Gemini completion for node summaries.

    Args:
        model: Gemini model id, or `None` for the package default.
        prompt: Prompt text sent to Gemini.

    Returns:
        Generated text, or an empty string after retry exhaustion.
    """
    model = model or os.getenv("PAGEINDEX_GEMINI_MODEL") or "gemini-2.5-flash"
    max_retries = 10
    for i in range(max_retries):
        try:
            response = await _get_genai_client().aio.models.generate_content(
                model=model,
                contents=prompt,
                config=_get_generate_config(),
            )
            return response.text or ""
        except Exception as e:
            print('************* Retrying *************')
            logging.error(f"Error: {e}")
            if i < max_retries - 1:
                await asyncio.sleep(1)
            else:
                logging.error('Max retries reached for prompt: ' + prompt)
                return ""
            
            
def get_json_content(response):
    """
    Extract text inside an optional fenced JSON block.

    Args:
        response: Raw LLM response that may contain ```json fences.

    Returns:
        The stripped JSON-looking content without markdown fences.
    """
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content
         

def extract_json(content):
    """
    Parse JSON from an LLM response with light cleanup.

    Args:
        content: Raw text expected to contain JSON, optionally fenced.

    Returns:
        Parsed JSON object, or `{}` when parsing fails.
    """
    try:
        # First, try to extract JSON enclosed within ```json and ```
        start_idx = content.find("```json")
        if start_idx != -1:
            start_idx += 7  # Adjust index to start after the delimiter
            end_idx = content.rfind("```")
            json_content = content[start_idx:end_idx].strip()
        else:
            # If no delimiters, assume entire content could be JSON
            json_content = content.strip()

        # Clean up common issues that might cause parsing errors
        json_content = json_content.replace('None', 'null')  # Replace Python None with JSON null
        json_content = json_content.replace('\n', ' ').replace('\r', ' ')  # Remove newlines
        json_content = ' '.join(json_content.split())  # Normalize whitespace

        # Attempt to parse and return the JSON object
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to extract JSON: {e}")
        # Try to clean up the content further if initial parsing fails
        try:
            # Remove any trailing commas before closing brackets/braces
            json_content = json_content.replace(',]', ']').replace(',}', '}')
            return json.loads(json_content)
        except:
            logging.error("Failed to parse JSON even after cleanup")
            return {}
    except Exception as e:
        logging.error(f"Unexpected error while extracting JSON: {e}")
        return {}

def write_node_id(data, node_id=0):
    """
    Recursively assign zero-padded node ids to a PageIndex tree.

    Args:
        data: Node dictionary or list of node dictionaries.
        node_id: Starting numeric id for recursive assignment.

    Returns:
        The next available numeric id after processing the tree.
    """
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id

def get_nodes(structure):
    """
    Flatten a PageIndex structure into node dictionaries without children.

    Args:
        structure: Node dictionary or list of nodes.

    Returns:
        A flat list of shallow node copies with `nodes` removed.
    """
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    
def structure_to_list(structure):
    """
    Flatten a PageIndex structure while preserving node dictionaries.

    Args:
        structure: Node dictionary or list of nodes.

    Returns:
        A preorder list of node objects. Nodes are not copied, so callers can
        mutate returned nodes to update the tree.
    """
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

    
def get_leaf_nodes(structure):
    """
    Return all leaf nodes from a PageIndex structure.

    Args:
        structure: Node dictionary or list of nodes.

    Returns:
        A list of copied leaf node dictionaries with child lists removed.
    """
    if isinstance(structure, dict):
        if not structure['nodes']:
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes

def is_leaf_node(data, node_id):
    """
    Check whether a node id points to a leaf node.

    Args:
        data: PageIndex tree or list of trees.
        node_id: Node id to locate.

    Returns:
        `True` when the node exists and has no children, otherwise `False`.
    """
    # Helper function to find the node by its node_id
    def find_node(data, node_id):
        """
        Recursively search for a node id inside a tree.

        Args:
            data: Current node or sibling list.
            node_id: Node id to find.

        Returns:
            Matching node dictionary or `None`.
        """
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get('nodes'):
        return True
    return False

def get_last_node(structure):
    """
    Return the final root-level node in a structure list.

    Args:
        structure: List of root PageIndex nodes.

    Returns:
        The last node in the list.
    """
    return structure[-1]


def sanitize_filename(filename, replacement='-'):
    """
    Replace path separators in a string intended for filenames.

    Args:
        filename: Candidate filename.
        replacement: Replacement for `/` characters.

    Returns:
        Sanitized filename string.
    """
    return filename.replace('/', replacement)


class JsonLogger:
    """
    Minimal JSON-list logger used by legacy/debug workflows.

    The logger appends dictionary-like records to an in-memory list and rewrites
    a timestamped JSON file under `./logs` on every call. It is not part of the
    main Markdown indexing path, but remains useful when debugging long-running
    flows.
    """

    def __init__(self, file_path):
        """
        Initialize the log file name from a document path.

        Args:
            file_path: Source path used to derive a readable log filename.
        """
        doc_name = sanitize_filename(Path(file_path).name)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{doc_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        self.log_data = []

    def log(self, level, message, **kwargs):
        """
        Append one log record and persist the log file.

        Args:
            level: Text level such as `INFO`, `ERROR`, or `DEBUG`.
            message: Message string or dictionary payload.
            **kwargs: Additional metadata accepted for API compatibility.
        """
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # Add new message to the log data
        
        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        """
        Log an informational record.

        Args:
            message: Message string or dictionary payload.
            **kwargs: Optional metadata.
        """
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        """
        Log an error record.

        Args:
            message: Message string or dictionary payload.
            **kwargs: Optional metadata.
        """
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        """
        Log a debug record.

        Args:
            message: Message string or dictionary payload.
            **kwargs: Optional metadata.
        """
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        """
        Log an exception-style error record.

        Args:
            message: Message string or dictionary payload.
            **kwargs: Optional metadata. The method adds `exception=True`.
        """
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        """
        Return the current log file path.

        Returns:
            Relative path under the `logs` directory.
        """
        return os.path.join("logs", self.filename)

def remove_fields(data, fields=['text']):
    """
    Recursively remove selected keys from nested structures.

    Args:
        data: Dictionary, list, or scalar value.
        fields: Field names to remove from dictionaries.

    Returns:
        A copied nested structure with the selected keys omitted.
    """
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def print_toc(tree, indent=0):
    """
    Print a simple table of contents from a PageIndex tree.

    Args:
        tree: List of PageIndex nodes.
        indent: Current indentation level for recursive printing.
    """
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)

def print_json(data, max_len=40, indent=2):
    """
    Pretty-print JSON-like data with long strings shortened.

    Args:
        data: JSON-serializable object to print.
        max_len: Maximum string length before truncation.
        indent: JSON indentation width.
    """
    def simplify_data(obj):
        """
        Recursively shorten long string values before printing.

        Args:
            obj: Current nested value.

        Returns:
            Simplified value with long strings truncated.
        """
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    """
    Remove `text` fields from a PageIndex structure in place.

    Args:
        data: Node dictionary or list of nodes.

    Returns:
        The same object with `text` fields removed.
    """
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    """
    Print nodes whose stored text exceeds a token limit.

    Args:
        structure: PageIndex tree to inspect.
        limit: Token threshold to report.
    """
    nodes = structure_to_list(structure)
    for node in nodes:
        num_tokens = count_tokens(node.get('text', ''), model=None)
        if num_tokens > limit:
            print(f"Node ID: {node.get('node_id', '?')} has {num_tokens} tokens")
            print("Line:", node.get('line_num'))
            print("Title:", node.get('title', ''))
            print("\n")


def convert_page_to_int(data):
    """
    Convert `page` values in content records from strings to integers when possible.

    Args:
        data: List of dictionaries that may contain a `page` key.

    Returns:
        The same list after best-effort conversion.
    """
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


async def generate_node_summary(node, model=None):
    """
    Generate a Gemini summary for one PageIndex node.

    Args:
        node: Node dictionary containing `text`.
        model: Gemini model id for generation.

    Returns:
        Summary text for the node.
    """
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}
    
    Directly return the description, do not include any other text.
    """
    response = await llm_acompletion(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None):
    """
    Generate a `summary` field for every node in a structure.

    Args:
        structure: PageIndex tree to update in place.
        model: Gemini model id for generation.

    Returns:
        The same structure with `summary` fields populated.
    """
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


def create_clean_structure_for_description(structure):
    """
    Create a compact structure for document description generation.

    Args:
        structure: Full PageIndex tree.

    Returns:
        A recursively filtered tree containing only fields useful for a
        one-sentence document description.
    """
    if isinstance(structure, dict):
        clean_node = {}
        # Only include essential fields for description
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        
        # Recursively process child nodes
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


def generate_doc_description(structure, model=None):
    """
    Generate a one-sentence description for an indexed document.

    Args:
        structure: Compact structure, usually from
        `create_clean_structure_for_description`.
        model: Gemini model id for generation.

    Returns:
        Description text.
    """
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
        
    Document Structure: {structure}
    
    Directly return the description, do not include any other text.
    """
    response = llm_completion(model, prompt)
    return response


def reorder_dict(data, key_order):
    """
    Return a dictionary ordered by a preferred key list.

    Args:
        data: Dictionary to reorder.
        key_order: Preferred key order.

    Returns:
        A new dictionary containing keys that exist in `data`, ordered by
        `key_order`.
    """
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    """
    Recursively reorder node fields and remove empty child lists.

    Args:
        structure: PageIndex node or list of nodes.
        order: Preferred key order for each node dictionary.

    Returns:
        The formatted structure. The input dictionaries may be mutated as part
        of recursive formatting.
    """
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    """
    Load and validate PageIndex configuration from YAML.

    The client, CLI, and indexing service use this class to merge caller
    overrides with `docindex/config.yaml` while rejecting unknown keys. That
    keeps configuration mistakes visible instead of silently ignored.
    """

    def __init__(self, default_path: str = None):
        """
        Load the default configuration file.

        Args:
            default_path: Optional path to a YAML config file. When omitted,
            `docindex/config.yaml` is used.
        """
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        """
        Read a YAML file into a dictionary.

        Args:
            path: YAML config file path.

        Returns:
            Parsed dictionary, or an empty dictionary for an empty file.
        """
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        """
        Reject user overrides that are not in the default config.

        Args:
            user_dict: Caller-provided override dictionary.

        Raises:
            ValueError: If an unknown configuration key is present.
        """
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.

        Args:
            user_opt: Optional dictionary, `SimpleNamespace`, or `None`.

        Returns:
            `SimpleNamespace` containing merged config values.
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)

def create_node_mapping(tree):
    """
    Create a flat dictionary mapping node ids to node objects.

    Args:
        tree: PageIndex tree list.

    Returns:
        Dictionary keyed by `node_id`, pointing to the original node objects.
    """
    mapping = {}
    def _traverse(nodes):
        """
        Recursively populate the mapping from a list of sibling nodes.

        Args:
            nodes: Current sibling list in the PageIndex tree.
        """
        for node in nodes:
            if node.get('node_id'):
                mapping[node['node_id']] = node
            if node.get('nodes'):
                _traverse(node['nodes'])
    _traverse(tree)
    return mapping

def print_tree(tree, indent=0):
    """
    Print a readable tree with node ids, titles, and short summaries.

    Args:
        tree: PageIndex tree list.
        indent: Current indentation depth for recursive output.
    """
    for node in tree:
        summary = node.get('summary') or node.get('prefix_summary', '')
        summary_str = f"  —  {summary[:60]}..." if summary else ""
        print('  ' * indent + f"[{node.get('node_id', '?')}] {node.get('title', '')}{summary_str}")
        if node.get('nodes'):
            print_tree(node['nodes'], indent + 1)

def print_wrapped(text, width=100):
    """
    Print text wrapped to a fixed terminal width.

    Args:
        text: Text to print.
        width: Maximum line width.
    """
    for line in text.splitlines():
        print(textwrap.fill(line, width=width))

