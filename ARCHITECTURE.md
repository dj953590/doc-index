# Architecture Summary: doc-index Project

## Quick Overview

**doc-index** is a document indexing and retrieval framework that converts Markdown files into hierarchical tree structures for intelligent Q&A without vector databases.

---

## The Three Approaches

### 1. **ADK Agent (Simplest - AI Orchestration)**
```
Your Question
    ↓
AI Agent decides what to do
    ↓
- Index document if needed
- Search relevant sections
- Retrieve content
- Synthesize answer
    ↓
Answer
```
**Entry Point:** `answer_with_pageindex_agent(markdown_path, query)`  
**Use When:** You want automatic, intelligent Q&A  
**File:** `examples/agent_qa_example.py`

---

### 2. **Direct Orchestration (Full Control - Programmatic)**
```
Markdown File
    ↓
[YOU CONTROL EACH STEP]
    ├─ Index with MarkdownDocumentIndexer
    ├─ Browse with print_tree()
    ├─ Retrieve with DocumentRepository
    ├─ Analyze with PageSelector
    └─ Export/Process as needed
    ↓
Your Custom Workflow
```
**Entry Point:** `MarkdownDocumentIndexer().index()` + `DocumentRepository()`  
**Use When:** You need custom logic and deterministic behavior  
**File:** `examples/direct_orchestration_example.py`

---

### 3. **CLI (Quick Command-Line)**
```bash
python app.py --md_path "document.md" --gemini_api_key "key"
```
**Entry Point:** `app.py`  
**Use When:** Simple one-off indexing from command line

---

## Architecture Layers

### Layer 1: Parsing (Input → Flat List)
```
Raw Markdown
    ↓ MarkdownHeaderParser
    └─ Extract headers with line numbers
    ↓ MarkdownContentExtractor
    └─ Attach text + heading levels
    ↓
Flat node list (ready for tree building)
```

### Layer 2: Optimization (Flat List → Flat List)
```
Flat node list
    ↓ MarkdownTreeThinner (OPTIONAL)
    ├─ Count tokens in each subtree
    └─ Merge nodes < min_token_threshold
    ↓
Optimized flat node list
```

### Layer 3: Structuring (Flat List → Tree)
```
Flat node list
    ↓ MarkdownTreeBuilder
    ├─ Use stack algorithm
    ├─ Convert levels → parent-child relationships
    └─ Assign zero-padded node IDs
    ↓
Hierarchical tree
```

### Layer 4: Enhancement (Tree → Enhanced Tree)
```
Hierarchical tree
    ↓ MarkdownSummaryService (OPTIONAL)
    ├─ Generate AI summaries for nodes < threshold
    └─ Add prefix_summary to internal nodes
    ├─ Generate doc_description
    ↓
Enhanced tree with summaries
```

### Layer 5: Retrieval (Tree + Query → Content)
```
Query/Line Range
    ↓ PageSelector
    └─ Parse "1-5,10,20-25" → [1,2,3,4,5,10,20,21,22,23,24,25]
    ↓ MarkdownContentRetriever
    ├─ Walk tree to find matching nodes
    └─ Collect text for line numbers
    ↓
Retrieved content
```

---

## Class Hierarchy

### Indexing Pipeline (doc_index.py)

```
MarkdownHeaderParser
  └─ extract_nodes(markdown_content) → node_list

MarkdownContentExtractor
  └─ extract_text_content(node_list, lines) → nodes_with_text

MarkdownTreeThinner (optional)
  ├─ add_token_counts(nodes) → nodes_with_tokens
  └─ thin_for_index(nodes, threshold) → merged_nodes

MarkdownTreeBuilder
  ├─ build_tree(flat_nodes) → tree
  └─ clean_tree_for_output(tree) → clean_tree

MarkdownSummaryService (optional)
  └─ generate_summaries_for_structure(tree) → tree_with_summaries

MarkdownDocumentIndexer (ORCHESTRATOR)
  └─ index(md_path, config...) → result_dict
```

### Retrieval Pipeline (retrieve.py)

```
PageSelector
  └─ parse(range_str) → line_numbers

MarkdownContentRetriever
  └─ get_content(document, lines) → content_list

DocumentRepository (ORCHESTRATOR)
  ├─ get_document(doc_id) → metadata
  ├─ get_document_structure(doc_id) → tree
  └─ get_page_content(doc_id, range) → json
```

### Agent Tools (adk_agent.py)

```
@tool ensure_pageindex_json(markdown_path) → doc_id

@tool get_pageindex_structure(doc_id) → tree_structure

@tool search_pageindex(query, doc_id) → results

@tool retrieve_markdown_content(doc_id, lines) → content

answer_with_pageindex_agent(markdown_path, query)
  → LLM orchestrates all tools → answer
```

---

## Key Algorithms

### Stack-Based Tree Building

```python
# Input: Flat list with levels
[
  {title: "H1", level: 1},
  {title: "H2", level: 2},
  {title: "H2b", level: 2},
  {title: "H3", level: 3},
  {title: "H1b", level: 1},
]

# Algorithm: Use stack to track parent path
FOR EACH node IN flat_list:
    WHILE stack.top.level >= current.level:
        stack.pop()  # Go up the tree
    
    IF stack.empty:
        Add to root_nodes
    ELSE:
        Add as child of stack.top
    
    stack.push(node)

# Output: Hierarchical tree
H1
├─ H2
├─ H2b
│  └─ H3
H1b
```

### Token-Aware Thinning

```python
# Input: Flat nodes with token counts
nodes = [
  {title: "Section", tokens: 500},      # Small
  {title: "Section.1", tokens: 200},    # Very small
  {title: "Section.2", tokens: 100},    # Very small
  {title: "Big", tokens: 5000},         # Large
]

# With min_token_threshold = 3000:
# Merge Section + its children into parent
# Keep Big (> 3000)

# Output: Optimized nodes
[
  {title: "Section", tokens: 800},  # Merged: 500+200+100
  {title: "Big", tokens: 5000},
]
```

---

## Configuration

### config.yaml

```yaml
model: "gemini-2.5-flash"                # For summaries/descriptions
retrieve_model: "gemini-2.5-flash"       # For retrieval
gemini_api_key: "optional-api-key"       # Direct API key
google_cloud_project: "optional-project" # For Vertex AI
google_cloud_location: "optional-zone"   # For Vertex AI

if_thinning: false                        # Merge small sections?
min_token_threshold: 5000                 # Min tokens to keep
summary_token_threshold: 200              # When to use summary vs raw

if_add_node_id: "yes"                     # Add node IDs?
if_add_node_summary: "yes"                # Add summaries?
if_add_doc_description: "yes"             # Add doc description?
if_add_node_text: "yes"                   # Keep text in JSON?
```

---

## Data Structures

### Node Structure (with all options)

```json
{
  "title": "Section Title",
  "node_id": "0001",
  "line_num": 5,
  "text": "# Section Title\nThis is the content...",
  "summary": "One-sentence summary of this node",
  "prefix_summary": "Summary of header area (for internal nodes)",
  "nodes": [
    // Child nodes recursively...
  ]
}
```

### Document Result (from index())

```json
{
  "doc_name": "filename",
  "line_count": 500,
  "doc_description": "One-sentence doc summary (optional)",
  "structure": [
    // Root nodes...
  ]
}
```

### Retrieved Content

```json
[
  {
    "page": 10,
    "title": "Section Title",
    "node_id": "0001",
    "content": "# Section Title\nContent here..."
  },
  {
    "page": 25,
    "title": "Subsection",
    "node_id": "0002",
    "content": "## Subsection\nMore content..."
  }
]
```

---

## Execution Flow Examples

### ADK Agent Flow

```
User: "What are the main topics?"
    ↓
Agent invokes: ensure_pageindex_json(markdown_path)
    └─ Creates index if needed
    ↓
Agent invokes: search_pageindex("What are the main topics?")
    └─ Semantic search via Gemini
    ↓
Agent invokes: retrieve_markdown_content(doc_id, lines)
    └─ Gets relevant sections
    ↓
Agent invokes: Gemini synthesis
    └─ Combines results into answer
    ↓
Answer: "The document covers: ..."
```

### Direct Orchestration Flow

```
indexer = MarkdownDocumentIndexer()
    ↓
result = await indexer.index("file.md")
    │
    ├─ MarkdownHeaderParser.extract_nodes()
    ├─ MarkdownContentExtractor.extract_text_content()
    ├─ MarkdownTreeThinner.thin_for_index() [if enabled]
    ├─ MarkdownTreeBuilder.build_tree()
    └─ MarkdownSummaryService.generate_summaries() [if enabled]
    ↓
repo = DocumentRepository({result})
    ↓
content = repo.get_page_content("doc_id", "10-20")
    │
    ├─ PageSelector.parse("10-20") → [10,11,...,20]
    └─ MarkdownContentRetriever.get_content() → results
    ↓
Your custom logic with content
```

---

## Decision Tree: Which Approach?

```
Need AI to answer questions?
├─ YES → Use ADK Agent (simplest)
│   └─ await answer_with_pageindex_agent(path, query)
└─ NO → Continue...

Need simple indexing only?
├─ YES → Use Direct Orchestration
│   └─ await MarkdownDocumentIndexer().index(path)
└─ NO → Continue...

Need one-off CLI?
├─ YES → Use app.py
│   └─ python app.py --md_path "file.md"
└─ NO → Continue...

Need persistent storage/service?
└─ Use PageIndexClient (future/planned)
```

---

## Example Usage Cheat Sheet

### Quick Q&A (ADK Agent)
```python
answer = await answer_with_pageindex_agent(
    markdown_path="doc.md",
    query="What is this about?"
)
```

### Index and Store (Direct)
```python
indexer = MarkdownDocumentIndexer()
result = await indexer.index("doc.md")
repo = DocumentRepository({result})
content = repo.get_page_content("doc_1", "1-50")
```

### From CLI
```bash
python app.py --md_path "doc.md"
```

---

## Performance Notes

- **Parsing**: ~1-2 seconds for typical documents
- **Summaries**: ~5-10 seconds per 1000 tokens (depends on Gemini)
- **Retrieval**: <100ms for line range queries
- **No vector DB**: Saves computation vs embedding-based methods

---

## Next Steps

1. **Read** the full `Readme.md` for detailed docs
2. **Run** `examples/agent_qa_example.py` for AI Q&A
3. **Run** `examples/direct_orchestration_example.py` for programmatic control
4. **Integrate** into your application using either approach
5. **Customize** based on your specific needs


