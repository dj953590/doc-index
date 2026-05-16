# doc-index: Advanced RAG without Vector Databases

A production-ready Python framework for parsing Markdown documents into hierarchical PageIndex structures, enabling intelligent document retrieval and Q&A without vector embeddings.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [Approach 1: ADK Agent (AI-Powered Q&A)](#approach-1-adk-agent-ai-powered-qa)
  - [Approach 2: Direct Orchestration (Programmatic)](#approach-2-direct-orchestration-programmatic)
  - [CLI Interface](#cli-interface)
- [Core Components](#core-components)
- [Configuration](#configuration)
- [API Reference](#api-reference)

---

## Overview

**doc-index** transforms Markdown documents into intelligent, queryable tree structures without relying on vector databases. Instead, it uses:

- **Hierarchical parsing** - Respects Markdown heading levels (H1-H6)
- **Token-aware thinning** - Merges small sections to optimize retrieval
- **LLM-powered summaries** - Optional AI-generated node summaries
- **Document-aware retrieval** - Gemini-based semantic search within your structure
- **Google ADK integration** - AI agent for natural Q&A

### Key Benefits

✅ **No vector DB needed** - Pure JSON structure + line-based retrieval  
✅ **Deterministic** - Reproducible parsing and retrieval  
✅ **Flexible** - Control every step of the pipeline  
✅ **Fast** - Optional summaries; caching supported  
✅ **AI-Native** - Google Gemini integration built-in  
✅ **Zero-Config** - Sensible defaults included  

---

## Architecture

### Three-Layer Design

```
┌─────────────────────────────────────────────────────────────┐
│  USER INTERFACE LAYER                                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐                │
│  │  CLI (app.py)    │  │  ADK Agent       │                │
│  │                  │  │  (adk_agent.py)  │                │
│  └────────┬─────────┘  └────────┬─────────┘                │
├─────────────────────────────────────────────────────────────┤
│  ORCHESTRATION LAYER                                        │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐ │
│  │  MarkdownDocumentIndexer                              │ │
│  │  (Coordinates all indexing steps)                     │ │
│  └───────────────────┬────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  DocumentRepository                                   │ │
│  │  (Coordinates all retrieval steps)                    │ │
│  └───────────────────┬────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  PIPELINE LAYERS                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  INDEXING PIPELINE:                                         │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │ HeaderParser     │→ │ ContentExtractor │              │
│  └──────────────────┘  └─────────┬────────┘              │
│                                  ↓                         │
│                          ┌──────────────────┐              │
│                          │ TreeThinner      │ (optional)   │
│                          │ (merge small)    │              │
│                          └─────────┬────────┘              │
│                                  ↓                         │
│                          ┌──────────────────┐              │
│                          │ TreeBuilder      │              │
│                          │ (create hierarchy)              │
│                          └─────────┬────────┘              │
│                                  ↓                         │
│                          ┌──────────────────┐              │
│                          │ SummaryService   │ (optional)   │
│                          │ (LLM summaries)  │              │
│                          └──────────────────┘              │
│                                                             │
│  RETRIEVAL PIPELINE:                                        │
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │ PageSelector     │→ │ ContentRetriever │              │
│  │ (parse ranges)   │  │ (walk tree)      │              │
│  └──────────────────┘  └──────────────────┘              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│  STORAGE LAYER                                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐              │
│  │  Markdown Files  │  │  JSON Structures │              │
│  │  (.md)           │  │  (.json)         │              │
│  └──────────────────┘  └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow Example

```
Input Markdown
     ↓
┌─────────────────────────────────────────┐
│ # Introduction                          │
│ This is an intro.                       │
│                                         │
│ ## Section 1                            │
│ Content for section 1.                  │
│                                         │
│ ### Subsection 1.1                      │
│ Deep content here.                      │
│                                         │
│ ## Section 2                            │
│ More content.                           │
└─────────────────────────────────────────┘
     ↓
[PARSE HEADERS]
     ↓
Flat node list:
{title: "Introduction", level: 1, line_num: 1, ...}
{title: "Section 1", level: 2, line_num: 4, ...}
{title: "Subsection 1.1", level: 3, line_num: 7, ...}
{title: "Section 2", level: 2, line_num: 11, ...}
     ↓
[EXTRACT TEXT CONTENT]
     ↓
{title: "Introduction", text: "# Introduction\nThis is an intro.", ...}
{title: "Section 1", text: "## Section 1\nContent for section 1.", ...}
... (with text boundaries)
     ↓
[OPTIONAL: THIN SMALL NODES]
     ↓
[BUILD TREE]
     ↓
Hierarchical structure:
{
  title: "Introduction",
  node_id: "0001",
  nodes: []
}
{
  title: "Section 1",
  node_id: "0002",
  nodes: [
    {title: "Subsection 1.1", node_id: "0003", nodes: []}
  ]
}
{
  title: "Section 2",
  node_id: "0004",
  nodes: []
}
     ↓
[OPTIONAL: ADD SUMMARIES]
     ↓
Final Output JSON
```

### Class Hierarchy

#### Indexing Pipeline (doc_index.py)

```python
MarkdownHeaderParser
  └─ extract_nodes(markdown_content)
     → List of headings with line numbers
     → Skips headings in code blocks

MarkdownContentExtractor
  └─ extract_text_content(node_list, markdown_lines)
     → Adds heading level and section text
     → Text bounded by header lines

MarkdownTreeThinner
  └─ add_token_counts(node_list)
     → Count tokens in each subtree
  └─ thin_for_index(node_list, min_token_threshold)
     → Merge nodes below threshold
     → Reduces granularity

MarkdownTreeBuilder
  └─ build_tree(node_list)
     → Convert flat → hierarchical
     → Stack-based algorithm
  └─ clean_tree_for_output(tree_nodes)
     → Remove empty child lists

MarkdownSummaryService
  └─ generate_summaries_for_structure(structure)
     → Add "summary" to leaf nodes
     → Add "prefix_summary" to internal nodes
     → Respects token threshold

MarkdownDocumentIndexer (ORCHESTRATOR)
  └─ index(md_path, ...)
     → Coordinates all above steps
     → Returns {doc_name, line_count, structure}
```

#### Retrieval Pipeline (retrieve.py)

```python
PageSelector
  └─ parse(range_str)
     → Convert "1-5,10,15-20" → [1,2,3,4,5,10,15,16,17,18,19,20]

MarkdownContentRetriever
  └─ get_content(document, line_numbers)
     → Walk tree and collect nodes in range
     → Return matching content

DocumentRepository (ORCHESTRATOR)
  └─ get_document(doc_id)
     → Metadata + structure
  └─ get_document_structure(doc_id)
     → Tree without text
  └─ get_page_content(doc_id, page_range)
     → Retrieve text for line range
```

#### AI Agent (adk_agent.py)

```python
@tool
ensure_pageindex_json(markdown_path, ...)
  → Create or load index

@tool
get_pageindex_structure(doc_id, ...)
  → Browse tree structure

@tool
search_pageindex(query, ...)
  → Semantic search via Gemini

@tool
retrieve_markdown_content(doc_id, line_range)
  → Get actual text

answer_with_pageindex_agent(markdown_path, query)
  → LLM coordinates all tools
  → Returns answer
```

---

## Installation

### Prerequisites

- Python 3.8+
- Google Cloud credentials or Gemini API key

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/doc-index.git
cd doc-index

# Install dependencies
pip install -r requirements.txt

# For AI features (optional)
pip install google-adk        # For AI agent
pip install google-genai       # For Gemini API key auth
```

### Configuration

Create or edit `docindex/config.yaml`:

```yaml
# Gemini model for summaries and descriptions
model: "gemini-2.5-flash"
retrieve_model: "gemini-2.5-flash"

# Google Cloud (for Vertex AI - optional)
google_cloud_project: 
google_cloud_location: 

# Gemini API key (alternative to credentials - optional)
gemini_api_key: 

# Indexing options
if_thinning: false                    # Merge small sections
min_token_threshold: 5000             # Min tokens to keep section
summary_token_threshold: 200          # Threshold for summary vs raw text

# Output options
if_add_node_id: "yes"                 # Add sequential node IDs
if_add_node_summary: "yes"            # Add AI summaries
if_add_doc_description: "yes"         # Add doc-level summary
if_add_node_text: "yes"               # Include section text in output
```

---

## Quick Start

### 1. Index a Markdown File

```python
import asyncio
from docindex.doc_index import md_to_tree

async def main():
    result = await md_to_tree(
        md_path="examples/documents/four-lectures.md",
        if_add_node_summary="yes",
    )
    
    print(result)
    # {
    #   "doc_name": "four-lectures",
    #   "line_count": 500,
    #   "structure": [...]
    # }

asyncio.run(main())
```

### 2. Ask an AI Agent a Question

```python
import asyncio
from docindex.adk_agent import answer_with_pageindex_agent

async def main():
    answer = await answer_with_pageindex_agent(
        markdown_path="examples/documents/four-lectures.md",
        query="What are the main topics?"
    )
    print(answer)

asyncio.run(main())
```

### 3. Retrieve Content Programmatically

```python
import asyncio
from docindex.doc_index import MarkdownDocumentIndexer
from docindex.retrieve import DocumentRepository

async def main():
    # Index
    indexer = MarkdownDocumentIndexer()
    result = await indexer.index("examples/documents/four-lectures.md")
    
    # Store in repository
    documents = {
        "doc_1": {
            "doc_name": result['doc_name'],
            "structure": result['structure'],
        }
    }
    
    # Retrieve lines 10-30
    repo = DocumentRepository(documents)
    content = repo.get_page_content("doc_1", "10-30")
    print(content)

asyncio.run(main())
```

---

## Usage Examples

### Approach 1: ADK Agent (AI-Powered Q&A)

The **ADK Agent** is the simplest path for natural language Q&A. The AI agent decides which tools to use and combines results intelligently.

#### What It Does

1. **Reads your Markdown** - Parses and indexes automatically
2. **Understands your query** - Uses LLM to interpret intent
3. **Searches intelligently** - Finds relevant sections
4. **Retrieves content** - Gets the actual text
5. **Generates answer** - LLM synthesizes the response

#### Example: Document Q&A

```python
"""
Example: Ask questions about a Markdown document using AI agent.
"""

import asyncio
from docindex.adk_agent import answer_with_pageindex_agent
from docindex.utils import ConfigLoader, set_gemini_api_key

async def main():
    # Load configuration (including API key if available)
    config_loader = ConfigLoader()
    config = config_loader.load()
    
    if config.gemini_api_key:
        set_gemini_api_key(config.gemini_api_key)
    
    # Path to your Markdown document
    markdown_path = "examples/documents/PRML.pdf"  # Can be .md or processed PDF
    
    # Example questions
    questions = [
        "What are the main topics covered in this document?",
        "Summarize the introduction section",
        "What are the key concepts explained?",
    ]
    
    print("="*70)
    print("DOCUMENT Q&A WITH AI AGENT")
    print("="*70)
    print(f"\nDocument: {markdown_path}")
    print(f"Model: {config.model}\n")
    
    for question in questions:
        print(f"\n{'─'*70}")
        print(f"Q: {question}")
        print(f"{'─'*70}")
        
        try:
            answer = await answer_with_pageindex_agent(
                markdown_path=markdown_path,
                query=question,
                model=config.model,
            )
            print(f"A: {answer}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    asyncio.run(main())
```

#### How to Run

```bash
# Set your API key in config.yaml first, then:
python examples/agent_qa_example.py
```

#### Output Example

```
═══════════════════════════════════════════════════════════════
DOCUMENT Q&A WITH AI AGENT
═══════════════════════════════════════════════════════════════

Document: examples/documents/PRML.pdf
Model: gemini-2.5-flash

─────────────────────────────────────────────────────────────
Q: What are the main topics covered in this document?
─────────────────────────────────────────────────────────────
A: The document covers: Machine Learning foundations including 
probabilistic modeling, Bayesian methods, graphical models, 
and practical applications. Key sections include regression, 
classification, clustering, and dimensionality reduction 
techniques.

─────────────────────────────────────────────────────────────
Q: Summarize the introduction section
─────────────────────────────────────────────────────────────
A: The introduction section establishes why machine learning is
important in modern computing, introduces key concepts like 
supervised vs unsupervised learning, and previews the content 
of the book including theory and practical applications.
```

---

### Approach 2: Direct Orchestration (Programmatic)

Use **MarkdownDocumentIndexer** and **DocumentRepository** directly for custom workflows. You have full control over every step.

#### Example: Custom Workflow

```python
"""
Example: Direct orchestration for custom retrieval workflows.
"""

import asyncio
import json
from pathlib import Path
from docindex.doc_index import MarkdownDocumentIndexer
from docindex.retrieve import (
    DocumentRepository,
    PageSelector,
    MarkdownContentRetriever,
)
from docindex.utils import ConfigLoader, set_gemini_api_key, print_tree

async def main():
    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load()
    if config.gemini_api_key:
        set_gemini_api_key(config.gemini_api_key)
    
    print("="*70)
    print("DIRECT ORCHESTRATION: Custom Workflow")
    print("="*70)
    
    markdown_path = "examples/documents/four-lectures.md"
    
    # ─────────────────────────────────────────────────────────────
    # STEP 1: Index the document
    # ─────────────────────────────────────────────────────────────
    print(f"\n[STEP 1] Indexing: {Path(markdown_path).name}")
    print("─" * 70)
    
    indexer = MarkdownDocumentIndexer(model=config.model)
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
    
    print(f"✓ Document indexed: {result['doc_name']}")
    print(f"  • Line count: {result['line_count']}")
    print(f"  • Nodes: {len(result['structure'])}")
    if result.get('doc_description'):
        print(f"  • Description: {result['doc_description']}")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 2: Browse the tree structure
    # ─────────────────────────────────────────────────────────────
    print(f"\n[STEP 2] Tree Structure")
    print("─" * 70)
    print_tree(result['structure'][:3])
    remaining = len(result['structure']) - 3
    if remaining > 0:
        print(f"  ... and {remaining} more root sections")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 3: Store in repository for retrieval
    # ─────────────────────────────────────────────────────────────
    print(f"\n[STEP 3] Creating Repository")
    print("─" * 70)
    
    documents = {
        "doc_1": {
            "doc_name": result['doc_name'],
            "line_count": result['line_count'],
            "structure": result['structure'],
        }
    }
    
    repo = DocumentRepository(documents)
    print("✓ Repository created with 1 document")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 4: Retrieve content by line ranges
    # ─────────────────────────────────────────────────────────────
    print(f"\n[STEP 4] Retrieve Content by Line Range")
    print("─" * 70)
    
    try:
        # Get lines 1-50 (typically includes first few sections)
        content_json = repo.get_page_content("doc_1", "1-50")
        content_list = json.loads(content_json)
        
        print(f"Retrieved {len(content_list)} sections from lines 1-50:")
        for item in content_list[:3]:
            snippet = item['content'][:80].replace('\n', ' ')
            print(f"  • Line {item['page']}: {snippet}...")
        
        if len(content_list) > 3:
            print(f"  ... and {len(content_list) - 3} more sections")
    
    except Exception as e:
        print(f"Error retrieving content: {e}")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 5: Custom retrieval with PageSelector
    # ─────────────────────────────────────────────────────────────
    print(f"\n[STEP 5] Custom Retrieval Logic")
    print("─" * 70)
    
    # Parse complex range: "10-15, 25, 30-35"
    selector = PageSelector()
    try:
        line_numbers = selector.parse("10-15,25,30-35")
        print(f"Parsed range '10-15,25,30-35' → {line_numbers}")
        
        retriever = MarkdownContentRetriever()
        custom_content = retriever.get_content(
            documents['doc_1'],
            line_numbers
        )
        
        print(f"Retrieved {len(custom_content)} nodes from custom range:")
        for item in custom_content[:2]:
            print(f"  • {item.get('title', 'Untitled')}: lines {item.get('page')}")
    
    except Exception as e:
        print(f"Error: {e}")
    
    # ─────────────────────────────────────────────────────────────
    # STEP 6: Export results
    # ─────────────────────────────────────────────────────────────
    print(f"\n[STEP 6] Export Results")
    print("─" * 70)
    
    output_path = Path("results") / f"{result['doc_name']}_structure.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Saved: {output_path}")
    
    print("\n" + "="*70)
    print("Workflow Complete!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())
```

#### How to Run

```bash
python examples/direct_orchestration_example.py
```

#### Output

```
══════════════════════════════════════════════════════════════
DIRECT ORCHESTRATION: Custom Workflow
══════════════════════════════════════════════════════════════

[STEP 1] Indexing: four-lectures.md
──────────────────────────────────────────────────────────────
✓ Document indexed: four-lectures
  • Line count: 850
  • Nodes: 12
  • Description: A collection of four academic lectures covering...

[STEP 2] Tree Structure
──────────────────────────────────────────────────────────────
• Lecture 1: Introduction
  • Background
  • Core Concepts
• Lecture 2: Advanced Topics
  • Part A
  • Part B
... and 9 more root sections

[STEP 3] Creating Repository
──────────────────────────────────────────────────────────────
✓ Repository created with 1 document

[STEP 4] Retrieve Content by Line Range
──────────────────────────────────────────────────────────────
Retrieved 5 sections from lines 1-50:
  • Line 1: # Lecture 1: Introduction This lecture covers...
  • Line 8: ## Background The historical context of...
  • Line 15: ## Core Concepts The fundamental ideas...
  ... and 2 more sections

[STEP 5] Custom Retrieval Logic
──────────────────────────────────────────────────────────────
Parsed range '10-15,25,30-35' → [10, 11, 12, 13, 14, 15, 25, 30, 31, 32, 33, 34, 35]
Retrieved 3 nodes from custom range:
  • Background: lines 10
  • Core Concepts: lines 25
  • Advanced Topics: lines 30

[STEP 6] Export Results
──────────────────────────────────────────────────────────────
✓ Saved: results/four-lectures_structure.json

══════════════════════════════════════════════════════════════
Workflow Complete!
══════════════════════════════════════════════════════════════
```

---

### CLI Interface

The `app.py` command-line tool provides quick access to indexing.

#### Usage

```bash
# Index a Markdown file
python app.py --md_path "path/to/document.md"

# With all options
python app.py \
  --md_path "documents/file.md" \
  --if_thinning true \
  --min_token_threshold 3000 \
  --if_add_node_summary yes \
  --summary_token_threshold 150 \
  --model "gemini-2.5-flash"

# With API key
python app.py \
  --md_path "documents/file.md" \
  --gemini_api_key "your-api-key-here"
```

#### Output

The CLI creates a JSON structure file in `results/`:

```json
{
  "doc_name": "file",
  "line_count": 500,
  "doc_description": "A comprehensive guide to...",
  "structure": [
    {
      "title": "Chapter 1",
      "node_id": "0001",
      "line_num": 1,
      "summary": "An introduction covering...",
      "nodes": [
        {
          "title": "Section 1.1",
          "node_id": "0002",
          "line_num": 5,
          "summary": "Details about..."
        }
      ]
    }
  ]
}
```

---

## Core Components

### `doc_index.py` - Document Indexing Pipeline

Transforms raw Markdown into hierarchical tree structures.

**Main Classes:**

- `MarkdownHeaderParser` - Extracts ATX headings (`#` to `######`)
- `MarkdownContentExtractor` - Attaches heading levels and text boundaries
- `MarkdownTreeThinner` - Merges small sections below token threshold
- `MarkdownTreeBuilder` - Converts flat list to hierarchical tree
- `MarkdownSummaryService` - Generates AI summaries asynchronously
- `MarkdownDocumentIndexer` - Orchestrates entire pipeline

**Key Features:**

- Skips headings inside fenced code blocks (` ``` `)
- Deterministic section boundaries (heading to heading)
- Token-aware merging to optimize document structure
- Concurrent summary generation with semaphore control
- Zero-padded node IDs for stable references

### `retrieve.py` - Content Retrieval Pipeline

Queries indexed documents and retrieves matching content.

**Main Classes:**

- `PageSelector` - Parses line number ranges ("1-5,10,15-20")
- `MarkdownContentRetriever` - Walks tree and collects nodes
- `DocumentRepository` - Manages document storage and access

**Key Features:**

- Flexible range syntax support
- Fast tree traversal
- JSON serialization for API responses
- Support for multiple documents

### `adk_agent.py` - AI-Powered Agent

Orchestrates tools using Google ADK framework for natural Q&A.

**Tools:**

- `ensure_pageindex_json()` - Create/load index
- `get_pageindex_structure()` - Browse tree structure
- `search_pageindex()` - Semantic search
- `retrieve_markdown_content()` - Get text content

**Main Function:**

- `answer_with_pageindex_agent()` - Run full agent

### `utils.py` - Shared Utilities

Common functions used across the pipeline.

**Key Functions:**

- `count_tokens()` - Gemini token counting with fallback
- `llm_completion()` - Synchronous Gemini calls
- `llm_acompletion()` - Asynchronous Gemini calls
- `generate_node_summary()` - AI-generated summaries
- `format_structure()` - Reorder node fields
- `ConfigLoader` - YAML configuration management
- `_get_genai_client()` - Cached Gemini client

---

## Configuration

### `config.yaml` Reference

```yaml
# AI Model Configuration
model: "gemini-2.5-flash"               # Model for summaries, descriptions
retrieve_model: "gemini-2.5-flash"      # Model for retrieval

# Google Cloud Authentication (Vertex AI)
google_cloud_project: "your-project-id" # For Vertex AI
google_cloud_location: "us-central1"     # For Vertex AI

# Gemini API Key (alternative authentication)
gemini_api_key: "your-api-key-here"     # Direct API key (overrides env)

# Indexing Pipeline Options
if_thinning: false                       # Merge nodes below threshold?
min_token_threshold: 5000                # Minimum tokens to keep node
summary_token_threshold: 200             # Threshold for LLM summary vs raw text

# Output Generation
if_add_node_id: "yes"                    # Add zero-padded node IDs
if_add_node_summary: "yes"               # Add AI summaries to nodes
if_add_doc_description: "yes"            # Add one-sentence doc summary
if_add_node_text: "yes"                  # Include section text in JSON
```

### Environment Variables

```bash
# Set model for token counting (if config.yaml is empty)
export PAGEINDEX_GEMINI_MODEL="gemini-2.5-flash"

# Google Cloud credentials (for Vertex AI)
export GOOGLE_CLOUD_PROJECT="your-project"
export GOOGLE_CLOUD_LOCATION="us-central1"

# Alternative: Gemini API key
export GOOGLE_API_KEY="your-api-key"
```

---

## API Reference

### MarkdownDocumentIndexer.index()

```python
async def index(
    md_path: str,
    if_thinning: bool = False,
    min_token_threshold: int = None,
    if_add_node_summary: str = "no",
    summary_token_threshold: int = None,
    if_add_doc_description: str = "no",
    if_add_node_text: str = "no",
    if_add_node_id: str = "yes",
) -> dict:
    """
    Index a Markdown file into a PageIndex tree structure.
    
    Args:
        md_path: Path to .md file
        if_thinning: Merge small sections?
        min_token_threshold: Minimum tokens to retain section
        if_add_node_summary: Generate node summaries?
        summary_token_threshold: Tokens below which use raw text
        if_add_doc_description: Generate doc-level summary?
        if_add_node_text: Include section text in output?
        if_add_node_id: Rewrite node IDs?
    
    Returns:
        {
            "doc_name": str,
            "line_count": int,
            "doc_description": str (optional),
            "structure": [nodes]  # Hierarchical tree
        }
    """
```

### DocumentRepository.get_page_content()

```python
def get_page_content(
    doc_id: str,
    page_range: str
) -> str:
    """
    Retrieve content for a document within a line range.
    
    Args:
        doc_id: Document identifier
        page_range: Line range string ("1-5,10,15-20")
    
    Returns:
        JSON string containing matching sections
    """
```

### answer_with_pageindex_agent()

```python
async def answer_with_pageindex_agent(
    markdown_path: str,
    query: str,
    model: str = None,
) -> str:
    """
    Use AI agent to answer questions about a Markdown document.
    
    Args:
        markdown_path: Path to .md file
        query: Question to answer
        model: Gemini model ID
    
    Returns:
        Answer string generated by LLM
    """
```

---

## Comparison: When to Use What

| Use Case | Approach | Entry Point |
|----------|----------|-------------|
| Simple Q&A chatbot | ADK Agent | `answer_with_pageindex_agent()` |
| Custom retrieval logic | Direct Orchestration | `MarkdownDocumentIndexer()` |
| Fast indexing only | Direct | `MarkdownDocumentIndexer.index()` |
| CLI tool | CLI | `app.py` |
| Persistent storage | Future | `PageIndexClient` (planned) |

---

## Advanced Topics

### Token Counting & Thinning

The `min_token_threshold` parameter controls document structure optimization:

```python
# Without thinning (default)
result = await indexer.index(
    md_path="doc.md",
    if_thinning=False  # Keep all sections
)

# With thinning (merge small sections)
result = await indexer.index(
    md_path="doc.md",
    if_thinning=True,
    min_token_threshold=5000  # Merge sections < 5000 tokens
)
```

### Custom Summary Thresholds

```python
# Use raw text for sections < 200 tokens
# Generate summaries for larger sections
result = await indexer.index(
    md_path="doc.md",
    if_add_node_summary="yes",
    summary_token_threshold=200  # Adjust this
)
```

### Batch Processing

```python
import asyncio
from pathlib import Path

async def batch_index():
    indexer = MarkdownDocumentIndexer()
    md_files = list(Path("documents").glob("*.md"))
    
    # Index all files concurrently
    tasks = [
        indexer.index(str(f), if_add_node_summary="yes")
        for f in md_files
    ]
    
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(batch_index())
```

---

## Troubleshooting

### Issue: "google-genai is required for Gemini"

**Solution:** Install the package:
```bash
pip install google-genai
```

### Issue: "google-adk is required for agent tools"

**Solution:** Install for AI agent support:
```bash
pip install google-adk
```

### Issue: Authentication errors

**Solution:** Set credentials via one of:
1. `config.yaml` → `gemini_api_key`
2. Environment → `GOOGLE_API_KEY`
3. Google Cloud SDK → `gcloud auth`

### Issue: Slow summaries

**Solution:** Increase `summary_token_threshold`:
```python
result = await indexer.index(
    "doc.md",
    if_add_node_summary="yes",
    summary_token_threshold=500  # Skip summaries for < 500 token nodes
)
```

---

## License

[Your License Here]

## Contributing

Contributions welcome! Please submit issues and PRs.

---

## References

- [Google Gemini API](https://ai.google.dev/)
- [Google ADK Framework](https://github.com/google/google-ai-agentic-framework-for-python)
- [Markdown Specification](https://spec.commonmark.org/)

