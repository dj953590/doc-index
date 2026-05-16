# Quick Reference Guide

## Files Created/Updated

### 📄 Documentation
- **`Readme.md`** - Complete project documentation (1000+ lines)
  - Overview and key benefits
  - Three-layer architecture diagrams
  - Installation and configuration
  - Two main usage approaches with full examples
  - API reference
  - Troubleshooting guide

- **`ARCHITECTURE.md`** - Technical architecture summary
  - Class hierarchies
  - Algorithm explanations
  - Data structures
  - Execution flow diagrams

### 💻 Example Code

#### 1. **`examples/agent_qa_example.py`** - ADK Agent Q&A
```bash
python examples/agent_qa_example.py
```
**What it does:**
- Loads configuration (including API key)
- Indexes a Markdown document
- Asks multiple questions using AI agent
- Demonstrates automatic orchestration

**Output:**
```
Q: What is this document about?
A: The document covers...

Q: What are the main sections?
A: The main sections are...
```

**Interactive Mode Available:**
```python
# Uncomment in the file:
asyncio.run(interactive_mode())
```

#### 2. **`examples/direct_orchestration_example.py`** - Programmatic Control
```bash
python examples/direct_orchestration_example.py
```
**What it does:**
- Step-by-step workflow example
- Shows all 9 steps of the pipeline
- Demonstrates batch indexing
- Shows retrieval patterns
- Exports results to JSON

**Output:**
```
[STEP 1] Indexing Document
✓ Indexed: four-lectures (12 nodes)

[STEP 2] Tree Structure
• Lecture 1: Introduction
  • Background
  • Core Concepts

[STEP 3] Creating Repository
✓ Repository created

[STEP 4] Retrieve by Line Range
✓ Retrieved 5 sections from lines 1-50:
  1. Line 1: # Lecture 1...
  2. Line 8: ## Background...
```

---

## Architecture at a Glance

```
MARKDOWN FILE
    ↓
[USER CHOOSES APPROACH]
    │
    ├─ APPROACH 1: ADK Agent
    │   └─ answer_with_pageindex_agent(path, query)
    │   └─ AI orchestrates everything automatically
    │   └─ Good for: Q&A, simple use cases
    │
    └─ APPROACH 2: Direct Orchestration
        ├─ MarkdownDocumentIndexer().index(path)
        ├─ DocumentRepository(documents)
        ├─ PageSelector + MarkdownContentRetriever
        └─ Good for: Custom logic, APIs, services
            ↓
            RESULT
```

---

## Three Indexing Stages

### 1️⃣ Parsing
```
MarkdownHeaderParser     → Extract headings
MarkdownContentExtractor → Attach text + levels
```

### 2️⃣ Optimization (Optional)
```
MarkdownTreeThinner → Merge nodes < threshold
```

### 3️⃣ Building
```
MarkdownTreeBuilder      → Create hierarchy
MarkdownSummaryService   → Add summaries (optional)
```

---

## Running Examples

### ✅ Prerequisites
```bash
# Set API key in config.yaml (optional)
nano docindex/config.yaml
```

### ✅ Run ADK Agent Example
```bash
cd C:\Users\dj153\git\doc-index
python examples/agent_qa_example.py
```

### ✅ Run Direct Orchestration Example
```bash
python examples/direct_orchestration_example.py
```

### ✅ Interactive Q&A Mode
Edit `examples/agent_qa_example.py`:
```python
# Change this:
asyncio.run(main())

# To this:
asyncio.run(interactive_mode())
```

Then run:
```bash
python examples/agent_qa_example.py
```

### ✅ CLI Quick Index
```bash
python app.py --md_path "examples/documents/PRML.pdf"
```

---

## Common Tasks

### Task 1: Index a Document with Summaries
```python
import asyncio
from docindex.doc_index import md_to_tree

async def main():
    result = await md_to_tree(
        md_path="my_document.md",
        if_add_node_summary="yes",
        model="gemini-2.5-flash"
    )
    print(result)

asyncio.run(main())
```

### Task 2: Ask Questions Using Agent
```python
import asyncio
from docindex.adk_agent import answer_with_pageindex_agent

async def main():
    answer = await answer_with_pageindex_agent(
        markdown_path="my_document.md",
        query="What are the main points?",
        model="gemini-2.5-flash"
    )
    print(answer)

asyncio.run(main())
```

### Task 3: Retrieve Specific Sections
```python
import asyncio
from docindex.doc_index import MarkdownDocumentIndexer
from docindex.retrieve import DocumentRepository

async def main():
    indexer = MarkdownDocumentIndexer()
    result = await indexer.index("my_document.md")
    
    repo = DocumentRepository({"doc_1": result})
    content = repo.get_page_content("doc_1", "10-20")
    print(content)

asyncio.run(main())
```

### Task 4: Batch Index Multiple Files
```python
import asyncio
from pathlib import Path
from docindex.doc_index import MarkdownDocumentIndexer

async def main():
    indexer = MarkdownDocumentIndexer()
    files = list(Path("documents").glob("*.md"))
    
    tasks = [indexer.index(str(f)) for f in files]
    results = await asyncio.gather(*tasks)
    
    for r in results:
        print(f"{r['doc_name']}: {len(r['structure'])} sections")

asyncio.run(main())
```

---

## Configuration Options

### In config.yaml:

```yaml
# Model to use
model: "gemini-2.5-flash"

# API authentication
gemini_api_key: "your-key-here"           # OR
google_cloud_project: "your-project"      # For Vertex AI
google_cloud_location: "us-central1"

# Indexing options
if_thinning: false                         # Merge small sections?
min_token_threshold: 5000                  # Min tokens to keep

# Output options
if_add_node_summary: "yes"                 # Add summaries?
if_add_doc_description: "yes"              # Add doc summary?
if_add_node_text: "yes"                    # Include text in JSON?
```

---

## Troubleshooting

### ❌ "File not found"
```python
# Check file exists:
from pathlib import Path
Path("examples/documents/PRML.pdf").exists()  # Should be True
```

### ❌ "google-adk is required"
```bash
pip install google-adk
```

### ❌ "google-genai is required"
```bash
pip install google-genai
```

### ❌ "Authentication failed"
**Solution 1: Use API Key**
```python
from docindex.utils import set_gemini_api_key
set_gemini_api_key("your-api-key")
```

**Solution 2: Use Google Cloud SDK**
```bash
gcloud auth application-default login
```

**Solution 3: Use Environment Variable**
```bash
export GOOGLE_API_KEY="your-api-key"
```

### ❌ "Slow execution with summaries"
```python
# Increase token threshold to skip small sections:
result = await indexer.index(
    "doc.md",
    if_add_node_summary="yes",
    summary_token_threshold=500  # Skip summaries for < 500 tokens
)
```

---

## Key Classes You'll Use

### For Indexing:
```python
from docindex.doc_index import MarkdownDocumentIndexer

indexer = MarkdownDocumentIndexer(model="gemini-2.5-flash")
result = await indexer.index("doc.md")
```

### For Retrieval:
```python
from docindex.retrieve import DocumentRepository

repo = DocumentRepository(documents)
content = repo.get_page_content("doc_id", "10-20")
```

### For Agents:
```python
from docindex.adk_agent import answer_with_pageindex_agent

answer = await answer_with_pageindex_agent("doc.md", "question?")
```

### For Utils:
```python
from docindex.utils import (
    count_tokens,
    llm_completion,
    format_structure,
    ConfigLoader,
    print_tree,
)
```

---

## Files Overview

### Core Files
- **`app.py`** - CLI interface
- **`docindex/doc_index.py`** - Indexing pipeline (740+ lines)
- **`docindex/retrieve.py`** - Retrieval pipeline
- **`docindex/adk_agent.py`** - AI agent orchestration
- **`docindex/utils.py`** - Shared utilities (890+ lines)
- **`docindex/config.yaml`** - Configuration template

### Example Files
- **`examples/agent_qa_example.py`** - NEW: ADK Agent demo
- **`examples/direct_orchestration_example.py`** - NEW: Programmatic demo
- **`examples/gemini_api_example.py`** - Basic Gemini API usage
- **`examples/agentic_vectorless_rag_demo.py`** - Full RAG demo

### Documentation
- **`Readme.md`** - UPDATED: Complete documentation
- **`ARCHITECTURE.md`** - NEW: Technical architecture
- **`QUICKREF.md`** - This file

---

## Next Steps

1. **Read the Readme:**
   ```bash
   cat README.md
   ```

2. **Run the examples:**
   ```bash
   python examples/agent_qa_example.py
   python examples/direct_orchestration_example.py
   ```

3. **Integrate into your project:**
   - Choose your approach (ADK Agent or Direct)
   - Copy example code as starter template
   - Adapt for your specific use case

4. **Customize:**
   - Modify `config.yaml` for your preferences
   - Adjust token thresholds for optimal results
   - Add custom retrieval logic as needed

---

## Performance Tips

- ⚡ **Skip summaries** if speed is critical: `if_add_node_summary="no"`
- ⚡ **Increase token threshold** to skip small nodes: `summary_token_threshold=500`
- ⚡ **Batch index** multiple documents: `asyncio.gather(*tasks)`
- ⚡ **Cache results** in JSON to avoid re-indexing
- ⚡ **Disable thinning** if structure is important: `if_thinning=False`

---

## Architecture Summary

```
INPUT: Markdown File
    ↓
PARSE: Headers + Lines (MarkdownHeaderParser)
    ↓
EXTRACT: Text + Levels (MarkdownContentExtractor)
    ↓
OPTIMIZE: Merge Small Nodes (MarkdownTreeThinner) [optional]
    ↓
BUILD: Flat → Tree (MarkdownTreeBuilder)
    ↓
ENHANCE: Add Summaries (MarkdownSummaryService) [optional]
    ↓
OUTPUT: Hierarchical JSON Tree
    ↓
RETRIEVE: Query → Content (DocumentRepository)
    ↓
OR: Agent Orchestrates Everything (answer_with_pageindex_agent)
```

---

**You're ready to go! Start with `examples/agent_qa_example.py` 🚀**

