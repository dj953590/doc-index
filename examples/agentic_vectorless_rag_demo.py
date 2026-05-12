"""
Google ADK PageIndex Demo

This example creates a Markdown PageIndex JSON file when needed, then answers a
question by letting a Google ADK agent call PageIndex retrieval tools.

Before running, configure Vertex AI credentials, for example:
  GOOGLE_GENAI_USE_VERTEXAI=true
  GOOGLE_CLOUD_PROJECT=your-project
  GOOGLE_CLOUD_LOCATION=us-central1
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from docindex import answer_with_pageindex_agent, ensure_pageindex_json


EXAMPLE_MD = Path(__file__).parent / "tutorials" / "doc-search" / "README.md"


async def main():
    """
    Demonstrate the ADK PageIndex agent against a local Markdown tutorial.

    The example first ensures the PageIndex JSON exists for the sample Markdown
    file, then asks the ADK agent a question. During the answer, the agent can
    inspect the structure, search indexed nodes, and retrieve node text.
    """
    index_result = ensure_pageindex_json(str(EXAMPLE_MD))
    print(f"Index status: {index_result}")

    answer = await answer_with_pageindex_agent(
        markdown_path=str(EXAMPLE_MD),
        query="What is this tutorial about, and what files should I read first?",
    )
    print("\nAnswer:\n")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
