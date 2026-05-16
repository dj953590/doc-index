"""
Example: Document Q&A Using AI Agent

This example demonstrates the simplest path to intelligent document Q&A:
1. Load configuration (including API key)
2. Ask questions about a Markdown document
3. The AI agent orchestrates everything automatically

The agent:
- Creates or loads the PageIndex if needed
- Searches for relevant sections
- Retrieves matching content
- Synthesizes an answer using Gemini

This is the "Approach 1: ADK Agent (AI-Powered Q&A)" from the README.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from docindex.adk_agent import answer_with_pageindex_agent
from docindex.utils import ConfigLoader, set_gemini_api_key, resolve_path


async def main():
    """
    Main entry point: Ask questions about a document using AI agent.
    """

    # ─────────────────────────────────────────────────────────────
    # Load Configuration
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("DOCUMENT Q&A WITH AI AGENT")
    print("="*70)

    print("\n[Step 1] Loading configuration...")
    config_loader = ConfigLoader()
    config = config_loader.load()

    # Set API key if available in config
    if config.gemini_api_key:
        print(f"✓ Found gemini_api_key in config")
        set_gemini_api_key(config.gemini_api_key)
    else:
        print("ℹ No gemini_api_key in config. Using default credentials.")

    print(f"✓ Model: {config.model}")

    # ─────────────────────────────────────────────────────────────
    # Document and Questions
    # ─────────────────────────────────────────────────────────────

    # You can use any of the example documents:
    # - examples/documents/four-lectures.pdf
    # - examples/documents/PRML.pdf
    # - examples/documents/attention-residuals.pdf
    # - etc.

    markdown_path = str(resolve_path("examples/documents/q1-fy25-earnings.md"))
    print("markdown path: ", markdown_path)

    # Example questions to ask the agent
    questions = [
        "What is this document about? Provide a brief overview.",
        "What are the main sections or chapters covered?",
        "What key concepts or methodologies are introduced?",
        "Who is the intended audience for this document?",
    ]

    print(f"\n[Step 2] Document: {markdown_path}")
    print(f"[Step 3] Preparing questions: {len(questions)} questions\n")

    # ─────────────────────────────────────────────────────────────
    # Ask Questions and Get Answers
    # ─────────────────────────────────────────────────────────────

    print("─" * 70)
    print("AGENT SESSION")
    print("─" * 70)

    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        print(f"Q: {question}")
        print(f"{'─'*70}")

        try:
            # Call the AI agent
            # It will automatically:
            # 1. Index the document (if needed)
            # 2. Search for relevant sections
            # 3. Retrieve the content
            # 4. Generate an answer

            answer = await answer_with_pageindex_agent(
                markdown_path=markdown_path,
                query=question,
                model=config.model,
            )

            # Display the answer
            print(f"A: {answer}\n")

        except Exception as e:
            print(f"❌ Error: {e}\n")
            print("Troubleshooting tips:")
            print("  1. Check that the file exists: " + str(Path(markdown_path).absolute()))
            print("  2. Ensure google-adk is installed: pip install google-adk")
            print("  3. Verify credentials are set (API key or gcloud auth)")
            print()

    # ─────────────────────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────────────────────

    print("─" * 70)
    print("SESSION COMPLETE")
    print("─" * 70)
    print(f"\n✓ Processed {len(questions)} questions")
    print("✓ Document: " + Path(markdown_path).name)
    print(f"✓ Model: {config.model}")


# ─────────────────────────────────────────────────────────────
# Interactive Mode (Optional)
# ─────────────────────────────────────────────────────────────

async def interactive_mode():
    """
    Interactive mode: Ask custom questions about a document.
    """

    print("\n" + "="*70)
    print("INTERACTIVE DOCUMENT Q&A")
    print("="*70)

    # Load config
    config_loader = ConfigLoader()
    config = config_loader.load()

    if config.gemini_api_key:
        set_gemini_api_key(config.gemini_api_key)

    # Get document path from user
    print("\nAvailable documents:")
    docs_dir = (resolve_path("examples/documents"))

    docs = list(docs_dir.glob("*.pdf")) + list(docs_dir.glob("*.md"))

    for i, doc in enumerate(docs[:10], 1):  # Show first 10
        print(f"  {i}. {doc.name}")

    try:
        choice = input("\nSelect document (1-10) or enter path: ").strip()
        if choice.isdigit():
            markdown_path = str(docs[int(choice) - 1])
        else:
            markdown_path = choice
    except (ValueError, IndexError):
        print("Invalid selection. Using default: examples/documents/PRML.pdf")
        markdown_path = "examples/documents/PRML.pdf"

    print(f"\nDocument: {markdown_path}")
    print(f"Model: {config.model}")
    print("\nEnter questions (type 'quit' to exit):\n")

    # Ask questions in loop
    while True:
        question = input("Q: ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        if not question:
            continue

        try:
            print("\n🤔 Thinking...\n")
            answer = await answer_with_pageindex_agent(
                markdown_path=markdown_path,
                query=question,
                model=config.model,
            )
            print(f"A: {answer}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    # Run the main example
    asyncio.run(interactive_mode())

    # Uncomment to run interactive mode instead:
    # asyncio.run(interactive_mode())

