"""
Example script demonstrating how to use the Gemini API client.

This example shows how to use _get_genai_client() and llm_completion()
from the utils module to ask the Gemini API a simple question.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from docindex.utils import llm_completion, _get_genai_client, set_gemini_api_key, ConfigLoader


def main():
    """
    Main function demonstrating Gemini API usage.

    This example:
    1. Loads the config.yaml to get gemini_api_key
    2. Sets the API key if available
    3. Gets the Gemini client
    4. Asks a simple question using llm_completion
    5. Prints the response
    """
    print("=" * 60)
    print("Gemini API Example - Capital of France Question")
    print("=" * 60)

    # Load configuration
    print("\n1. Loading configuration from config.yaml...")
    config_loader = ConfigLoader()
    config = config_loader.load()
    print(f"   ✓ Config loaded")

    # Set Gemini API key if available
    if config.gemini_api_key:
        print(f"\n2. Setting Gemini API key from config...")
        set_gemini_api_key(config.gemini_api_key)
        print(f"   ✓ API key configured")
        client_init_step = 3
    else:
        print(f"\n2. No Gemini API key in config, using default credentials...")
        client_init_step = 3

    # Initialize the Gemini client
    print(f"\n{client_init_step}. Initializing Gemini client...")
    client = _get_genai_client()
    print(f"   ✓ Client initialized: {type(client).__name__}")

    # Ask a simple question
    question = "What is the capital of France?"
    print(f"\n{client_init_step + 1}. Asking Gemini: {question}")

    # Get the response using llm_completion
    print("   Waiting for response...")
    response = llm_completion(
        model=None,  # Uses default model from config
        prompt=question
    )

    print("\n" + "=" * 60)
    print("Gemini Response:")
    print("=" * 60)
    print(response)

    # Additional example with a more complex prompt
    print("\n\n" + "=" * 60)
    print("Additional Example - Complex Prompt")
    print("=" * 60)

    complex_prompt = """
    List the top 3 tourist attractions in Paris and briefly explain why each is significant.
    Format the response as a numbered list.
    """

    print(f"Prompt: {complex_prompt}")
    print("Waiting for response...")

    response = llm_completion(
        model=None,
        prompt=complex_prompt
    )

    print("\n" + "=" * 60)
    print("Gemini Response:")
    print("=" * 60)
    print(response)
    print("-" * 60)

    print("\n✓ Example completed successfully!")


if __name__ == "__main__":
    main()

