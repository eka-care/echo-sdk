"""
Example: Using the Prompt Management System.

This demonstrates:
- Fetching prompts from Langfuse with variables compiled at fetch time
- Using version-specific prompts
- Accessing ready-to-use AgentConfig from prompts

Prerequisites:
- Install langfuse: pip install 'echo[langfuse]'
- Set environment variables:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_BASE_URL (optional, defaults to https://cloud.langfuse.com)
"""

import asyncio

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

from echo.prompts import PromptFetchError, get_prompt_provider


def br() -> str:
    return "=" * 60 + "\n"


def prompt_name() -> str:
    return "sa-self-care-bot"


async def fetch_prompts():
    """Demonstrate fetching specific versions with variables."""
    print(f"{br()}Version-Specific Fetching{br()}")

    provider = get_prompt_provider()

    try:
        # Fetch a specific version
        prompt_v1 = await provider.get_prompt(prompt_name(), version="1")
        print(f"Version 1: {prompt_v1.version}")

        prompt_v2 = await provider.get_prompt(prompt_name(), version="2")
        print(f"Version 2: {prompt_v2.version}")

        # Fetch latest (no version specified)
        prompt_latest = await provider.get_prompt(prompt_name())
        print(f"Latest: {prompt_latest.version}")
        print()

        # Variables are compiled at fetch time - pass them directly to get_prompt()
        prompt_with_vars = await provider.get_prompt(
            prompt_name(),
            prompt_variables={"specialty": "cardiology"},
        )
        print("Compiled task description:")
        print(prompt_with_vars.agent_config.task.description)

    except PromptFetchError as e:
        print(f"Failed to fetch: {e}")
    except ValueError as e:
        print(f"Configuration error: {e}")


async def create_agent_config_from_prompt():
    """Demonstrate using ready-to-use AgentConfig from a prompt."""
    print(f"{br()}Using AgentConfig from Prompt{br()}")

    provider = get_prompt_provider()

    try:
        # Fetch prompt with variables - AgentConfig is ready to use
        prompt = await provider.get_prompt(
            prompt_name(),
            version='3',
            prompt_variables={"specialty": "cardiology"},
        )

        # Access the ready-to-use agent_config directly
        config = prompt.agent_config

        print(f"Using prompt: {prompt.name} v{prompt.version}")
        print("AgentConfig:")
        print(config.model_dump_json(indent=2))

    except PromptFetchError:
        print("Failed to fetch prompt")

    except ValueError as e:
        print(f"Configuration error: {e}")


async def main():
    """Main entry point."""
    print(f"{br()}Prompt Management Usage{br()}")

    examples = [
        ("1", "Version-specific fetching", fetch_prompts),
        ("2", "Create AgentConfig from prompt", create_agent_config_from_prompt),
        ("3", "Run all examples", None),
    ]

    print("Choose an example:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print()

    choice = input("Enter choice (1-3): ").strip()

    if choice == "3":
        # Run all examples
        for num, name, func in examples[:-1]:
            if func:
                await func()
    else:
        # Run selected example
        for num, name, func in examples:
            if num == choice and func:
                await func()
                break
        else:
            print(f"Invalid choice: {choice}")


if __name__ == "__main__":
    asyncio.run(main())
