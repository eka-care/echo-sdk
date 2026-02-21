"""
Example: Using the Prompt Management System.

This demonstrates:
- Fetching prompt from Langfuse with variables compiled at fetch time
- Fetching dataset from Langfuse
- Running experiment using fetched dataset and custom evaluator

Prerequisites:
- Install langfuse: pip install 'echo[langfuse]'
- Set environment variables:
  - LANGFUSE_PUBLIC_KEY
  - LANGFUSE_SECRET_KEY
  - LANGFUSE_BASE_URL (optional, defaults to https://cloud.langfuse.com)
"""

import asyncio
import json

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.prompts import PromptFetchError, get_prompt_provider
from echo.evals import get_eval_provider
from echo.agents.generic_agent import GenericAgent
from echo.llm import LLMConfig
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)

def br() -> str:
    return "=" * 60 + "\n"


def prompt_name() -> str:
    return "sa-self-care-bot"


llm_config = LLMConfig(
    provider="bedrock",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.2,
    max_tokens=2000,
    max_iterations=5,
)

async def run_agent_eval():
    """Demonstrate using ready-to-use AgentConfig from a prompt."""
    print(f"{br()}Using AgentConfig from Prompt{br()}")

    provider = get_prompt_provider()

    try:
        # Fetch prompt with variables - AgentConfig is ready to use
        prompt = await provider.get_prompt(
            prompt_name(),
            prompt_variables={"specialty": "cardiology"},
        )
        print(f"Using prompt: {prompt.name} v{prompt.version}")

    except PromptFetchError:
        print("Failed to fetch prompt")
        return

    except ValueError as e:
        print(f"Configuration error: {e}")
        return

    # Access the ready-to-use agent_config directly
    agent = GenericAgent(
        agent_config=prompt.agent_config,
        llm_config=llm_config,
    )

    async def get_agent_reply(*, item, **kwargs):
        # Create conversation context
        context = ConversationContext()
        chat_history = item.input.get("chat_history", [])
        for message in chat_history:
            context.add_message(
                Message(
                    role=MessageRole.USER if message["role"] == "user" else MessageRole.ASSISTANT,
                    content=[TextMessage(text=message["message"])],
                )
            )
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text=item.input.get("question"))],
            )
        )
        result = await agent.run(context, "tdgsf")
        if result.error:
            return f"Agent Error: {result.error}"
        elif result.llm_response:
            return result.llm_response.text
        else:
            return result

    try:
        provider = get_eval_provider()

        await provider.run_experiment(
            name="Orange test run",
            dataset_name="orange-scenarios",
            description="Generating output for dataset with chat history",
            run_func=get_agent_reply, # see above for the task definition
        )

    except Exception as e:
        print(f"Exception while fetching dataset: {e}")

    print("=" * 60)
    print("Eval ended")
    print("=" * 60)


async def main():
    """Main entry point."""
    print(f"{br()}Prompt Evaluation Usage{br()}")

    await run_agent_eval()


if __name__ == "__main__":
    asyncio.run(main())
