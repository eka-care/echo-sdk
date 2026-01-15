"""
Example: Using LLM providers with streaming responses.

This demonstrates:
- Using invoke_stream() for real-time response streaming
- Handling different StreamEvent types (TEXT, TOOL_CALL_START/END, DONE, ERROR)
- Multi-turn conversation with user input
- All three providers: Bedrock, Anthropic, OpenAI
"""

import asyncio

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

from echo.llm import LLMConfig, get_llm
from echo.llm.schemas import StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)


def create_llm(provider: str):
    """Get LLM instance for the specified provider."""
    if provider == "bedrock":
        return get_llm(LLMConfig())
    elif provider == "anthropic":
        return get_llm(
            LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        )
    elif provider == "openai":
        return get_llm(LLMConfig(provider="openai", model="gpt-4o-mini"))
    else:
        raise ValueError(f"Unknown provider: {provider}")


async def stream_response(llm, context, tools=None, system_prompt=None):
    """
    Stream a response from the LLM and handle events.

    Returns:
        Tuple of (final_response, final_context) from the DONE event
    """
    final_response = None
    final_context = None

    try:
        async for event in llm.invoke_stream(
            context, tools=tools, system_prompt=system_prompt
        ):
            if event.type == StreamEventType.TEXT:
                print(event.text, end="", flush=True)
            elif event.type == StreamEventType.TOOL_CALL_START:
                tool_info = event.details or {}
                print(
                    f"\n[Tool: {tool_info.get('tool_name', 'unknown')}]",
                    end="",
                    flush=True,
                )
            elif event.type == StreamEventType.TOOL_CALL_END:
                tool_info = event.details or {}
                print(f" Done", end="", flush=True)
            elif event.type == StreamEventType.DONE:
                print()  # Newline after streaming
                final_response = event.llm_response
                final_context = event.context
            elif event.type == StreamEventType.ERROR:
                print(f"\n[Error: {event.error}]")
                return None, context
    except Exception as e:
        print(f"\n[Exception: {e}]")
        import traceback

        traceback.print_exc()
        return None, context

    return final_response, final_context


async def run_conversation(provider: str = "bedrock"):
    """Run a multi-turn streaming conversation."""
    print("=" * 60)
    print(f"Streaming Conversation ({provider.upper()})")
    print("=" * 60)
    print()

    llm = create_llm(provider)
    context = ConversationContext()

    system_prompt = (
        """You are a helpful assistant. Be concise and friendly in your responses."""
    )

    print("Type 'quit' to exit, or enter your message.")
    print("-" * 60)
    print()

    for turn in range(20):  # Max 20 turns
        # Get user input
        try:
            user_input = input(f"[{turn+1}] You: ").strip()
        except EOFError:
            print("\nEnd of input. Exiting.")
            break

        if user_input.lower() == "quit":
            print("Exiting conversation.")
            break

        if not user_input:
            print("Empty input, skipping...")
            continue

        # Add user message to context
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text=user_input)],
            )
        )

        # Stream response
        print()
        print("Assistant: ", end="", flush=True)

        response, context = await stream_response(
            llm, context, system_prompt=system_prompt
        )

        if response is None:
            print("No response received. Try again or type 'quit' to exit.")
            continue

        # Show verbose info if there were tool calls
        if response.verbose:
            tool_calls = [v for v in response.verbose if v.type == "tool"]
            if tool_calls:
                print(f"  [Tools used: {', '.join(t.tool_name for t in tool_calls)}]")

        print()

    print()
    print("=" * 60)
    print("Conversation ended")
    print("=" * 60)


async def simple_streaming_test(provider: str = "bedrock"):
    """Simple test to stream a text response without tools."""
    print("=" * 60)
    print(f"Simple Streaming Test ({provider.upper()}) - No Tools")
    print("=" * 60)
    print()

    llm = create_llm(provider)
    context = ConversationContext()

    # Add a user message
    context.add_message(
        Message(
            role=MessageRole.USER,
            content=[TextMessage(text="Tell me a short joke about programming.")],
        )
    )

    print("Streaming response:")
    print("-" * 40)
    print("Assistant: ", end="", flush=True)

    response, _ = await stream_response(
        llm, context, system_prompt="You are a helpful assistant. Keep responses brief."
    )

    if response:
        print("-" * 40)
        print(f"[Verbose items: {len(response.verbose)}]")

    print()


async def main():
    """Main entry point."""
    print("Choose a provider:")
    print("1. Bedrock (default)")
    print("2. Anthropic")
    print("3. OpenAI")
    print()

    provider_choice = input("Enter provider (1/2/3): ").strip()
    provider_map = {"1": "bedrock", "2": "anthropic", "3": "openai"}
    provider = provider_map.get(provider_choice, "bedrock")

    print()
    print("Choose a test:")
    print("1. Simple streaming test (no tools)")
    print("2. Multi-turn conversation")
    print()

    test_choice = input("Enter test (1 or 2): ").strip()

    if test_choice == "1":
        await simple_streaming_test(provider)
    else:
        await run_conversation(provider)


if __name__ == "__main__":
    asyncio.run(main())
