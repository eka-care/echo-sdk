"""
Integration tests for LLM providers with verbose response handling.

These tests require actual LLM API credentials:
- AWS credentials for Bedrock
- OPENAI_API_KEY for OpenAI
- ANTHROPIC_API_KEY for Anthropic

Run with: uv run pytest tests/test_llm_providers_integration.py -v
Skip with: uv run pytest -k "not Integration"
"""

import pytest

from echo.llm import LLMConfig, get_llm
from echo.llm.schemas import LLMResponse, VerboseResponseItem
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)
from echo.tools.base_tool import BaseTool


class SimpleMockTool(BaseTool):
    """A simple mock tool for testing tool calls."""

    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "Get the current time. Use this when the user asks what time it is."

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "The timezone to get time for (e.g., 'UTC', 'EST')",
                    "default": "UTC",
                }
            },
            "required": [],
        }

    async def run(self, **kwargs) -> str:
        return "The current time is 10:30 AM UTC"


class TestBedrockLLMIntegration:
    """Integration tests for Bedrock LLM provider."""

    @pytest.mark.asyncio
    async def test_simple_text_response_has_verbose(self):
        """Test that a simple text response populates verbose."""
        config = LLMConfig(provider="bedrock")
        llm = get_llm(config)

        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text="Say hello in one sentence.")],
            )
        )

        response, _ = await llm.invoke(context)

        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert len(response.verbose) > 0
        assert response.verbose[0].type == "text"
        assert response.verbose[0].text == response.text

    @pytest.mark.asyncio
    async def test_tool_call_response_has_verbose(self):
        """Test that tool calls are captured in verbose output."""
        config = LLMConfig(provider="bedrock")
        llm = get_llm(config)

        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text="What time is it right now?")],
            )
        )

        tool = SimpleMockTool()
        response, _ = await llm.invoke(context, tools=[tool])

        assert isinstance(response, LLMResponse)
        assert len(response.verbose) > 0

        # Check if there's at least one text and one tool entry
        types = [item.type for item in response.verbose]
        # The response should have text and possibly tool calls
        assert "text" in types or "tool" in types

    @pytest.mark.asyncio
    async def test_verbose_captures_intermediate_text(self):
        """Test that verbose captures text even when tool is called."""
        config = LLMConfig(provider="bedrock")
        llm = get_llm(config)

        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[
                    TextMessage(
                        text="I need to know the current time. Please check for me."
                    )
                ],
            )
        )

        tool = SimpleMockTool()
        response, _ = await llm.invoke(context, tools=[tool])

        # Extract text items from verbose
        text_items = [item for item in response.verbose if item.type == "text"]
        tool_items = [item for item in response.verbose if item.type == "tool"]

        # If the LLM called a tool, there should be text captured
        if tool_items:
            # There should be at least some text (before or after tool call)
            # Note: The final response.text should also be populated
            assert response.text != "" or len(text_items) > 0


class TestOpenAILLMIntegration:
    """Integration tests for OpenAI LLM provider."""

    @pytest.mark.asyncio
    async def test_simple_text_response_has_verbose(self):
        """Test that a simple text response populates verbose."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        llm = get_llm(config)

        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text="Say hello in one sentence.")],
            )
        )

        response, _ = await llm.invoke(context)

        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert len(response.verbose) > 0
        assert response.verbose[0].type == "text"

    @pytest.mark.asyncio
    async def test_tool_call_response_has_verbose(self):
        """Test that tool calls are captured in verbose output."""
        config = LLMConfig(provider="openai", model="gpt-4o-mini")
        llm = get_llm(config)

        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text="What time is it right now?")],
            )
        )

        tool = SimpleMockTool()
        response, _ = await llm.invoke(context, tools=[tool])

        assert isinstance(response, LLMResponse)
        assert len(response.verbose) > 0


class TestAnthropicLLMIntegration:
    """Integration tests for Anthropic LLM provider."""

    @pytest.mark.asyncio
    async def test_simple_text_response_has_verbose(self):
        """Test that a simple text response populates verbose."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku-20240307")
        llm = get_llm(config)

        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text="Say hello in one sentence.")],
            )
        )

        response, _ = await llm.invoke(context)

        assert isinstance(response, LLMResponse)
        assert response.text != ""
        assert len(response.verbose) > 0
        assert response.verbose[0].type == "text"

    @pytest.mark.asyncio
    async def test_tool_call_response_has_verbose(self):
        """Test that tool calls are captured in verbose output."""
        config = LLMConfig(provider="anthropic", model="claude-3-haiku-20240307")
        llm = get_llm(config)

        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text="What time is it right now?")],
            )
        )

        tool = SimpleMockTool()
        response, _ = await llm.invoke(context, tools=[tool])

        assert isinstance(response, LLMResponse)
        assert len(response.verbose) > 0
