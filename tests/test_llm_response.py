"""
Tests for LLM response handling, including verbose output.

These tests verify that:
- LLMResponse and VerboseResponseItem models work correctly
- Verbose output captures text and tool calls from all iterations
"""

import pytest

from echo.llm.schemas import LLMResponse, VerboseResponseItem


class TestVerboseResponseItem:
    """Tests for VerboseResponseItem model."""

    def test_create_text_item(self):
        """Test creating a text verbose item."""
        item = VerboseResponseItem(type="text", text="Hello, world!")

        assert item.type == "text"
        assert item.text == "Hello, world!"
        assert item.tool_name is None

    def test_create_tool_item(self):
        """Test creating a tool verbose item."""
        item = VerboseResponseItem(type="tool", tool_name="search_tool")

        assert item.type == "tool"
        assert item.tool_name == "search_tool"
        assert item.text is None

    def test_default_type_is_text(self):
        """Test that default type is text."""
        item = VerboseResponseItem(text="Some text")

        assert item.type == "text"

    def test_serialization(self):
        """Test that items can be serialized to dict."""
        item = VerboseResponseItem(type="text", text="Test message")
        data = item.model_dump()

        assert data["type"] == "text"
        assert data["text"] == "Test message"
        assert data["tool_name"] is None


class TestLLMResponse:
    """Tests for LLMResponse model."""

    def test_create_empty_response(self):
        """Test creating an empty response."""
        response = LLMResponse()

        assert response.text == ""
        assert response.verbose == []
        assert response.details is None
        assert response.pending_tool_result_processing is False
        assert response.error is None
        assert response.elicitations is None

    def test_create_response_with_text(self):
        """Test creating a response with text."""
        response = LLMResponse(text="Hello, I can help you with that.")

        assert response.text == "Hello, I can help you with that."

    def test_create_response_with_verbose(self):
        """Test creating a response with verbose items."""
        verbose_items = [
            VerboseResponseItem(type="text", text="Let me search for that."),
            VerboseResponseItem(type="tool", tool_name="search_tool"),
            VerboseResponseItem(type="text", text="Here are the results."),
        ]

        response = LLMResponse(text="Here are the results.", verbose=verbose_items)

        assert len(response.verbose) == 3
        assert response.verbose[0].type == "text"
        assert response.verbose[0].text == "Let me search for that."
        assert response.verbose[1].type == "tool"
        assert response.verbose[1].tool_name == "search_tool"
        assert response.verbose[2].type == "text"

    def test_append_to_verbose(self):
        """Test appending items to verbose list."""
        response = LLMResponse()

        response.verbose.append(VerboseResponseItem(type="text", text="First message"))
        response.verbose.append(VerboseResponseItem(type="tool", tool_name="my_tool"))

        assert len(response.verbose) == 2
        assert response.verbose[0].text == "First message"
        assert response.verbose[1].tool_name == "my_tool"

    def test_response_with_error(self):
        """Test creating a response with an error."""
        response = LLMResponse(error="Something went wrong")

        assert response.error == "Something went wrong"
        assert response.text == ""

    def test_response_with_json(self):
        """Test creating a response with JSON data."""
        json_data = {"intent": "medical", "confidence": 0.95}
        response = LLMResponse(text="Classified as medical", details=json_data)

        assert response.details == json_data
        assert response.details["intent"] == "medical"

    def test_pending_tool_result_processing(self):
        """Test pending_tool_result_processing flag."""
        response = LLMResponse(pending_tool_result_processing=True)

        assert response.pending_tool_result_processing is True

    def test_serialization(self):
        """Test that response can be serialized to dict."""
        response = LLMResponse(
            text="Final response",
            verbose=[VerboseResponseItem(type="text", text="Intermediate")],
        )
        data = response.model_dump()

        assert data["text"] == "Final response"
        assert len(data["verbose"]) == 1
        assert data["verbose"][0]["text"] == "Intermediate"


class TestVerboseOutputScenarios:
    """Tests for realistic verbose output scenarios."""

    def test_simple_text_only_response(self):
        """Test a simple response with only text (no tool calls)."""
        response = LLMResponse()
        response.verbose.append(
            VerboseResponseItem(type="text", text="Hello! How can I help you today?")
        )
        response.text = "Hello! How can I help you today?"

        assert len(response.verbose) == 1
        assert response.verbose[0].type == "text"
        assert response.text == response.verbose[0].text

    def test_text_with_single_tool_call(self):
        """Test response with text followed by a tool call."""
        response = LLMResponse()

        # Simulate: LLM says something, then calls a tool
        response.verbose.append(
            VerboseResponseItem(
                type="text",
                text="Let me look up that information for you.",
            )
        )
        response.verbose.append(
            VerboseResponseItem(type="tool", tool_name="search_database")
        )

        # Final text after tool execution
        response.text = "Based on my search, here's what I found..."

        assert len(response.verbose) == 2
        assert response.verbose[0].type == "text"
        assert response.verbose[1].type == "tool"
        assert response.verbose[1].tool_name == "search_database"

    def test_multi_iteration_tool_calls(self):
        """Test response with multiple iterations of tool calls."""
        response = LLMResponse()

        # First iteration: text + tool
        response.verbose.append(
            VerboseResponseItem(type="text", text="I'll check the appointment system.")
        )
        response.verbose.append(
            VerboseResponseItem(type="tool", tool_name="check_availability")
        )

        # Second iteration: text + another tool
        response.verbose.append(
            VerboseResponseItem(type="text", text="Now let me verify your identity.")
        )
        response.verbose.append(
            VerboseResponseItem(type="tool", tool_name="verify_phone")
        )

        # Final iteration: just text
        response.verbose.append(
            VerboseResponseItem(type="text", text="Great! Your appointment is confirmed.")
        )
        response.text = "Great! Your appointment is confirmed."

        assert len(response.verbose) == 5
        # Verify order
        assert response.verbose[0].type == "text"
        assert response.verbose[1].type == "tool"
        assert response.verbose[2].type == "text"
        assert response.verbose[3].type == "tool"
        assert response.verbose[4].type == "text"

    def test_extract_all_text_from_verbose(self):
        """Test extracting all text content from verbose output."""
        response = LLMResponse()
        response.verbose = [
            VerboseResponseItem(type="text", text="First message. "),
            VerboseResponseItem(type="tool", tool_name="tool1"),
            VerboseResponseItem(type="text", text="Second message. "),
            VerboseResponseItem(type="tool", tool_name="tool2"),
            VerboseResponseItem(type="text", text="Final message."),
        ]

        # Extract all text
        all_text = "".join(
            item.text for item in response.verbose if item.type == "text" and item.text
        )

        assert all_text == "First message. Second message. Final message."

    def test_extract_all_tool_names_from_verbose(self):
        """Test extracting all tool names from verbose output."""
        response = LLMResponse()
        response.verbose = [
            VerboseResponseItem(type="text", text="Starting..."),
            VerboseResponseItem(type="tool", tool_name="search_tool"),
            VerboseResponseItem(type="text", text="Searching..."),
            VerboseResponseItem(type="tool", tool_name="booking_tool"),
            VerboseResponseItem(type="tool", tool_name="notification_tool"),
        ]

        # Extract all tool names
        tool_names = [
            item.tool_name
            for item in response.verbose
            if item.type == "tool" and item.tool_name
        ]

        assert tool_names == ["search_tool", "booking_tool", "notification_tool"]

    def test_elicitation_scenario(self):
        """Test verbose output when elicitation is triggered."""
        response = LLMResponse()

        # LLM says something and triggers elicitation tool
        response.verbose.append(
            VerboseResponseItem(
                type="text",
                text="I'll need to collect some information. What symptoms are you experiencing?",
            )
        )
        response.verbose.append(
            VerboseResponseItem(type="tool", tool_name="elicit_selection")
        )

        # Text is extracted from the response
        response.text = "I'll need to collect some information. What symptoms are you experiencing?"

        assert len(response.verbose) == 2
        assert response.verbose[1].tool_name == "elicit_selection"
        assert response.text != ""  # Text should be present even with elicitation
