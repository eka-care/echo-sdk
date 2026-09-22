"""GPT-6 tool calling goes through the OpenAI Responses API."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from echo.llm.config import LLMConfig
from echo.llm.openai import OpenAILLM
from echo.llm.schemas import StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
    ToolCall,
)
from echo.tools.core import BaseTool


class EchoTool(BaseTool):
    name = "echo_tool"
    description = "Echo the input back."
    input_schema = {"type": "object", "properties": {"q": {"type": "string"}}}

    async def run(self, **kwargs):
        return f"echo:{kwargs.get('q')}"


class Item(SimpleNamespace):
    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


def _usage():
    return SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        input_tokens_details=SimpleNamespace(cached_tokens=4),
    )


def _reasoning():
    return Item(type="reasoning", id="rs_1", encrypted_content="enc", summary=[])


def _call(args='{"q": "hi"}'):
    return Item(type="function_call", call_id="call_1", name="echo_tool", arguments=args)


def _text(text):
    return Item(
        type="message",
        role="assistant",
        content=[SimpleNamespace(type="output_text", text=text)],
    )


def _response(*output):
    return SimpleNamespace(output=list(output), usage=_usage(), incomplete_details=None)


def _context():
    ctx = ConversationContext()
    ctx.add_message(Message(role=MessageRole.USER, content=[TextMessage(text="hello")]))
    return ctx


def _llm(model="gpt-6-luna"):
    llm = OpenAILLM(LLMConfig(provider="openai", model=model))
    llm._client = MagicMock()
    return llm


def _snapshot(create):
    """MagicMock keeps references to mutated kwargs; copy input at call time."""
    calls = []

    def side_effect(**kwargs):
        calls.append({**kwargs, "input": list(kwargs["input"])})
        return create(len(calls))

    return calls, side_effect


def test_request_shape():
    llm = _llm()
    kwargs = llm._build_responses_kwargs(
        [], [EchoTool().to_openai_schema()], "sys", "suffix", {}
    )
    assert kwargs["instructions"] == "sys\n\nsuffix"
    assert kwargs["tools"] == [
        {
            "type": "function",
            "name": "echo_tool",
            "description": "Echo the input back.",
            "parameters": EchoTool.input_schema,
            "strict": False,
        }
    ]
    assert kwargs["store"] is False
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    assert "temperature" not in kwargs
    assert "prompt_cache_key" in kwargs


def test_history_translation():
    items = OpenAILLM._to_responses_input(
        [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "https://x/y.png"}},
            ]},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "c1", "type": "function",
                "function": {"name": "echo_tool", "arguments": "{}"},
            }]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
    )
    assert items == [
        {"role": "user", "content": [
            {"type": "input_text", "text": "look"},
            {"type": "input_image", "image_url": "https://x/y.png"},
        ]},
        {"type": "function_call", "call_id": "c1", "name": "echo_tool", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
        {"role": "assistant", "content": "done"},
    ]


async def test_chat_completions_still_used_without_tools():
    llm = _llm()
    llm._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None))],
        usage=None,
    )
    response, _ = await llm.invoke(_context())
    assert response.text == "hi"
    llm._client.responses.create.assert_not_called()


async def test_invoke_runs_tool_loop_and_replays_reasoning():
    llm = _llm()
    calls, side_effect = _snapshot(
        lambda n: _response(_reasoning(), _call()) if n == 1 else _response(_text("final"))
    )
    llm._client.responses.create.side_effect = side_effect

    response, ctx = await llm.invoke(_context(), tools=[EchoTool()], system_prompt="sys")

    assert response.text == "final"
    llm._client.chat.completions.create.assert_not_called()
    second_input = calls[1]["input"]
    assert second_input[1]["type"] == "reasoning"
    assert second_input[1]["encrypted_content"] == "enc"
    assert second_input[2]["type"] == "function_call"
    assert second_input[3] == {
        "type": "function_call_output", "call_id": "call_1", "output": "echo:hi",
    }
    tool_call = ctx.messages[1].content[0]
    assert isinstance(tool_call, ToolCall) and tool_call.tool_input == {"q": "hi"}
    assert ctx.messages[1].usage.cache_read_t == 4


async def test_stream_emits_tool_events_and_text():
    llm = _llm()

    def first_stream():
        yield SimpleNamespace(
            type="response.output_item.added", output_index=1,
            item=SimpleNamespace(type="function_call", call_id="call_1", name="echo_tool"),
        )
        yield SimpleNamespace(
            type="response.function_call_arguments.delta", output_index=1, delta='{"q": "hi"}'
        )
        yield SimpleNamespace(
            type="response.completed", response=_response(_reasoning(), _call())
        )

    def second_stream():
        yield SimpleNamespace(type="response.output_text.delta", delta="fin")
        yield SimpleNamespace(type="response.output_text.delta", delta="al")
        yield SimpleNamespace(type="response.completed", response=_response(_text("final")))

    calls, side_effect = _snapshot(lambda n: first_stream() if n == 1 else second_stream())
    llm._client.responses.create.side_effect = side_effect

    events = [e async for e in llm.invoke_stream(_context(), tools=[EchoTool()])]
    types = [e.type for e in events]

    assert types == [
        StreamEventType.TOOL_CALL_START,
        StreamEventType.TOOL_CALL_ARGS,
        StreamEventType.TOOL_CALL_END,
        StreamEventType.TEXT,
        StreamEventType.TEXT,
        StreamEventType.DONE,
    ]
    assert all(c["stream"] is True for c in calls)
    assert calls[1]["input"][-1]["type"] == "function_call_output"
    done = events[-1]
    assert [m.role for m in done.context.messages] == [
        MessageRole.USER, MessageRole.ASSISTANT, MessageRole.TOOL, MessageRole.ASSISTANT,
    ]


async def test_stream_failure_becomes_error_event():
    llm = _llm()
    llm._client.responses.create.return_value = iter([
        SimpleNamespace(
            type="response.failed",
            response=SimpleNamespace(error=SimpleNamespace(message="boom")),
        )
    ])

    events = [e async for e in llm.invoke_stream(_context(), tools=[EchoTool()])]
    assert [e.type for e in events] == [StreamEventType.ERROR]
    assert events[0].error == "boom"
