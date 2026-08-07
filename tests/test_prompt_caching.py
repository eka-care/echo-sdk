"""Prompt-cache markers on the Anthropic provider: TTL and message-tail."""

from echo.llm.anthropic import AnthropicLLM
from echo.llm.config import LLMConfig

MODEL = "claude-sonnet-4-20250514"


def _llm(**kwargs) -> AnthropicLLM:
    return AnthropicLLM(LLMConfig(provider="anthropic", model=MODEL, **kwargs))


def _messages():
    return [
        {"role": "user", "content": [{"type": "text", "text": "transcript"}]},
        {"role": "user", "content": [{"type": "text", "text": "latest"}]},
    ]


# --- TTL on the system block ------------------------------------------------

def test_system_block_default_has_no_ttl():
    blocks = _llm()._cached_system("prompt")
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


def test_system_block_carries_configured_ttl():
    blocks = _llm(cache_ttl="1h")._cached_system("prompt")
    assert blocks[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_system_suffix_stays_unmarked():
    blocks = _llm(cache_ttl="1h")._cached_system("prompt", "volatile")
    assert "cache_control" not in blocks[1]


def test_one_hour_ttl_sends_beta_header():
    kwargs = _llm(cache_ttl="1h")._build_request_kwargs(_messages(), {})
    assert kwargs["extra_headers"] == {
        "anthropic-beta": "extended-cache-ttl-2025-04-11"
    }


def test_default_ttl_sends_no_beta_header():
    kwargs = _llm(cache_ttl="5m")._build_request_kwargs(_messages(), {})
    assert "extra_headers" not in kwargs


# --- message-tail breakpoint ------------------------------------------------

def test_message_tail_marked_with_ttl():
    messages = _messages()
    _llm(cache_messages=True, cache_ttl="5m")._mark_message_tail(messages)
    assert messages[-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "5m",
    }
    # only the tail is marked
    assert "cache_control" not in messages[0]["content"][0]


def test_mark_message_tail_handles_empty_messages():
    _llm(cache_messages=True)._mark_message_tail([])  # must not raise


def _captured_invoke_request(llm) -> dict:
    """Run invoke() against a stub client and return the request kwargs."""
    import asyncio
    from unittest.mock import MagicMock

    from echo.models.user_conversation import (
        ConversationContext,
        Message,
        MessageRole,
        TextMessage,
    )

    captured = {}

    def fake_create(**request_kwargs):
        import copy

        # snapshot: the provider mutates the messages list after the call
        captured.update(copy.deepcopy(request_kwargs))
        response = MagicMock()
        response.content = []
        response.stop_reason = "end_turn"
        response.usage = MagicMock(
            input_tokens=1,
            output_tokens=1,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        )
        return response

    llm._client = MagicMock()
    llm._client.messages.create.side_effect = fake_create

    ctx = ConversationContext()
    ctx.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text="transcript")])
    )
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        llm.invoke(context=ctx, system_prompt="prompt")
    )
    return captured


def test_invoke_marks_tail_only_when_flag_set():
    on = _captured_invoke_request(_llm(cache_messages=True))
    assert on["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }

    off = _captured_invoke_request(_llm())
    assert "cache_control" not in off["messages"][-1]["content"][-1]


def test_config_rejects_invalid_ttl():
    import pytest

    with pytest.raises(Exception):
        LLMConfig(provider="anthropic", model=MODEL, cache_ttl="2h")


def test_defaults_leave_caching_behaviour_unchanged():
    cfg = LLMConfig(provider="anthropic", model=MODEL)
    assert cfg.cache_ttl is None
    assert cfg.cache_messages is False
