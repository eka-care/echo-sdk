"""
Unit tests for Claude model capability detection and the request shapes the
Anthropic and Bedrock providers build from it.

No network or credentials: providers create their client lazily, so the request
kwargs can be asserted directly.

Run with: uv run pytest tests/test_model_capabilities.py -v
"""

import pytest

from echo.llm import AnthropicEffort, LLMConfig, ThinkingConfig, get_llm
from echo.llm.model_capabilities import claude_capabilities

SONNET_5 = "claude-sonnet-5"
SONNET_46 = "claude-sonnet-4-6"
HAIKU_3 = "claude-3-haiku-20240307"


def anthropic_llm(model: str, thinking: ThinkingConfig = None):
    return get_llm(
        LLMConfig(provider="anthropic", model=model, thinking=thinking, api_key="test")
    )


class TestModelIdParsing:
    """Both first-party and Bedrock model ID forms must resolve."""

    @pytest.mark.parametrize(
        "model",
        [
            SONNET_5,
            "anthropic.claude-sonnet-5",
            "us.anthropic.claude-sonnet-5-v1:0",
        ],
    )
    def test_sonnet_5_recognised_in_every_id_form(self, model):
        assert claude_capabilities(model).accepts_sampling_params is False

    @pytest.mark.parametrize(
        "model",
        [
            SONNET_46,
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-1-20250805",
            HAIKU_3,
            "claude-3-5-sonnet-20241022",
        ],
    )
    def test_pre_47_models_keep_sampling_params(self, model):
        assert claude_capabilities(model).accepts_sampling_params is True

    @pytest.mark.parametrize(
        "model", ["gpt-5.2", "gemini-3-pro-preview", "meta.llama3-70b-instruct-v1:0", ""]
    )
    def test_non_claude_models_fall_back_to_the_permissive_surface(self, model):
        caps = claude_capabilities(model)
        assert caps.accepts_sampling_params is True
        assert caps.accepts_budget_tokens is False
        assert caps.supports_adaptive_thinking is False
        assert caps.thinking_on_by_default is False


class TestCapabilityMatrix:
    def test_sonnet_5_is_adaptive_only(self):
        caps = claude_capabilities(SONNET_5)
        assert caps.accepts_budget_tokens is False
        assert caps.supports_adaptive_thinking is True
        assert caps.thinking_on_by_default is True
        assert caps.can_disable_thinking is True
        assert caps.supports_effort is True

    def test_sonnet_46_keeps_the_legacy_budget_form(self):
        caps = claude_capabilities(SONNET_46)
        assert caps.accepts_budget_tokens is True
        assert caps.supports_adaptive_thinking is True
        # Thinking stays off unless it is asked for.
        assert caps.thinking_on_by_default is False

    def test_opus_47_dropped_budget_tokens_but_not_default_off_thinking(self):
        caps = claude_capabilities("claude-opus-4-7")
        assert caps.accepts_budget_tokens is False
        assert caps.accepts_sampling_params is False
        assert caps.thinking_on_by_default is False

    def test_haiku_45_has_budget_thinking_but_no_effort(self):
        caps = claude_capabilities("claude-haiku-4-5")
        assert caps.accepts_budget_tokens is True
        assert caps.supports_adaptive_thinking is False
        assert caps.supports_effort is False

    def test_haiku_3_has_no_thinking_at_all(self):
        caps = claude_capabilities(HAIKU_3)
        assert caps.accepts_budget_tokens is False
        assert caps.supports_adaptive_thinking is False

    def test_fable_5_cannot_have_thinking_disabled(self):
        caps = claude_capabilities("claude-fable-5")
        assert caps.thinking_on_by_default is True
        assert caps.can_disable_thinking is False


class TestAnthropicRequestShape:
    def test_sonnet_5_omits_temperature_and_uses_low_thinking_by_default(self):
        llm = anthropic_llm(SONNET_5)
        request = llm._build_request_kwargs([], {})

        assert "temperature" not in request
        assert request["thinking"] == {"type": "adaptive"}
        assert request["extra_body"] == {"output_config": {"effort": "low"}}

    def test_sonnet_5_translates_budget_tokens_to_adaptive(self):
        llm = anthropic_llm(SONNET_5, ThinkingConfig(budget_tokens=4096))
        request = llm._build_request_kwargs([], {})

        assert request["thinking"] == {"type": "adaptive"}
        assert "extra_body" not in request

    def test_sonnet_5_sends_effort_through_extra_body(self):
        llm = anthropic_llm(SONNET_5, ThinkingConfig(effort=AnthropicEffort.XHIGH))
        request = llm._build_request_kwargs([], {})

        assert request["thinking"] == {"type": "adaptive"}
        assert request["extra_body"] == {"output_config": {"effort": "xhigh"}}

    def test_sonnet_46_request_is_unchanged(self):
        llm = anthropic_llm(SONNET_46, ThinkingConfig(budget_tokens=4096))
        request = llm._build_request_kwargs([], {})

        assert request["temperature"] == 0.2
        assert request["thinking"] == {"type": "enabled", "budget_tokens": 4096}

    def test_effort_on_sonnet_46_uses_adaptive_thinking(self):
        llm = anthropic_llm(SONNET_46, ThinkingConfig(effort=AnthropicEffort.MEDIUM))
        request = llm._build_request_kwargs([], {})

        assert request["thinking"] == {"type": "adaptive"}
        assert request["extra_body"] == {"output_config": {"effort": "medium"}}

    def test_older_models_send_no_thinking_field(self):
        llm = anthropic_llm(HAIKU_3, ThinkingConfig(budget_tokens=4096))
        request = llm._build_request_kwargs([], {})

        assert request["temperature"] == 0.2
        assert "thinking" not in request

    def test_fable_5_omits_thinking_rather_than_disabling_it(self):
        llm = anthropic_llm("claude-fable-5")
        assert "thinking" not in llm._build_request_kwargs([], {})

    def test_caller_temperature_override_still_dropped_on_sonnet_5(self):
        llm = anthropic_llm(SONNET_5)
        assert "temperature" not in llm._build_request_kwargs([], {"temperature": 0.9})

    def test_caller_temperature_override_honoured_on_older_models(self):
        llm = anthropic_llm(SONNET_46)
        request = llm._build_request_kwargs([], {"temperature": 0.9})
        assert request["temperature"] == 0.9


class TestAssistantTurnSerialization:
    """Thinking blocks must survive the round trip back onto the wire."""

    class _Block:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    def test_thinking_block_is_echoed_verbatim(self):
        llm = anthropic_llm(SONNET_5)
        block = self._Block(type="thinking", thinking="", signature="sig-abc")

        # Empty thinking text is what display="omitted" returns; the block still
        # has to go back unmodified or the next tool-use turn is rejected.
        assert llm._serialize_block(block) == {
            "type": "thinking",
            "thinking": "",
            "signature": "sig-abc",
        }

    def test_text_and_tool_use_blocks_round_trip(self):
        llm = anthropic_llm(SONNET_5)

        assert llm._serialize_block(self._Block(type="text", text="hi")) == {
            "type": "text",
            "text": "hi",
        }
        tool_block = self._Block(
            type="tool_use", id="tu_1", name="get_time", input={"tz": "UTC"}
        )
        assert llm._serialize_block(tool_block) == {
            "type": "tool_use",
            "id": "tu_1",
            "name": "get_time",
            "input": {"tz": "UTC"},
        }

    def test_unknown_block_types_are_skipped(self):
        llm = anthropic_llm(SONNET_5)
        assert llm._serialize_block(self._Block(type="server_tool_use")) is None


class TestBedrockRequestShape:
    def test_sonnet_5_drops_temperature_and_uses_low_thinking(self):
        llm = get_llm(LLMConfig(provider="bedrock", model="anthropic.claude-sonnet-5"))

        assert "temperature" not in llm._inference_config({})
        assert llm._additional_request_fields() == {
            "additionalModelRequestFields": {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "low"},
            }
        }

    def test_older_claude_models_are_untouched(self):
        llm = get_llm(
            LLMConfig(provider="bedrock", model=f"anthropic.{HAIKU_3}-v1:0")
        )

        assert llm._inference_config({})["temperature"] == 0.2
        assert llm._additional_request_fields() == {}

    def test_non_claude_bedrock_models_are_untouched(self):
        llm = get_llm(
            LLMConfig(provider="bedrock", model="meta.llama3-70b-instruct-v1:0")
        )

        assert llm._inference_config({})["temperature"] == 0.2
        assert llm._additional_request_fields() == {}


class TestThinkingConfigValidation:
    def test_effort_is_anthropic_only(self):
        with pytest.raises(ValueError, match="only supported for Anthropic"):
            LLMConfig(
                provider="bedrock",
                model="anthropic.claude-sonnet-5",
                thinking=ThinkingConfig(effort=AnthropicEffort.HIGH),
            )

    def test_effort_accepted_for_anthropic(self):
        config = LLMConfig(
            provider="anthropic",
            model=SONNET_5,
            thinking=ThinkingConfig(effort=AnthropicEffort.MEDIUM),
        )
        assert config.thinking.effort == AnthropicEffort.MEDIUM
