"""
OpenAI GPT-5.6 (Luna / Terra / Sol) and GPT-6 Astra support.

Unit tests (no network) assert the Chat Completions request each model gets.
Integration tests call the real API — put your key in `.env` (loaded by
conftest) or the environment:

    OPENAI_API_KEY=sk-...

Run:
    uv run pytest tests/test_openai_models.py -v                   # everything
    uv run pytest tests/test_openai_models.py -k "not Integration" # offline only
    uv run pytest tests/test_openai_models.py -k Integration -v -s # live only
"""

import os
from types import SimpleNamespace

import pytest

from echo.llm import LLMConfig, ReasoningEffort, ThinkingConfig, get_llm
from echo.llm.schemas import StreamEventType
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)
from echo.tools.core import BaseTool

LUNA = "gpt-5.6-luna"
TERRA = "gpt-5.6-terra"
SOL = "gpt-5.6-sol"
ASTRA = "gpt-6-astra"

GPT_5_6_MODELS = [LUNA, TERRA, SOL]
ALL_MODELS = GPT_5_6_MODELS + [ASTRA]

TOOL_SCHEMA = [{"type": "function", "function": {"name": "t", "parameters": {}}}]


def openai_llm(model: str, effort: ReasoningEffort = None, **kwargs):
    thinking = ThinkingConfig(reasoning_effort=effort) if effort else None
    return get_llm(
        LLMConfig(
            provider="openai", model=model, thinking=thinking, api_key="test", **kwargs
        )
    )


def request(llm, tools=None, system_prompt="prompt", **kwargs) -> dict:
    return llm._build_request_kwargs([], tools, system_prompt, kwargs)


def user_context(text: str) -> ConversationContext:
    context = ConversationContext()
    context.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text=text)])
    )
    return context


class TimeTool(BaseTool):
    @property
    def name(self) -> str:
        return "get_current_time"

    @property
    def description(self) -> str:
        return "Get the current time. Use this when the user asks what time it is."

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> str:
        return "The current time is 10:30 AM UTC"


# --- request shape -----------------------------------------------------------


class TestRequestShape:
    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_no_sampling_params_and_uses_max_completion_tokens(self, model):
        kwargs = request(openai_llm(model, max_tokens=2048), temperature=0.7)
        assert "temperature" not in kwargs
        assert "max_tokens" not in kwargs
        assert kwargs["max_completion_tokens"] == 2048

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_effort_omitted_when_unset(self, model):
        assert "reasoning_effort" not in request(openai_llm(model))

    @pytest.mark.parametrize("model", ALL_MODELS)
    @pytest.mark.parametrize(
        "effort",
        [
            ReasoningEffort.LOW,
            ReasoningEffort.MEDIUM,
            ReasoningEffort.HIGH,
            ReasoningEffort.XHIGH,
            ReasoningEffort.MAX,
        ],
    )
    def test_supported_efforts_pass_through(self, model, effort):
        kwargs = request(openai_llm(model, effort))
        assert kwargs["reasoning_effort"] == effort.value

    @pytest.mark.parametrize("model", GPT_5_6_MODELS)
    def test_gpt_5_6_accepts_none(self, model):
        kwargs = request(openai_llm(model, ReasoningEffort.NONE))
        assert kwargs["reasoning_effort"] == "none"

    @pytest.mark.parametrize("model", GPT_5_6_MODELS)
    def test_gpt_5_6_maps_minimal_to_low(self, model):
        kwargs = request(openai_llm(model, ReasoningEffort.MINIMAL))
        assert kwargs["reasoning_effort"] == "low"

    @pytest.mark.parametrize("effort", [ReasoningEffort.NONE, ReasoningEffort.MINIMAL])
    def test_astra_maps_none_and_minimal_to_low(self, effort):
        assert request(openai_llm(ASTRA, effort))["reasoning_effort"] == "low"

    @pytest.mark.parametrize("model", GPT_5_6_MODELS)
    @pytest.mark.parametrize("effort", [None, ReasoningEffort.HIGH, ReasoningEffort.MAX])
    def test_gpt_5_6_tools_force_effort_none(self, model, effort):
        # Chat Completions rejects tools with any other effort (incl. default).
        kwargs = request(openai_llm(model, effort), tools=TOOL_SCHEMA)
        assert kwargs["tools"] == TOOL_SCHEMA
        assert kwargs["reasoning_effort"] == "none"

    @pytest.mark.parametrize("model", GPT_5_6_MODELS)
    def test_gpt_5_6_without_tools_keeps_effort(self, model):
        kwargs = request(openai_llm(model, ReasoningEffort.HIGH))
        assert kwargs["reasoning_effort"] == "high"

    def test_astra_rejects_tools_on_chat_completions(self):
        with pytest.raises(ValueError, match="Responses API"):
            request(openai_llm(ASTRA), tools=TOOL_SCHEMA)

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_prompt_cache_key_is_stable_per_system_prompt(self, model):
        llm = openai_llm(model)
        a = request(llm, system_prompt="agent A")["prompt_cache_key"]
        assert a == request(llm, system_prompt="agent A")["prompt_cache_key"]
        assert a != request(llm, system_prompt="agent B")["prompt_cache_key"]

    def test_legacy_models_unchanged(self):
        kwargs = request(openai_llm("gpt-4o-mini"), temperature=0.3)
        assert kwargs["temperature"] == 0.3
        assert "max_tokens" in kwargs
        assert "reasoning_effort" not in kwargs

    def test_older_gpt_5_effort_passes_through_unchanged(self):
        kwargs = request(openai_llm("gpt-5", ReasoningEffort.MINIMAL))
        assert kwargs["reasoning_effort"] == "minimal"


class TestUsageMetrics:
    def test_cache_reads_and_writes_reported(self):
        usage = SimpleNamespace(
            prompt_tokens=2000,
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=1500, cache_write_tokens=400
            ),
        )
        metrics = openai_llm(TERRA)._usage_metrics(usage)
        assert (metrics.in_t, metrics.op_t) == (2000, 50)
        assert (metrics.cache_read_t, metrics.cache_write_t) == (1500, 400)

    def test_missing_details_default_to_zero(self):
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None
        )
        metrics = openai_llm(LUNA)._usage_metrics(usage)
        assert (metrics.cache_read_t, metrics.cache_write_t) == (0, 0)


# --- live API ----------------------------------------------------------------

requires_key = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def live_llm(model: str, effort: ReasoningEffort = ReasoningEffort.LOW):
    # max_tokens covers reasoning tokens too, so leave headroom.
    return get_llm(
        LLMConfig(
            provider="openai",
            model=model,
            max_tokens=4096,
            thinking=ThinkingConfig(reasoning_effort=effort),
        )
    )


@requires_key
class TestOpenAINewModelsIntegration:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ALL_MODELS)
    async def test_simple_text(self, model):
        response, context = await live_llm(model).invoke(
            user_context("Reply with exactly the word: pong")
        )
        assert "pong" in response.text.lower()
        usage = context.messages[-1].usage
        assert usage.in_t > 0 and usage.op_t > 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ALL_MODELS)
    async def test_streaming(self, model):
        events = [
            event
            async for event in live_llm(model).invoke_stream(
                user_context("Count from 1 to 5, separated by spaces.")
            )
        ]
        assert not [e for e in events if e.type == StreamEventType.ERROR]
        text = "".join(e.text for e in events if e.type == StreamEventType.TEXT)
        assert "3" in text
        assert events[-1].type == StreamEventType.DONE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", GPT_5_6_MODELS)
    async def test_tool_call(self, model):
        response, _ = await live_llm(model).invoke(
            user_context("What time is it? Use the tool."), tools=[TimeTool()]
        )
        assert any(
            v.type == "tool" and v.tool_name == "get_current_time"
            for v in response.verbose
        )
        assert "10:30" in response.text

    @pytest.mark.asyncio
    async def test_astra_with_tools_fails_fast(self):
        with pytest.raises(ValueError, match="Responses API"):
            await live_llm(ASTRA).invoke(user_context("hi"), tools=[TimeTool()])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ALL_MODELS)
    async def test_prompt_caching(self, model):
        # Caching needs >= 1024 input tokens and is best-effort on OpenAI's
        # side, so allow a few attempts before calling it a miss.
        system_prompt = "You are a terse assistant. " + " ".join(
            f"Rule {i}: answer in as few words as possible." for i in range(200)
        )
        llm = live_llm(model)
        cache_reads = []
        for _ in range(3):
            _, context = await llm.invoke(
                user_context("Say ok."), system_prompt=system_prompt
            )
            cache_reads.append(context.messages[-1].usage.cache_read_t)
            if cache_reads[-1] > 0:
                break
        assert cache_reads[-1] > 0, f"no cache hit after 3 calls: {cache_reads}"
