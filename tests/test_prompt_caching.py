"""Prompt-cache prefix behaviour.

The invariant under test: the cacheable half of the system prompt is
byte-identical across sessions of one agent, and the volatile half never
lands inside it. Everything else here follows from that.
"""

from echo.llm import prompt_cache_id
from echo.llm.anthropic import AnthropicLLM
from echo.llm.bedrock import BedrockLLM
from echo.llm.config import LLMConfig
from echo.llm.model_capabilities import claude_capabilities
from echo.llm.openai import OpenAILLM
from echo.prompts.schemas import AgentPrompt, PromptPersona, PromptTask

STABLE = "You are a clinic assistant.\n\nBook appointments."
VOLATILE_A = "<USER_CONTEXT>{}</USER_CONTEXT>\n\nSession context: {'sid': 'a'}"
VOLATILE_B = "<USER_CONTEXT>{}</USER_CONTEXT>\n\nSession context: {'sid': 'b'}"


# --- prompt_cache_id --------------------------------------------------------


def test_cache_id_is_stable_for_identical_prompts():
    assert prompt_cache_id(STABLE) == prompt_cache_id(STABLE)


def test_cache_id_differs_between_agents():
    assert prompt_cache_id(STABLE) != prompt_cache_id(STABLE + " Also triage.")


def test_cache_id_of_empty_prompt_is_none():
    assert prompt_cache_id(None) is None
    assert prompt_cache_id("") is None


# --- Anthropic --------------------------------------------------------------


def test_anthropic_marks_only_the_stable_block():
    blocks = AnthropicLLM._cached_system(STABLE, VOLATILE_A)

    assert len(blocks) == 2
    assert blocks[0]["text"] == STABLE
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == VOLATILE_A
    assert "cache_control" not in blocks[1]


def test_anthropic_cached_block_is_identical_across_sessions():
    """The whole point: two sessions differing only in volatile context must
    present the same cached prefix, or neither can read the other's entry."""
    a = AnthropicLLM._cached_system(STABLE, VOLATILE_A)
    b = AnthropicLLM._cached_system(STABLE, VOLATILE_B)

    assert a[0] == b[0]
    assert a[1] != b[1]


def test_anthropic_omits_empty_suffix_block():
    assert AnthropicLLM._cached_system(STABLE, None) == [
        {"type": "text", "text": STABLE, "cache_control": {"type": "ephemeral"}}
    ]


# --- Bedrock ----------------------------------------------------------------


def _bedrock(model: str) -> BedrockLLM:
    return BedrockLLM(LLMConfig(provider="bedrock", model=model))


def test_bedrock_inserts_cache_point_between_halves():
    llm = _bedrock("anthropic.claude-sonnet-4-5")
    blocks = llm._system_blocks(STABLE, VOLATILE_A)

    assert blocks == [
        {"text": STABLE},
        {"cachePoint": {"type": "default"}},
        {"text": VOLATILE_A},
    ]


def test_bedrock_omits_cache_point_on_models_that_reject_it():
    """A cachePoint on a pre-caching model is a ValidationException, not a
    silent no-op — so the prompt must still go out without one."""
    llm = _bedrock("anthropic.claude-3-sonnet-20240229")
    blocks = llm._system_blocks(STABLE, VOLATILE_A)

    assert blocks == [{"text": STABLE}, {"text": VOLATILE_A}]


def test_bedrock_omits_cache_point_for_non_claude_models():
    blocks = _bedrock("amazon.nova-lite-v1:0")._system_blocks(STABLE, VOLATILE_A)

    assert all("cachePoint" not in b for b in blocks)


# --- OpenAI -----------------------------------------------------------------


def test_openai_puts_the_stable_half_first():
    content = OpenAILLM._system_content(STABLE, VOLATILE_A)

    assert content.startswith(STABLE)
    assert content.endswith(VOLATILE_A)


def test_openai_content_is_unchanged_without_a_suffix():
    assert OpenAILLM._system_content(STABLE, None) == STABLE


# --- capability flag --------------------------------------------------------


def test_prompt_caching_capability_by_model():
    supported = [
        "claude-opus-5",
        "anthropic.claude-sonnet-5",
        "claude-haiku-4-5",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-haiku-20241022",
    ]
    unsupported = [
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229",
        "amazon.nova-lite-v1:0",
    ]
    for model in supported:
        assert claude_capabilities(model).supports_prompt_caching, model
    for model in unsupported:
        assert not claude_capabilities(model).supports_prompt_caching, model


# --- AgentPrompt ------------------------------------------------------------


def test_agent_prompt_context_defaults_to_none():
    prompt = AgentPrompt(
        persona=PromptPersona(role="assistant"), task=PromptTask(description="do it")
    )

    assert prompt.context is None
