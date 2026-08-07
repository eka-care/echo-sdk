"""Tests for the on-prem provider additions (sarvam, openai_compatible, file prompts)."""

import asyncio

import pytest


def test_backcompat_aliases():
    from echo import AgentConfig, PersonaConfig, TaskConfig
    from echo.prompts.schemas import AgentPrompt, PromptPersona, PromptTask

    assert AgentConfig is AgentPrompt
    assert PersonaConfig is PromptPersona
    assert TaskConfig is PromptTask


def test_sarvam_provider_registered(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    from echo.audio.transcription.config import TranscriberConfig
    from echo.audio.transcription.factory import get_transcriber

    config = TranscriberConfig(provider="sarvam", language="hi")
    assert config.model == "saarika:v2.5"
    t = get_transcriber(config)
    assert t.base_url is None  # SDK default endpoint
    assert t._language_code() == "hi-IN"


def test_sarvam_base_url_override(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    from echo.audio.transcription.config import TranscriberConfig
    from echo.audio.transcription.factory import get_transcriber

    t = get_transcriber(TranscriberConfig(provider="sarvam", base_url="https://proxy.local"))
    assert t.client._client_wrapper.get_environment().base == "https://proxy.local"


def test_openai_compatible_llm(monkeypatch):
    monkeypatch.delenv("ECHO_LLM_BASE_URL", raising=False)
    from echo.llm import LLMConfig, get_llm

    llm = get_llm(
        LLMConfig(
            provider="openai_compatible",
            model="qwen3:14b",
            base_url="http://vllm.local:8000/v1",
        )
    )
    assert llm.base_url == "http://vllm.local:8000/v1"
    assert llm._uses_max_completion_tokens() is False
    assert llm._supports_reasoning_effort() is False


def test_file_prompt_provider_langfuse_semantics(tmp_path):
    (tmp_path / "my_agent.md").write_text(
        'You are {{role_name}}. Output JSON like {"a": 1, "b": {{count}}}.'
    )
    from echo.prompts.file_provider import FilePromptProvider

    provider = FilePromptProvider(prompt_dir=str(tmp_path))
    fetched = asyncio.run(
        provider.get_prompt("my_agent", prompt_variables={"role_name": "a scribe", "count": 3})
    )
    desc = fetched.agent_prompt.task.description
    assert "You are a scribe." in desc
    assert '{"a": 1, "b": 3}' in desc  # literal braces survive


def test_file_prompt_provider_versions(tmp_path):
    d = tmp_path / "team_summary_agent"
    d.mkdir()
    (d / "1.yaml").write_text("prompt: v1 body\n")
    (d / "2.yaml").write_text("prompt: v2 body\nrole: summarizer\n")
    (d / "production").write_text("2\n")
    from echo.prompts.file_provider import FilePromptProvider

    provider = FilePromptProvider(prompt_dir=str(tmp_path))
    fetched = asyncio.run(provider.get_prompt("team/summary/agent"))
    assert fetched.version == "2"
    assert fetched.agent_prompt.persona.role == "summarizer"


def test_file_prompt_provider_missing(tmp_path):
    from echo.prompts.base import PromptFetchError
    from echo.prompts.file_provider import FilePromptProvider

    with pytest.raises(PromptFetchError):
        asyncio.run(FilePromptProvider(prompt_dir=str(tmp_path)).get_prompt("nope"))


def test_prompt_factory_file_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("ECHO_PROMPT_PROVIDER", "file")
    monkeypatch.setenv("ECHO_PROMPT_DIR", str(tmp_path))
    from echo.prompts.factory import get_prompt_provider, reset_prompt_provider
    from echo.prompts.file_provider import FilePromptProvider

    reset_prompt_provider()
    assert isinstance(get_prompt_provider(), FilePromptProvider)
    reset_prompt_provider()
