"""Tests for the Open WebUI LLM provider."""

import pytest

from echo.llm import LLMConfig, get_llm
from echo.llm.openwebui import OpenWebUILLM, _normalize_base_url

ENV_VARS = [
    "OPENWEBUI_BASE_URL",
    "OPENWEBUI_API_KEY",
    "OPENWEBUI_ENABLE_THINKING",
    "OPENWEBUI_CHAT_TEMPLATE_KWARGS",
    "OPENWEBUI_VERIFY_SSL",
    "OPENWEBUI_CA_BUNDLE",
    "OPENWEBUI_DISABLE_TOOLS",
    "ECHO_LLM_VERIFY_SSL",
    "ECHO_LLM_CA_BUNDLE",
    "ECHO_LLM_BASE_URL",
    "ECHO_LLM_API_KEY",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _config(**kwargs):
    kwargs.setdefault("provider", "openwebui")
    kwargs.setdefault("model", "qwen3:14b")
    return LLMConfig(**kwargs)


def test_openwebui_registered_and_api_suffix():
    llm = get_llm(_config(base_url="http://openwebui.local:3000"))
    assert isinstance(llm, OpenWebUILLM)
    assert llm.base_url == "http://openwebui.local:3000/api"


def test_api_suffix_not_doubled():
    llm = OpenWebUILLM(_config(base_url="http://openwebui.local:3000/api/"))
    assert llm.base_url == "http://openwebui.local:3000/api"


def test_normalize_handles_path_prefix():
    assert _normalize_base_url("https://ai.corp.in/webui/") == "https://ai.corp.in/webui/api"


def test_env_base_url(monkeypatch):
    monkeypatch.setenv("OPENWEBUI_BASE_URL", "http://owui:8080")
    assert OpenWebUILLM(_config()).base_url == "http://owui:8080/api"


def test_openwebui_env_beats_generic_env(monkeypatch):
    monkeypatch.setenv("OPENWEBUI_BASE_URL", "http://owui:8080")
    monkeypatch.setenv("ECHO_LLM_BASE_URL", "http://other:1111/v1")
    assert OpenWebUILLM(_config()).base_url == "http://owui:8080/api"


def test_generic_env_fallback(monkeypatch):
    monkeypatch.setenv("ECHO_LLM_BASE_URL", "http://shared-llm:3000")
    assert OpenWebUILLM(_config()).base_url == "http://shared-llm:3000/api"


def test_default_base_url():
    assert OpenWebUILLM(_config()).base_url == "http://localhost:3000/api"


def test_api_key_from_config():
    llm = OpenWebUILLM(_config(base_url="http://owui:3000", api_key="sk-owui-cfg"))
    assert llm.client.api_key == "sk-owui-cfg"


def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("OPENWEBUI_API_KEY", "sk-owui-env")
    monkeypatch.setenv("ECHO_LLM_API_KEY", "sk-generic")
    llm = OpenWebUILLM(_config(base_url="http://owui:3000"))
    assert llm.client.api_key == "sk-owui-env"


def test_api_key_generic_fallback(monkeypatch):
    monkeypatch.setenv("ECHO_LLM_API_KEY", "sk-generic")
    llm = OpenWebUILLM(_config(base_url="http://owui:3000"))
    assert llm.client.api_key == "sk-generic"


def test_missing_api_key_warns_not_raises(caplog):
    llm = OpenWebUILLM(_config(base_url="http://owui:3000"))
    with caplog.at_level("WARNING"):
        client = llm.client
    assert client.api_key == "not-needed"
    assert any("Open WebUI API key" in r.message for r in caplog.records)


def test_client_base_url_points_at_api():
    llm = OpenWebUILLM(_config(base_url="http://owui:3000", api_key="k"))
    # the OpenAI SDK appends /chat/completions to this
    assert str(llm.client.base_url).rstrip("/") == "http://owui:3000/api"


def test_open_model_capability_flags():
    llm = OpenWebUILLM(_config(base_url="http://owui:3000"))
    assert llm._uses_max_completion_tokens() is False
    assert llm._supports_reasoning_effort() is False
    assert llm._is_reasoning_model() is False


# ---------------------------------------------------------------- extra_body


def test_extra_body_default_none():
    assert OpenWebUILLM(_config())._extra_body() is None


@pytest.mark.parametrize(
    "raw,expected", [("false", False), ("true", True), ("0", False), ("1", True)]
)
def test_extra_body_enable_thinking_flag(monkeypatch, raw, expected):
    monkeypatch.setenv("OPENWEBUI_ENABLE_THINKING", raw)
    assert OpenWebUILLM(_config())._extra_body() == {
        "chat_template_kwargs": {"enable_thinking": expected}
    }


def test_extra_body_raw_json(monkeypatch):
    monkeypatch.setenv(
        "OPENWEBUI_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false, "custom": 1}'
    )
    assert OpenWebUILLM(_config())._extra_body() == {
        "chat_template_kwargs": {"enable_thinking": False, "custom": 1}
    }


def test_extra_body_flag_wins_over_raw_json(monkeypatch):
    monkeypatch.setenv("OPENWEBUI_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": true}')
    monkeypatch.setenv("OPENWEBUI_ENABLE_THINKING", "false")
    body = OpenWebUILLM(_config())._extra_body()
    assert body["chat_template_kwargs"]["enable_thinking"] is False


def test_extra_body_invalid_json_ignored(monkeypatch, caplog):
    monkeypatch.setenv("OPENWEBUI_CHAT_TEMPLATE_KWARGS", "not-json")
    with caplog.at_level("WARNING"):
        assert OpenWebUILLM(_config())._extra_body() is None
    assert any("OPENWEBUI_CHAT_TEMPLATE_KWARGS" in r.message for r in caplog.records)


# ------------------------------------------------- request wiring via invoke


import asyncio
from types import SimpleNamespace


def _fake_response(text="ok"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _fake_response()


class _FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_invoke_sends_chat_template_kwargs(monkeypatch):
    monkeypatch.setenv("OPENWEBUI_ENABLE_THINKING", "false")
    from echo.models.user_conversation import ConversationContext

    llm = OpenWebUILLM(_config(base_url="http://owui:3000", api_key="k"))
    fake = _FakeOpenAIClient()
    llm._client = fake
    response, _ = asyncio.run(
        llm.invoke(ConversationContext(), system_prompt="Return only valid JSON.")
    )
    call = fake.chat.completions.calls[0]
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert call["model"] == "qwen3:14b"
    assert "max_tokens" in call and "max_completion_tokens" not in call
    assert "temperature" in call
    assert response.text == "ok"


def test_invoke_no_extra_body_when_unconfigured():
    from echo.models.user_conversation import ConversationContext

    llm = OpenWebUILLM(_config(base_url="http://owui:3000", api_key="k"))
    fake = _FakeOpenAIClient()
    llm._client = fake
    asyncio.run(llm.invoke(ConversationContext(), system_prompt="s"))
    assert "extra_body" not in fake.chat.completions.calls[0]


def test_openai_compatible_never_sends_extra_body(monkeypatch):
    # the openwebui env toggle must not leak into the generic provider
    monkeypatch.setenv("OPENWEBUI_ENABLE_THINKING", "false")
    from echo.llm.openai_compatible import OpenAICompatibleLLM
    from echo.models.user_conversation import ConversationContext

    llm = OpenAICompatibleLLM(
        LLMConfig(
            provider="openai_compatible",
            model="qwen3:14b",
            base_url="http://vllm:8000/v1",
            api_key="k",
        )
    )
    fake = _FakeOpenAIClient()
    llm._client = fake
    asyncio.run(llm.invoke(ConversationContext(), system_prompt="s"))
    assert "extra_body" not in fake.chat.completions.calls[0]


# ------------------------------------------------------------------ TLS config


def test_ssl_verify_default_true():
    from echo.llm.openai_compatible import resolve_ssl_verify

    assert resolve_ssl_verify(("OPENWEBUI_VERIFY_SSL",), ("OPENWEBUI_CA_BUNDLE",)) is True


def test_ssl_verify_disabled(monkeypatch, caplog):
    from echo.llm.openai_compatible import resolve_ssl_verify

    monkeypatch.setenv("OPENWEBUI_VERIFY_SSL", "false")
    with caplog.at_level("WARNING"):
        result = resolve_ssl_verify(
            ("OPENWEBUI_VERIFY_SSL", "ECHO_LLM_VERIFY_SSL"),
            ("OPENWEBUI_CA_BUNDLE",),
        )
    assert result is False
    assert any("DISABLED" in r.message for r in caplog.records)


def test_ssl_ca_bundle_context(monkeypatch):
    import echo.llm.openai_compatible as oc

    sentinel = object()
    seen = {}

    def fake_ctx(cafile=None):
        seen["cafile"] = cafile
        return sentinel

    monkeypatch.setattr(oc.ssl, "create_default_context", fake_ctx)
    monkeypatch.setenv("OPENWEBUI_CA_BUNDLE", "/certs/bharatai-ca.pem")
    result = oc.resolve_ssl_verify(
        ("OPENWEBUI_VERIFY_SSL",), ("OPENWEBUI_CA_BUNDLE", "ECHO_LLM_CA_BUNDLE")
    )
    assert result is sentinel
    assert seen["cafile"] == "/certs/bharatai-ca.pem"


def test_ssl_disable_wins_over_ca_bundle(monkeypatch):
    from echo.llm.openai_compatible import resolve_ssl_verify

    monkeypatch.setenv("OPENWEBUI_VERIFY_SSL", "false")
    monkeypatch.setenv("OPENWEBUI_CA_BUNDLE", "/certs/ca.pem")
    assert (
        resolve_ssl_verify(("OPENWEBUI_VERIFY_SSL",), ("OPENWEBUI_CA_BUNDLE",)) is False
    )


def test_build_custom_http_client(monkeypatch):
    import httpx

    from echo.llm.openai_compatible import build_custom_http_client

    assert (
        build_custom_http_client(("OPENWEBUI_VERIFY_SSL",), ("OPENWEBUI_CA_BUNDLE",))
        is None
    )
    monkeypatch.setenv("OPENWEBUI_VERIFY_SSL", "false")
    client = build_custom_http_client(
        ("OPENWEBUI_VERIFY_SSL",), ("OPENWEBUI_CA_BUNDLE",)
    )
    assert isinstance(client, httpx.Client)
    assert client.timeout.read == 600.0


def test_openwebui_client_with_verify_disabled(monkeypatch):
    monkeypatch.setenv("OPENWEBUI_VERIFY_SSL", "false")
    llm = OpenWebUILLM(_config(base_url="https://bharatai.gov.in", api_key="k"))
    client = llm.client  # constructs OpenAI client with custom http_client
    assert str(client.base_url).rstrip("/") == "https://bharatai.gov.in/api"


# ---------------------------------------------------------------- tools toggle


class _FakeTool:
    name = "get_weather"

    def to_openai_schema(self):
        return {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {"type": "object", "properties": {}},
            },
        }


def test_tools_enabled_by_default():
    llm = OpenWebUILLM(_config(base_url="http://owui:3000", api_key="k"))
    assert llm._tools_enabled() is True
    from echo.models.user_conversation import ConversationContext

    fake = _FakeOpenAIClient()
    llm._client = fake
    asyncio.run(
        llm.invoke(ConversationContext(), tools=[_FakeTool()], system_prompt="s")
    )
    assert "tools" in fake.chat.completions.calls[0]


@pytest.mark.parametrize("raw", ["true", "1", "yes", "on"])
def test_disable_tools_env(monkeypatch, raw, caplog):
    monkeypatch.setenv("OPENWEBUI_DISABLE_TOOLS", raw)
    from echo.models.user_conversation import ConversationContext

    llm = OpenWebUILLM(_config(base_url="http://owui:3000", api_key="k"))
    fake = _FakeOpenAIClient()
    llm._client = fake
    with caplog.at_level("WARNING"):
        asyncio.run(
            llm.invoke(ConversationContext(), tools=[_FakeTool()], system_prompt="s")
        )
    assert "tools" not in fake.chat.completions.calls[0]
    assert any("OPENWEBUI_DISABLE_TOOLS" in r.message for r in caplog.records)


def test_disable_tools_false_value_keeps_tools(monkeypatch):
    monkeypatch.setenv("OPENWEBUI_DISABLE_TOOLS", "false")
    llm = OpenWebUILLM(_config(base_url="http://owui:3000", api_key="k"))
    assert llm._tools_enabled() is True


def test_openai_compatible_ignores_disable_tools(monkeypatch):
    monkeypatch.setenv("OPENWEBUI_DISABLE_TOOLS", "true")
    from echo.llm.openai_compatible import OpenAICompatibleLLM
    from echo.models.user_conversation import ConversationContext

    llm = OpenAICompatibleLLM(
        LLMConfig(
            provider="openai_compatible",
            model="m",
            base_url="http://vllm:8000/v1",
            api_key="k",
        )
    )
    fake = _FakeOpenAIClient()
    llm._client = fake
    asyncio.run(
        llm.invoke(ConversationContext(), tools=[_FakeTool()], system_prompt="s")
    )
    assert "tools" in fake.chat.completions.calls[0]
