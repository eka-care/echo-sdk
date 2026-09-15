"""Tests for the TLS options of the openai_compatible LLM provider
(ECHO_LLM_VERIFY_SSL / ECHO_LLM_CA_BUNDLE for endpoints behind a private CA)."""

import pytest

from echo.llm import LLMConfig
from echo.llm.openai_compatible import (
    OpenAICompatibleLLM,
    build_custom_http_client,
    resolve_ssl_verify,
)

VERIFY_VARS = ("ECHO_LLM_VERIFY_SSL",)
CA_VARS = ("ECHO_LLM_CA_BUNDLE",)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in VERIFY_VARS + CA_VARS:
        monkeypatch.delenv(var, raising=False)


def test_ssl_verify_default_true():
    assert resolve_ssl_verify(VERIFY_VARS, CA_VARS) is True


def test_ssl_verify_disabled(monkeypatch, caplog):
    monkeypatch.setenv("ECHO_LLM_VERIFY_SSL", "false")
    with caplog.at_level("WARNING"):
        result = resolve_ssl_verify(VERIFY_VARS, CA_VARS)
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
    monkeypatch.setenv("ECHO_LLM_CA_BUNDLE", "/certs/private-ca.pem")
    result = oc.resolve_ssl_verify(VERIFY_VARS, CA_VARS)
    assert result is sentinel
    assert seen["cafile"] == "/certs/private-ca.pem"


def test_ssl_disable_wins_over_ca_bundle(monkeypatch):
    monkeypatch.setenv("ECHO_LLM_VERIFY_SSL", "false")
    monkeypatch.setenv("ECHO_LLM_CA_BUNDLE", "/certs/ca.pem")
    assert resolve_ssl_verify(VERIFY_VARS, CA_VARS) is False


def test_build_custom_http_client(monkeypatch):
    import httpx

    assert build_custom_http_client(VERIFY_VARS, CA_VARS) is None
    monkeypatch.setenv("ECHO_LLM_VERIFY_SSL", "false")
    client = build_custom_http_client(VERIFY_VARS, CA_VARS)
    assert isinstance(client, httpx.Client)
    assert client.timeout.read == 600.0


def test_client_with_verify_disabled(monkeypatch):
    monkeypatch.setenv("ECHO_LLM_VERIFY_SSL", "false")
    llm = OpenAICompatibleLLM(
        LLMConfig(
            provider="openai_compatible",
            model="m",
            base_url="https://llm.internal/v1",
            api_key="k",
        )
    )
    client = llm.client  # constructs the OpenAI client with the custom http_client
    assert str(client.base_url).rstrip("/") == "https://llm.internal/v1"
