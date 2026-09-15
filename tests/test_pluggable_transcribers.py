"""Tests for the OpenAI-compatible HTTP transcription provider."""

import asyncio

import orjson
import pytest

from echo.audio.transcription.config import TranscriberConfig
from echo.audio.transcription.factory import (
    generate_transcriber_config,
    get_transcriber,
)
from echo.audio.transcription.openai_compatible import OpenAICompatibleTranscriber

ENV_VARS = [
    "ECHO_DEFAULT_TRANSCRIBER_PROVIDER",
    "ECHO_DEFAULT_TRANSCRIBER_MODEL",
    "ECHO_TRANSCRIBER_BASE_URL",
    "ECHO_TRANSCRIBER_API_KEY",
    "OPENAI_COMPAT_STT_MODEL",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class FakeResponse:
    def __init__(self, status_code=200, json_body=None, text_body=None):
        self.status_code = status_code
        if json_body is not None:
            self.content = orjson.dumps(json_body)
            self.text = self.content.decode()
        else:
            self.text = text_body or ""
            self.content = self.text.encode()


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        return self.response


# ---------------------------------------------------------------- config/factory


def test_openai_compatible_registered():
    config = TranscriberConfig(
        provider="openai_compatible", base_url="http://stt.local:8000/v1/"
    )
    # the gemini-shaped default model gets swapped for the OpenAI default
    assert config.model == "whisper-1"
    t = get_transcriber(config)
    assert isinstance(t, OpenAICompatibleTranscriber)
    assert t.base_url == "http://stt.local:8000/v1"


def test_env_default_provider(monkeypatch):
    monkeypatch.setenv("ECHO_DEFAULT_TRANSCRIBER_PROVIDER", "openai_compatible")
    monkeypatch.setenv("ECHO_TRANSCRIBER_BASE_URL", "http://stt.local:8000/v1")
    config = TranscriberConfig()
    assert config.provider == "openai_compatible"
    assert get_transcriber(config).base_url == "http://stt.local:8000/v1"


def test_openai_compatible_env_base_url(monkeypatch):
    monkeypatch.setenv("ECHO_TRANSCRIBER_BASE_URL", "http://stt.local:8000/v1")
    t = get_transcriber(TranscriberConfig(provider="openai_compatible"))
    assert t.base_url == "http://stt.local:8000/v1"


def test_openai_compat_model_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_COMPAT_STT_MODEL", "whisper-large-v3-turbo")
    assert (
        TranscriberConfig(provider="openai_compatible").model == "whisper-large-v3-turbo"
    )


def test_openai_compat_explicit_model_wins():
    config = TranscriberConfig(provider="openai_compatible", model="my-finetune")
    assert config.model == "my-finetune"


def test_generate_config_passes_base_url():
    config = generate_transcriber_config(
        provider="openai_compatible", base_url="http://stt.local:8000/v1"
    )
    assert config.base_url == "http://stt.local:8000/v1"


# ---------------------------------------------------------------- openai_compatible


def test_openai_compatible_request_shape():
    t = OpenAICompatibleTranscriber(
        TranscriberConfig(
            provider="openai_compatible",
            base_url="http://stt.local:8000/v1",
            model="whisper-large-v3",
            api_key="sk-test",
        )
    )
    fake = FakeClient(
        FakeResponse(json_body={"text": " namaste ", "language": "hi", "duration": 3.2})
    )
    t._client = fake
    result = asyncio.run(
        t.transcribe(
            b"\x00\x01", prompt="medical terms", mime_type="audio/mp3", language="hi-IN"
        )
    )
    call = fake.calls[0]
    assert call["url"] == "http://stt.local:8000/v1/audio/transcriptions"
    assert call["data"]["model"] == "whisper-large-v3"
    assert call["data"]["language"] == "hi"  # ISO part of hi-IN
    assert call["data"]["prompt"] == "medical terms"
    assert call["headers"]["Authorization"] == "Bearer sk-test"
    filename, _, mime = call["files"]["file"]
    assert filename == "audio.mp3"
    assert mime == "audio/mp3"
    assert result.error is None
    assert result.text == "namaste"
    assert result.language_detected == "hi"
    assert result.duration_s == 3.2


def test_openai_compatible_no_auth_no_language():
    t = OpenAICompatibleTranscriber(
        TranscriberConfig(provider="openai_compatible", base_url="http://stt.local/v1")
    )
    fake = FakeClient(FakeResponse(json_body={"text": "hello"}))
    t._client = fake
    result = asyncio.run(t.transcribe(b"\x00", mime_type="audio/wav"))
    call = fake.calls[0]
    assert "Authorization" not in call["headers"]
    assert "language" not in call["data"]
    assert result.text == "hello"
    assert result.language_detected is None


def test_openai_compatible_http_error():
    t = OpenAICompatibleTranscriber(
        TranscriberConfig(provider="openai_compatible", base_url="http://stt.local/v1")
    )
    t._client = FakeClient(FakeResponse(status_code=500, text_body="boom"))
    result = asyncio.run(t.transcribe(b"\x00"))
    assert result.text == ""
    assert "500" in result.error


def test_openai_compatible_requires_base_url():
    t = OpenAICompatibleTranscriber(TranscriberConfig(provider="openai_compatible"))
    result = asyncio.run(t.transcribe(b"\x00"))
    assert result.text == ""
    assert "ECHO_TRANSCRIBER_BASE_URL" in result.error


def test_openai_compatible_rejects_non_bytes():
    t = OpenAICompatibleTranscriber(
        TranscriberConfig(provider="openai_compatible", base_url="http://stt.local/v1")
    )
    result = asyncio.run(t.transcribe("/path/audio.mp3"))
    assert result.text == ""
    assert "bytes" in result.error
