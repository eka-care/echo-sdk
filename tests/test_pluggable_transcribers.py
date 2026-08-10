"""Tests for the pluggable HTTP transcription providers (openai_compatible, model_api)."""

import asyncio

import orjson
import pytest

from echo.audio.transcription.config import TranscriberConfig
from echo.audio.transcription.factory import (
    generate_transcriber_config,
    get_transcriber,
)
from echo.audio.transcription.model_api import ModelApiTranscriber
from echo.audio.transcription.openai_compatible import OpenAICompatibleTranscriber

ENV_VARS = [
    "ECHO_DEFAULT_TRANSCRIBER_PROVIDER",
    "ECHO_DEFAULT_TRANSCRIBER_MODEL",
    "ECHO_TRANSCRIBER_BASE_URL",
    "ECHO_TRANSCRIBER_API_KEY",
    "MODEL_API_TRANSCRIBE_URL",
    "MODEL_API_AUTH_TOKEN",
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


def test_model_api_registered():
    t = get_transcriber(
        TranscriberConfig(provider="model_api", base_url="http://model-host/transcribe")
    )
    assert isinstance(t, ModelApiTranscriber)
    assert t.endpoint == "http://model-host/transcribe"


def test_env_default_provider(monkeypatch):
    monkeypatch.setenv("ECHO_DEFAULT_TRANSCRIBER_PROVIDER", "model_api")
    monkeypatch.setenv("MODEL_API_TRANSCRIBE_URL", "http://model-host/transcribe")
    config = TranscriberConfig()
    assert config.provider == "model_api"
    assert get_transcriber(config).endpoint == "http://model-host/transcribe"


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
        provider="model_api", base_url="http://model-host/transcribe"
    )
    assert config.base_url == "http://model-host/transcribe"


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


# ---------------------------------------------------------------- model_api


def test_model_api_request_shape():
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://model-host/transcribe")
    )
    fake = FakeClient(FakeResponse(json_body={"text": "hello"}))
    t._client = fake
    result = asyncio.run(t.transcribe(b"\x00", mime_type="audio/wav", language="hi"))
    call = fake.calls[0]
    assert call["url"] == "http://model-host/transcribe"
    assert call["params"] == {"language": "hi"}
    assert "Authorization" not in call["headers"]
    assert call["files"]["file"][0] == "audio.wav"
    assert result.text == "hello"


def test_model_api_no_language_no_params():
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://model-host/transcribe")
    )
    fake = FakeClient(FakeResponse(json_body={"text": "hello"}))
    t._client = fake
    asyncio.run(t.transcribe(b"\x00"))
    assert fake.calls[0]["params"] is None


def test_model_api_bearer_from_env(monkeypatch):
    monkeypatch.setenv("MODEL_API_AUTH_TOKEN", "tok-123")
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://model-host/transcribe")
    )
    fake = FakeClient(FakeResponse(json_body={"text": "hello"}))
    t._client = fake
    asyncio.run(t.transcribe(b"\x00"))
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer tok-123"


@pytest.mark.parametrize(
    "body,expected_text",
    [
        ({"text": " hi "}, "hi"),
        ({"transcript": "hi"}, "hi"),
        ({"transcription": "hi"}, "hi"),
        ({"data": {"text": "hi"}}, "hi"),
        ("hi", "hi"),
    ],
)
def test_model_api_lenient_parse(body, expected_text):
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://h/t")
    )
    result = t._parse_response(FakeResponse(json_body=body), "hi")
    assert result.error is None
    assert result.text == expected_text


def test_model_api_plain_text_body():
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://h/t")
    )
    result = t._parse_response(FakeResponse(text_body="plain transcript"), "hi")
    assert result.error is None
    assert result.text == "plain transcript"
    assert result.language_detected == "hi"


def test_model_api_envelope_metadata():
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://h/t")
    )
    body = {"data": {"transcript": "hi", "language": "hi-IN", "audio_duration": 4.5}}
    result = t._parse_response(FakeResponse(json_body=body), None)
    assert result.text == "hi"
    assert result.language_detected == "hi-IN"
    assert result.duration_s == 4.5


def test_model_api_unrecognized_payload():
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://h/t")
    )
    result = t._parse_response(FakeResponse(json_body={"status": "ok"}), None)
    assert result.text == ""
    assert "unrecognized" in result.error


def test_model_api_http_error():
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://h/t")
    )
    result = t._parse_response(FakeResponse(status_code=404, text_body="nope"), None)
    assert result.text == ""
    assert "404" in result.error


def test_model_api_requires_endpoint():
    t = ModelApiTranscriber(TranscriberConfig(provider="model_api"))
    result = asyncio.run(t.transcribe(b"\x00"))
    assert result.text == ""
    assert "MODEL_API_TRANSCRIBE_URL" in result.error


def test_model_api_rejects_non_bytes():
    t = ModelApiTranscriber(
        TranscriberConfig(provider="model_api", base_url="http://h/t")
    )
    result = asyncio.run(t.transcribe("/path/audio.mp3"))
    assert result.text == ""
    assert "bytes" in result.error
