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
    "MODEL_API_BASE_URL",
    "MODEL_API_AUTH_TOKEN",
    "MODEL_API_STT_MODEL",
    "MODEL_API_MAX_TOKENS",
    "MODEL_API_TRANSCRIBE_PROMPT",
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
        TranscriberConfig(provider="model_api", base_url="http://model-host/v1/")
    )
    assert isinstance(t, ModelApiTranscriber)
    assert t.base_url == "http://model-host/v1"
    # gemini-shaped env default swapped for the model API default
    assert t.config.model == "ekascribe"


def test_env_default_provider(monkeypatch):
    monkeypatch.setenv("ECHO_DEFAULT_TRANSCRIBER_PROVIDER", "model_api")
    monkeypatch.setenv("MODEL_API_BASE_URL", "http://model-host/v1")
    config = TranscriberConfig()
    assert config.provider == "model_api"
    assert get_transcriber(config).base_url == "http://model-host/v1"


def test_model_api_legacy_env_honored(monkeypatch):
    monkeypatch.setenv("MODEL_API_TRANSCRIBE_URL", "http://model-host/v1")
    t = ModelApiTranscriber(TranscriberConfig(provider="model_api"))
    assert t.base_url == "http://model-host/v1"


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


def _model_api(**kw):
    kw.setdefault("provider", "model_api")
    kw.setdefault("base_url", "http://model-host/v1")
    return ModelApiTranscriber(TranscriberConfig(**kw))


def test_model_api_request_shape():
    t = _model_api(model="ekascribe-v2", api_key="tok-1")
    fake = FakeClient(
        FakeResponse(json_body={"choices": [{"message": {"content": " namaste "}}]})
    )
    t._client = fake
    audio = b"\x00\x01\x02"
    result = asyncio.run(t.transcribe(audio, mime_type="audio/mp3", language="hi"))
    call = fake.calls[0]
    assert call["url"] == "http://model-host/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer tok-1"
    body = orjson.loads(call["content"])
    assert body["model"] == "ekascribe-v2"
    assert body["temperature"] == 0.0
    assert body["top_p"] == 1.0
    assert body["max_completion_tokens"] == 1024
    parts = body["messages"][0]["content"]
    assert parts[0]["type"] == "text"
    assert "verbatim" in parts[0]["text"]
    assert "<|audio_bos|><audio><|audio_eos|>" in parts[0]["text"]
    assert parts[1]["type"] == "input_audio"
    import base64 as b64

    assert parts[1]["input_audio"]["data"] == b64.b64encode(audio).decode()
    assert parts[1]["input_audio"]["format"] == "mp3"
    assert result.error is None
    assert result.text == "namaste"
    assert result.language_detected == "hi"


def test_model_api_default_auth_is_empty():
    t = _model_api()
    fake = FakeClient(
        FakeResponse(json_body={"choices": [{"message": {"content": "hi"}}]})
    )
    t._client = fake
    asyncio.run(t.transcribe(b"\x00", mime_type="audio/wav"))
    body = orjson.loads(fake.calls[0]["content"])
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer EMPTY"
    assert body["messages"][0]["content"][1]["input_audio"]["format"] == "wav"


def test_model_api_env_overrides(monkeypatch):
    monkeypatch.setenv("MODEL_API_AUTH_TOKEN", "tok-env")
    monkeypatch.setenv("MODEL_API_MAX_TOKENS", "186")
    monkeypatch.setenv("MODEL_API_TRANSCRIBE_PROMPT", "Custom STT prompt.")
    t = _model_api()
    fake = FakeClient(
        FakeResponse(json_body={"choices": [{"message": {"content": "hi"}}]})
    )
    t._client = fake
    asyncio.run(t.transcribe(b"\x00"))
    call = fake.calls[0]
    body = orjson.loads(call["content"])
    assert call["headers"]["Authorization"] == "Bearer tok-env"
    assert body["max_completion_tokens"] == 186
    assert body["messages"][0]["content"][0]["text"] == "Custom STT prompt."


def test_model_api_caller_prompt_wins(monkeypatch):
    monkeypatch.setenv("MODEL_API_TRANSCRIBE_PROMPT", "env prompt")
    t = _model_api()
    fake = FakeClient(
        FakeResponse(json_body={"choices": [{"message": {"content": "hi"}}]})
    )
    t._client = fake
    asyncio.run(t.transcribe(b"\x00", prompt="caller prompt"))
    body = orjson.loads(fake.calls[0]["content"])
    assert body["messages"][0]["content"][0]["text"] == "caller prompt"


def test_model_api_http_error():
    t = _model_api()
    result = t._parse_response(FakeResponse(status_code=422, text_body="bad"), None)
    assert result.text == ""
    assert "422" in result.error


def test_model_api_no_choices_is_error():
    t = _model_api()
    result = t._parse_response(FakeResponse(json_body={"choices": []}), None)
    assert result.text == ""
    assert "no choices" in result.error


def test_model_api_non_json_is_error():
    t = _model_api()
    result = t._parse_response(FakeResponse(text_body="<html>oops</html>"), None)
    assert result.text == ""
    assert "non-JSON" in result.error


def test_model_api_requires_base_url():
    t = ModelApiTranscriber(TranscriberConfig(provider="model_api"))
    result = asyncio.run(t.transcribe(b"\x00"))
    assert result.text == ""
    assert "MODEL_API_BASE_URL" in result.error


def test_model_api_rejects_non_bytes():
    t = _model_api()
    result = asyncio.run(t.transcribe("/path/audio.mp3"))
    assert result.text == ""
    assert "bytes" in result.error
