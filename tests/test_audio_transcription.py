"""
Unit tests for audio transcription module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from echo.audio import TranscriptionConfig, TranscriptionResult, TranscriptionSegment, get_transcriber
from echo.audio.openai import OpenAITranscriber
from echo.audio.gemini import GeminiTranscriber


# ─── Config validation ───────────────────────────────────────────────


class TestTranscriptionConfig:
    def test_valid_openai_model(self):
        config = TranscriptionConfig(provider="openai", model="whisper-1")
        assert config.provider == "openai"
        assert config.model == "whisper-1"

    def test_valid_openai_gpt4o_transcribe(self):
        config = TranscriptionConfig(provider="openai", model="gpt-4o-transcribe")
        assert config.model == "gpt-4o-transcribe"

    def test_valid_gemini_model(self):
        config = TranscriptionConfig(provider="gemini", model="models/gemini-2.0-flash")
        assert config.provider == "gemini"

    def test_invalid_openai_model(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            TranscriptionConfig(provider="openai", model="not-a-real-model")

    def test_invalid_gemini_model(self):
        with pytest.raises(ValueError, match="Unsupported model"):
            TranscriptionConfig(provider="gemini", model="not-a-real-model")

    def test_optional_fields(self):
        config = TranscriptionConfig(
            provider="openai",
            model="whisper-1",
            language="en",
            response_format="text",
            api_key="sk-test",
            prompt="transcribe this",
        )
        assert config.language == "en"
        assert config.response_format == "text"
        assert config.api_key == "sk-test"
        assert config.prompt == "transcribe this"

    def test_no_defaults_for_provider_and_model(self):
        """Provider and model are required — no env var defaults."""
        with pytest.raises(Exception):
            TranscriptionConfig()


# ─── Factory ─────────────────────────────────────────────────────────


class TestFactory:
    def test_creates_openai_transcriber(self):
        config = TranscriptionConfig(provider="openai", model="whisper-1")
        transcriber = get_transcriber(config)
        assert isinstance(transcriber, OpenAITranscriber)

    def test_creates_gemini_transcriber(self):
        config = TranscriptionConfig(provider="gemini", model="models/gemini-2.0-flash")
        transcriber = get_transcriber(config)
        assert isinstance(transcriber, GeminiTranscriber)


# ─── OpenAI transcriber ─────────────────────────────────────────────


class TestOpenAITranscriber:
    @pytest.mark.asyncio
    async def test_transcribe_builds_correct_request(self):
        config = TranscriptionConfig(provider="openai", model="whisper-1", api_key="sk-test")
        transcriber = OpenAITranscriber(config)

        mock_response = MagicMock()
        mock_response.text = "Hello world"
        mock_response.segments = [
            {"text": "Hello world", "start": 0.0, "end": 1.5},
        ]
        mock_response.language = "en"
        mock_response.duration = 1.5
        mock_response.model_dump = MagicMock(return_value={"text": "Hello world"})

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        transcriber._client = mock_client

        result = await transcriber.transcribe(b"\x00\x01\x02", "audio/wav")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 1.5
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert result.segments[0].start == 0.0
        assert result.segments[0].end == 1.5

        # Verify the call was made with correct args
        call_kwargs = mock_client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-1"
        assert call_kwargs.kwargs["response_format"] == "verbose_json"
        # file tuple: (filename, BytesIO, mime_type)
        file_arg = call_kwargs.kwargs["file"]
        assert file_arg[0] == "audio.wav"
        assert file_arg[2] == "audio/wav"

    @pytest.mark.asyncio
    async def test_transcribe_with_language(self):
        config = TranscriptionConfig(
            provider="openai", model="whisper-1", api_key="sk-test", language="es"
        )
        transcriber = OpenAITranscriber(config)

        mock_response = MagicMock()
        mock_response.text = "Hola mundo"
        mock_response.segments = None
        mock_response.language = "es"
        mock_response.duration = 1.0
        mock_response.model_dump = MagicMock(return_value={})

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        transcriber._client = mock_client

        result = await transcriber.transcribe(b"\x00", "audio/mp3")

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["language"] == "es"
        assert result.text == "Hola mundo"

    @pytest.mark.asyncio
    async def test_transcribe_plain_text_format(self):
        config = TranscriptionConfig(
            provider="openai", model="whisper-1", api_key="sk-test", response_format="text"
        )
        transcriber = OpenAITranscriber(config)

        mock_response = MagicMock()
        mock_response.text = "Plain text result"
        mock_response.model_dump = MagicMock(return_value={})
        # No segments/language/duration for non-verbose format
        mock_response.segments = None
        mock_response.language = None
        mock_response.duration = None

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        transcriber._client = mock_client

        result = await transcriber.transcribe(b"\x00", "audio/wav")

        call_kwargs = mock_client.audio.transcriptions.create.call_args.kwargs
        assert call_kwargs["response_format"] == "text"

    @pytest.mark.asyncio
    async def test_mime_type_mapping(self):
        config = TranscriptionConfig(provider="openai", model="whisper-1", api_key="sk-test")
        transcriber = OpenAITranscriber(config)

        mock_response = MagicMock()
        mock_response.text = "test"
        mock_response.segments = None
        mock_response.language = None
        mock_response.duration = None
        mock_response.model_dump = MagicMock(return_value={})

        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = mock_response
        transcriber._client = mock_client

        await transcriber.transcribe(b"\x00", "audio/mp3")
        file_arg = mock_client.audio.transcriptions.create.call_args.kwargs["file"]
        assert file_arg[0] == "audio.mp3"

        await transcriber.transcribe(b"\x00", "audio/flac")
        file_arg = mock_client.audio.transcriptions.create.call_args.kwargs["file"]
        assert file_arg[0] == "audio.flac"

        # Unknown mime type falls back to audio.wav
        await transcriber.transcribe(b"\x00", "audio/unknown")
        file_arg = mock_client.audio.transcriptions.create.call_args.kwargs["file"]
        assert file_arg[0] == "audio.wav"


# ─── Gemini transcriber ─────────────────────────────────────────────


class TestGeminiTranscriber:
    @pytest.mark.asyncio
    async def test_transcribe_builds_correct_request(self):
        config = TranscriptionConfig(
            provider="gemini", model="models/gemini-2.0-flash", api_key="test-key"
        )
        transcriber = GeminiTranscriber(config)

        mock_response = MagicMock()
        mock_response.text = '{"text": "Hello world", "segments": [{"text": "Hello world", "start": 0.0, "end": 1.5}], "language": "en"}'

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content.return_value = mock_response

        mock_aio = MagicMock()
        mock_aio.models = mock_aio_models

        mock_client = MagicMock()
        mock_client.aio = mock_aio
        transcriber._client = mock_client

        result = await transcriber.transcribe(b"\x00\x01\x02", "audio/wav")

        assert isinstance(result, TranscriptionResult)
        assert result.text == "Hello world"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0].start == 0.0

        # Verify generate_content was called
        mock_aio_models.generate_content.assert_called_once()
        call_kwargs = mock_aio_models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "models/gemini-2.0-flash"

    @pytest.mark.asyncio
    async def test_transcribe_plain_text_fallback(self):
        """When Gemini returns non-JSON, fall back to plain text."""
        config = TranscriptionConfig(
            provider="gemini", model="models/gemini-2.0-flash", api_key="test-key"
        )
        transcriber = GeminiTranscriber(config)

        mock_response = MagicMock()
        mock_response.text = "This is just plain text transcription."

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content.return_value = mock_response

        mock_aio = MagicMock()
        mock_aio.models = mock_aio_models

        mock_client = MagicMock()
        mock_client.aio = mock_aio
        transcriber._client = mock_client

        result = await transcriber.transcribe(b"\x00", "audio/wav")

        assert result.text == "This is just plain text transcription."
        assert result.segments is None
        assert result.language is None

    @pytest.mark.asyncio
    async def test_transcribe_markdown_json_fences(self):
        """Handle JSON wrapped in markdown code fences."""
        config = TranscriptionConfig(
            provider="gemini", model="models/gemini-2.0-flash", api_key="test-key"
        )
        transcriber = GeminiTranscriber(config)

        mock_response = MagicMock()
        mock_response.text = '```json\n{"text": "Hello", "language": "en"}\n```'

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content.return_value = mock_response

        mock_aio = MagicMock()
        mock_aio.models = mock_aio_models

        mock_client = MagicMock()
        mock_client.aio = mock_aio
        transcriber._client = mock_client

        result = await transcriber.transcribe(b"\x00", "audio/wav")

        assert result.text == "Hello"
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_custom_prompt(self):
        """Custom prompt is passed to the API."""
        custom_prompt = "Transcribe in Spanish"
        config = TranscriptionConfig(
            provider="gemini",
            model="models/gemini-2.0-flash",
            api_key="test-key",
            prompt=custom_prompt,
        )
        transcriber = GeminiTranscriber(config)

        mock_response = MagicMock()
        mock_response.text = "Hola mundo"

        mock_aio_models = AsyncMock()
        mock_aio_models.generate_content.return_value = mock_response

        mock_aio = MagicMock()
        mock_aio.models = mock_aio_models

        mock_client = MagicMock()
        mock_client.aio = mock_aio
        transcriber._client = mock_client

        await transcriber.transcribe(b"\x00", "audio/wav")

        call_args = mock_aio_models.generate_content.call_args
        contents = call_args.kwargs["contents"]
        assert contents[0] == custom_prompt


# ─── Schemas ─────────────────────────────────────────────────────────


class TestSchemas:
    def test_transcription_segment(self):
        seg = TranscriptionSegment(text="hello", start=0.0, end=1.0, speaker="A")
        assert seg.text == "hello"
        assert seg.speaker == "A"

    def test_transcription_segment_optional_fields(self):
        seg = TranscriptionSegment(text="hello")
        assert seg.start is None
        assert seg.end is None
        assert seg.speaker is None

    def test_transcription_result(self):
        result = TranscriptionResult(
            text="hello world",
            segments=[TranscriptionSegment(text="hello world", start=0.0, end=1.5)],
            language="en",
            duration=1.5,
        )
        assert result.text == "hello world"
        assert len(result.segments) == 1
        assert result.language == "en"

    def test_transcription_result_minimal(self):
        result = TranscriptionResult(text="hello")
        assert result.segments is None
        assert result.language is None
        assert result.duration is None
        assert result.raw_response is None
