"""Tests for the real-time (streaming) transcription abstraction.

No network: provider websockets are replaced with in-memory fakes that replay
canned messages and record what was sent to them.
"""

import base64
import sys
import types
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from echo.audio import (
    StreamingEventType,
    StreamingTranscriptEvent,
    TranscriberConfig,
    get_streaming_transcriber,
    get_transcriber,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ns(**kw):
    return SimpleNamespace(**kw)


class FakeSarvamSocket:
    """Stands in for sarvamai's AsyncSpeechToTextStreamingSocketClient."""

    def __init__(self, messages, raise_after=None):
        self._messages = list(messages)
        self._raise_after = raise_after
        self.sent = []
        self.flushed = 0
        self._websocket = _ns(closed=False)

        async def _close():
            self._websocket.closed = True

        self._websocket.close = _close

    async def transcribe(self, audio, encoding="audio/wav", sample_rate=16000):
        self.sent.append((audio, encoding, sample_rate))

    async def flush(self):
        self.flushed += 1

    async def __aiter__(self):
        for m in self._messages:
            yield m
        if self._raise_after is not None:
            raise self._raise_after


def _sarvam_transcriber(monkeypatch, socket, **cfg):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    from echo.audio.transcription.sarvam import SarvamStreamingTranscriber

    t = SarvamStreamingTranscriber(TranscriberConfig(provider="sarvam", **cfg))
    captured = {}

    @asynccontextmanager
    async def fake_connect(**kwargs):
        captured.update(kwargs)
        yield socket

    t._client = _ns(speech_to_text_streaming=_ns(connect=fake_connect))
    return t, captured


async def _collect(session):
    return [e async for e in session]


# --------------------------------------------------------------------------- #
# factory / config
# --------------------------------------------------------------------------- #


def test_factory_returns_sarvam_streaming(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    from echo.audio.transcription.sarvam import SarvamStreamingTranscriber

    t = get_streaming_transcriber(TranscriberConfig(provider="sarvam"))
    assert isinstance(t, SarvamStreamingTranscriber)


def test_factory_rejects_batch_only_providers():
    with pytest.raises(ValueError, match="Streaming transcription not supported"):
        get_streaming_transcriber(TranscriberConfig(provider="gemini"))


def test_batch_factory_points_elevenlabs_to_streaming():
    with pytest.raises(ValueError, match="streaming-only"):
        get_transcriber(TranscriberConfig(provider="elevenlabs"))


def test_factory_import_hint_when_sdk_missing(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "elevenlabs", None)  # simulate not installed
    t = get_streaming_transcriber(TranscriberConfig(provider="elevenlabs"))
    with pytest.raises(ImportError, match=r"echo-sdk\[elevenlabs\]"):
        _ = t.client


def test_config_streaming_defaults(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_STT_MODEL", raising=False)
    cfg = TranscriberConfig(provider="elevenlabs")
    assert cfg.model == "scribe_v2_realtime"
    assert cfg.sample_rate == 16000
    assert cfg.encoding == "pcm_s16le"
    assert cfg.commit_strategy == "vad"

    with pytest.raises(ValueError, match="keyterms"):
        TranscriberConfig(provider="elevenlabs", keyterms=["x"] * 51)
    with pytest.raises(ValueError, match="sample_rate"):
        TranscriberConfig(provider="sarvam", sample_rate=0)


# --------------------------------------------------------------------------- #
# Sarvam
# --------------------------------------------------------------------------- #


def test_sarvam_stream_model_selection(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.delenv("SARVAM_STT_STREAM_MODEL", raising=False)
    from echo.audio.transcription.sarvam import SarvamStreamingTranscriber

    # config default (batch model) is treated as unset → saaras:v3
    assert SarvamStreamingTranscriber(TranscriberConfig(provider="sarvam"))._stream_model() == "saaras:v3"
    # explicit streaming model other than the batch default is honoured
    assert (
        SarvamStreamingTranscriber(
            TranscriberConfig(provider="sarvam", model="saaras:v3")
        )._stream_model()
        == "saaras:v3"
    )
    monkeypatch.setenv("SARVAM_STT_STREAM_MODEL", "saarika:v2.5")
    assert SarvamStreamingTranscriber(TranscriberConfig(provider="sarvam"))._stream_model() == "saarika:v2.5"

    with pytest.raises(ValueError, match="16000 or 8000"):
        SarvamStreamingTranscriber(TranscriberConfig(provider="sarvam", sample_rate=44100))


async def test_sarvam_connect_kwargs_and_send(monkeypatch):
    socket = FakeSarvamSocket([])
    t, captured = _sarvam_transcriber(
        monkeypatch, socket, language="hi", mode="verbatim", high_vad_sensitivity=True
    )

    async with t.stream() as session:
        await session.send_audio(b"\x00\x01\x02\x03")
        await session.send_audio(b"")  # ignored
        await session.flush()
        events = await _collect(session)

    assert captured == {
        "language_code": "hi-IN",
        "model": "saaras:v3",
        "sample_rate": "16000",
        "input_audio_codec": "pcm_s16le",
        "vad_signals": "true",
        "flush_signal": "true",
        "high_vad_sensitivity": "true",
        "mode": "verbatim",
    }
    assert socket.sent == [(base64.b64encode(b"\x00\x01\x02\x03").decode(), "audio/wav", 16000)]
    assert socket.flushed == 1
    assert [e.type for e in events] == [StreamingEventType.DONE]
    assert socket._websocket.closed is True  # closed on context exit


async def test_sarvam_event_mapping(monkeypatch):
    messages = [
        _ns(type="events", data=_ns(event_type="vad", signal_type="START", occured_at=0.1)),
        _ns(
            type="data",
            data=_ns(
                transcript="  नमस्ते डॉक्टर  ",
                request_id="r1",
                language_code="hi-IN",
                language_probability=0.98,
                timestamps=None,
                metrics=_ns(model_dump=lambda: {"latency_ms": 120}),
            ),
        ),
        _ns(type="data", data=_ns(transcript="", request_id="r2")),  # empty → dropped
        _ns(type="events", data=_ns(event_type="vad", signal_type="END", occured_at=2.0)),
        _ns(type="error", data=_ns(code="rate_limited", error="slow down")),
        _ns(type="something_new", data=None),  # ignored
    ]
    socket = FakeSarvamSocket(messages)
    t, _ = _sarvam_transcriber(monkeypatch, socket)

    async with t.stream() as session:
        events = await _collect(session)

    assert [e.type for e in events] == [
        StreamingEventType.SPEECH_STARTED,
        StreamingEventType.FINAL,
        StreamingEventType.SPEECH_ENDED,
        StreamingEventType.ERROR,
        StreamingEventType.DONE,
    ]
    final = events[1]
    assert final.text == "नमस्ते डॉक्टर"
    assert final.segment_id == "r1"
    assert final.language_detected == "hi-IN"
    assert final.details["language_probability"] == 0.98
    assert final.details["metrics"] == {"latency_ms": 120}
    err = events[3]
    assert err.recoverable is True
    assert "rate_limited" in err.error


async def test_sarvam_fatal_error_ends_stream(monkeypatch):
    messages = [
        _ns(type="error", data=_ns(code="invalid_api_key", error="bad key")),
        _ns(type="data", data=_ns(transcript="never seen", request_id="r9")),
    ]
    socket = FakeSarvamSocket(messages)
    t, _ = _sarvam_transcriber(monkeypatch, socket)

    async with t.stream() as session:
        events = await _collect(session)

    assert len(events) == 1
    assert events[0].type is StreamingEventType.ERROR
    assert events[0].recoverable is False


async def test_sarvam_socket_exception_is_non_recoverable_error(monkeypatch):
    socket = FakeSarvamSocket([], raise_after=RuntimeError("boom"))
    t, _ = _sarvam_transcriber(monkeypatch, socket)

    async with t.stream() as session:
        events = await _collect(session)

    assert [e.type for e in events] == [StreamingEventType.ERROR]
    assert events[0].recoverable is False
    assert "boom" in events[0].error


async def test_sarvam_clean_websocket_close_is_done(monkeypatch):
    ws_exc = pytest.importorskip("websockets.exceptions")
    exc = ws_exc.ConnectionClosedOK(None, None)
    socket = FakeSarvamSocket([], raise_after=exc)
    t, _ = _sarvam_transcriber(monkeypatch, socket)

    async with t.stream() as session:
        events = await _collect(session)

    assert [e.type for e in events] == [StreamingEventType.DONE]


def test_streaming_event_schema_roundtrip():
    ev = StreamingTranscriptEvent(type="final", text="hi", segment_id="1")
    assert ev.type is StreamingEventType.FINAL
    assert ev.recoverable is True
    assert ev.model_dump()["type"] == "final"


# --------------------------------------------------------------------------- #
# ElevenLabs
# --------------------------------------------------------------------------- #


class FakeElevenLabsConnection:
    def __init__(self, messages, raise_after=None):
        self._messages = list(messages)
        self._raise_after = raise_after
        self.sent = []
        self.commits = 0
        self.closed = False

    async def send(self, payload):
        self.sent.append(payload)

    async def commit(self):
        self.commits += 1

    async def close(self):
        self.closed = True

    async def __aiter__(self):
        for m in self._messages:
            yield m
        if self._raise_after is not None:
            raise self._raise_after


def _elevenlabs_transcriber(monkeypatch, conn, **cfg):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.delenv("ELEVENLABS_STT_MODEL", raising=False)
    from echo.audio.transcription.elevenlabs import ElevenLabsStreamingTranscriber

    t = ElevenLabsStreamingTranscriber(TranscriberConfig(provider="elevenlabs", **cfg))
    captured = {}

    @asynccontextmanager
    async def fake_connect(**kwargs):
        captured.update(kwargs)
        yield conn

    t._client = _ns(speech_to_text=_ns(realtime=_ns(connect=fake_connect)))
    return t, captured


def test_elevenlabs_factory_and_validation(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    from echo.audio.transcription.elevenlabs import (
        ElevenLabsStreamingTranscriber,
        _iso_language,
        _normalise_type,
    )

    t = get_streaming_transcriber(TranscriberConfig(provider="elevenlabs"))
    assert isinstance(t, ElevenLabsStreamingTranscriber)
    with pytest.raises(ValueError, match="sample rates"):
        ElevenLabsStreamingTranscriber(TranscriberConfig(provider="elevenlabs", sample_rate=12345))
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    with pytest.raises(ValueError, match="ELEVENLABS_API_KEY"):
        ElevenLabsStreamingTranscriber(TranscriberConfig(provider="elevenlabs"))

    assert _iso_language("hi-IN") == "hi"
    assert _iso_language("en_US") == "en"
    assert _iso_language("auto") is None
    assert _iso_language(None) is None
    assert _normalise_type("partialTranscript") == "partial_transcript"
    assert _normalise_type("committed_transcript") == "committed_transcript"
    assert _normalise_type(_ns(value="SessionStarted")) == "session_started"


async def test_elevenlabs_connect_kwargs_send_and_flush(monkeypatch):
    conn = FakeElevenLabsConnection([])
    t, captured = _elevenlabs_transcriber(
        monkeypatch,
        conn,
        language="hi-IN",
        keyterms=["Metformin"],
        vad_silence_threshold_s=1.2,
        commit_strategy="manual",
    )

    async with t.stream() as session:
        await session.send_audio(b"\x01\x02")
        await session.flush()
        events = await _collect(session)

    assert captured == {
        "model_id": "scribe_v2_realtime",
        "audio_format": "pcm_16000",
        "commit_strategy": "manual",
        "language_code": "hi",
        "vad_silence_threshold_secs": 1.2,
        "keyterms": ["Metformin"],
    }
    assert conn.sent == [{"audio_base_64": base64.b64encode(b"\x01\x02").decode(), "sample_rate": 16000}]
    assert conn.commits == 1
    assert conn.closed is True
    assert [e.type for e in events] == [StreamingEventType.DONE]


async def test_elevenlabs_event_mapping(monkeypatch):
    messages = [
        {"message_type": "session_started", "session_id": "s1", "config": {"model_id": "scribe_v2_realtime"}},
        {"message_type": "partial_transcript", "text": "patient compl"},
        {"message_type": "partial_transcript", "text": ""},  # dropped
        _ns(message_type="committedTranscript", text="Patient complains of chest pain.", language_code="en"),
        {"message_type": "committed_transcript_with_timestamps", "text": "Since two days.", "words": [{"text": "Since", "start": 0.1}]},
        {"message_type": "committed_transcript_entities", "entities": []},  # ignored
        {"message_type": "rate_limited", "message": "slow down"},
    ]
    conn = FakeElevenLabsConnection(messages)
    t, _ = _elevenlabs_transcriber(monkeypatch, conn)

    async with t.stream() as session:
        events = await _collect(session)

    assert [e.type for e in events] == [
        StreamingEventType.SESSION_STARTED,
        StreamingEventType.PARTIAL,
        StreamingEventType.FINAL,
        StreamingEventType.FINAL,
        StreamingEventType.ERROR,
        StreamingEventType.DONE,
    ]
    assert events[0].details["session_id"] == "s1"
    assert events[1].text == "patient compl"
    assert events[2].text == "Patient complains of chest pain."
    assert events[2].segment_id == "1"
    assert events[2].language_detected == "en"
    assert events[3].segment_id == "2"
    assert events[3].timestamps == {"words": [{"text": "Since", "start": 0.1}]}
    assert events[4].recoverable is True


async def test_elevenlabs_partials_suppressed_and_fatal_error(monkeypatch):
    messages = [
        {"message_type": "partial_transcript", "text": "x"},
        {"message_type": "scribeAuthError", "error": "bad key"},
        {"message_type": "committed_transcript", "text": "never"},
    ]
    conn = FakeElevenLabsConnection(messages)
    t, _ = _elevenlabs_transcriber(monkeypatch, conn, interim_results=False)

    async with t.stream() as session:
        events = await _collect(session)

    assert [e.type for e in events] == [StreamingEventType.ERROR]
    assert events[0].recoverable is False
    assert "scribe_auth_error" in events[0].error
