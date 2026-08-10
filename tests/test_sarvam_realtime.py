"""Tests for SarvamRealtimeClient. No network calls.

The client is a thin wrapper over sarvamai's speech_to_text_realtime_streaming,
so these assert on the two things that are ours: the session parameters handed
to connect(), and the framing of what goes up and comes down.
"""

import base64
import inspect

import pytest

from echo.audio.transcription.config import (
    SARVAM_REALTIME_LANGUAGES,
    SARVAM_REALTIME_MODEL,
    SARVAM_STREAM_TYPES,
)
from echo.audio.transcription.sarvam import SarvamRealtimeClient


def _client(**overrides) -> SarvamRealtimeClient:
    kwargs = {"api_key": "test-key", "language_code": "hi-IN"}
    kwargs.update(overrides)
    return SarvamRealtimeClient(**kwargs)


# ------------------------------------------------------------- construction


def test_requires_api_key():
    with pytest.raises(ValueError, match="API key"):
        SarvamRealtimeClient(api_key="")


def test_rejects_unsupported_language():
    with pytest.raises(ValueError, match="not supported"):
        _client(language_code="fr-FR")


def test_rejects_unknown_which_realtime_does_not_accept():
    """The older /speech-to-text/ws socket spelled auto-detect "unknown";
    this endpoint spells it "auto" and rejects the old word."""
    with pytest.raises(ValueError, match="not supported"):
        _client(language_code="unknown")


def test_auto_is_the_default_and_is_accepted():
    assert SarvamRealtimeClient(api_key="k").params["language_code"] == "auto"
    assert "auto" in SARVAM_REALTIME_LANGUAGES
    # Odia is or-IN here, not the od-IN the older socket used.
    assert "or-IN" in SARVAM_REALTIME_LANGUAGES
    assert "od-IN" not in SARVAM_REALTIME_LANGUAGES


def test_rejects_unsupported_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        _client(sample_rate=44100)


def test_rejects_unsupported_stream_type():
    with pytest.raises(ValueError, match="stream_type"):
        _client(stream_type="turbo")


# ------------------------------------------------------------ session params


def test_defaults_match_sarvams_published_example():
    params = _client().params
    assert params["model"] == SARVAM_REALTIME_MODEL == "saaras:v3-realtime"
    assert params["stream_type"] == "balanced"
    assert params["mode"] == "transcribe"
    # linear16 == raw PCM16, so no WAV container is needed on this endpoint.
    assert params["encoding"] == "linear16"
    assert params["sample_rate"] == "16000"


def test_unset_knobs_are_omitted_not_blanked():
    """An unset knob must not be sent at all, so Sarvam applies its own default."""
    params = _client().params
    for absent in (
        "prompt",
        "endpointing",
        "threshold",
        "prefix_padding_ms",
        "silence_duration_ms",
        "min_speech_duration_ms",
        "return_timestamps",
    ):
        assert absent not in params


def test_vad_knobs_are_forwarded_as_strings():
    params = _client(
        endpointing="vad",
        threshold=0.4,
        prefix_padding_ms=300,
        silence_duration_ms=400,
        min_speech_duration_ms=250,
        return_timestamps=True,
    ).params
    assert params["endpointing"] == "vad"
    assert params["threshold"] == "0.4"
    assert params["prefix_padding_ms"] == "300"
    assert params["silence_duration_ms"] == "400"
    assert params["min_speech_duration_ms"] == "250"
    assert params["return_timestamps"] == "true"


def test_every_param_binds_to_the_sdk_signature():
    """Guards against a rename in sarvamai silently breaking every session."""
    from sarvamai import AsyncSarvamAI

    connect = AsyncSarvamAI(
        api_subscription_key="k"
    ).speech_to_text_realtime_streaming.connect
    params = _client(
        prompt="medical dictation",
        endpointing="vad",
        threshold=0.4,
        return_timestamps=True,
    ).params
    inspect.signature(connect).bind(api_subscription_key="k", **params)


def test_stream_types_constant_matches_sdk():
    from sarvamai.speech_to_text_realtime_streaming.types import (
        SpeechToTextRealtimeStreamingStreamType,
    )

    import typing

    allowed = set(typing.get_args(typing.get_args(SpeechToTextRealtimeStreamingStreamType)[0]))
    assert set(SARVAM_STREAM_TYPES) == allowed


# ------------------------------------------------------------------ framing


class FakeSocket:
    """Captures what the client sends and replays canned events.

    Every send calls ``.dict()`` on what it is handed, exactly as the real
    ``AsyncSpeechToTextRealtimeStreamingSocketClient._send_model`` does. That is
    deliberate: a fake that accepted plain dicts would pass while the real
    client raised ``AttributeError: 'dict' object has no attribute 'dict'``.
    """

    def __init__(self, events=()):
        self.audio: list[dict] = []
        self.flushes: list[dict] = []
        self.ends: list[dict] = []
        self._events = list(events)

    async def send_realtime_audio_input(self, message):
        self.audio.append(message.dict())

    async def send_realtime_flush(self, message):
        self.flushes.append(message.dict())

    async def send_realtime_end(self, message):
        self.ends.append(message.dict())

    async def __aiter__(self):
        for e in self._events:
            yield e


def _connected(events=()) -> tuple[SarvamRealtimeClient, FakeSocket]:
    client = _client()
    sock = FakeSocket(events)
    client._sock = sock
    return client, sock


@pytest.mark.asyncio
async def test_send_audio_base64_encodes_raw_pcm():
    client, sock = _connected()
    await client.send_audio(b"\x00\x01\x02\x03")

    assert len(sock.audio) == 1
    msg = sock.audio[0]
    assert msg["event"] == "audio_input"
    # Raw PCM goes up as-is — no WAV header is prepended on this endpoint.
    assert base64.b64decode(msg["audio"]) == b"\x00\x01\x02\x03"


@pytest.mark.asyncio
async def test_flush_and_end_use_the_documented_envelopes():
    client, sock = _connected()
    await client.flush()
    await client.end()
    assert sock.flushes == [{"event": "flush"}]
    assert sock.ends == [{"event": "end"}]


@pytest.mark.asyncio
async def test_sends_pydantic_models_not_dicts():
    """The SDK calls .dict() on outgoing messages, so dicts blow up at runtime.

    Regression guard: this exact bug shipped once and only surfaced against the
    live socket, because the fake used to accept anything.
    """
    from sarvamai.types.realtime_audio_input import RealtimeAudioInput
    from sarvamai.types.realtime_end import RealtimeEnd
    from sarvamai.types.realtime_flush import RealtimeFlush

    sent: list[object] = []

    class TypeCapturingSocket:
        async def send_realtime_audio_input(self, message):
            sent.append(message)

        async def send_realtime_flush(self, message):
            sent.append(message)

        async def send_realtime_end(self, message):
            sent.append(message)

    client = _client()
    client._sock = TypeCapturingSocket()
    await client.send_audio(b"\x00")
    await client.flush()
    await client.end()

    assert [type(m) for m in sent] == [RealtimeAudioInput, RealtimeFlush, RealtimeEnd]


@pytest.mark.asyncio
async def test_events_yields_dicts_and_preserves_partials():
    frames = [
        {"event": "vad.speech_start", "confidence": 0.9},
        {"event": "transcript.partial", "text": "nam"},
        {"event": "transcript.final", "text": "namaste", "language": "hi-IN"},
    ]
    client, _ = _connected(frames)

    got = [e async for e in client.events()]

    assert [e["event"] for e in got] == [
        "vad.speech_start",
        "transcript.partial",
        "transcript.final",
    ]
    assert got[1]["text"] == "nam"


@pytest.mark.asyncio
async def test_events_skips_binary_frames():
    client, _ = _connected([b"\x00\x01", {"event": "transcript.final", "text": "hi"}])
    got = [e async for e in client.events()]
    assert len(got) == 1
    assert got[0]["text"] == "hi"


@pytest.mark.asyncio
async def test_events_normalizes_pydantic_models_to_dicts():
    class Model:
        def model_dump(self):
            return {"event": "transcript.final", "text": "from model"}

    client, _ = _connected([Model()])
    got = [e async for e in client.events()]
    assert got == [{"event": "transcript.final", "text": "from model"}]


@pytest.mark.asyncio
async def test_close_is_idempotent_when_never_connected():
    await _client().close()  # must not raise
