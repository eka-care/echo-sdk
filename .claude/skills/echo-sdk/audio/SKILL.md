---
name: echo-sdk-audio
description: Audio transcription provider abstraction — batch (Gemini, EkaCare, Sarvam) and real-time streaming (Sarvam, ElevenLabs). Use when transcribing audio, streaming mic audio to text, adding a new transcriber, or shaping AudioInput.
---

# Audio

## What you're working with

- `BaseTranscriber` in `audio/transcription/base.py` — abstract, batch (`transcribe()`).
- `BaseStreamingTranscriber` + `StreamingTranscriptionSession` in the same file — abstract, real-time.
  `stream()` is an async context manager yielding a session with `send_audio(pcm)`, `flush()`,
  `close()` and `events()` (also `async for ev in session`).
- Providers: `gemini.py`, `ekacare.py` (batch), `sarvam.py` (batch **and** streaming),
  `elevenlabs.py` (streaming only).
- Factories in `factory.py`: `get_transcriber(TranscriberConfig)` and
  `get_streaming_transcriber(TranscriberConfig)`.
- Schemas: `AudioInput`, `TranscriptionResponse` (batch); `StreamingTranscriptEvent`,
  `StreamingEventType` (`SESSION_STARTED | PARTIAL | FINAL | SPEECH_STARTED | SPEECH_ENDED | ERROR | DONE`).
- `TranscriberConfig` carries both batch and streaming fields (`sample_rate`, `encoding`,
  `commit_strategy`, `high_vad_sensitivity`, `interim_results`, `keyterms`, `mode`).

## Rules

- **Factory only.** `get_transcriber()` / `get_streaming_transcriber()` are the entry points; optional-deps handled there.
- **`AudioInput` carries** the audio data + metadata (mime type, sample rate where relevant). Don't pass raw bytes around.
- **Streaming input is raw `pcm_s16le` mono at `config.sample_rate`** (16 kHz for both providers). The provider class does base64/envelope wrapping; callers resample and chunk (100–250 ms per `send_audio`).
- **`events()` ends** after `DONE` or an `ERROR` with `recoverable=False`. Recoverable errors (rate limits) are surfaced and the stream continues.
- **Sarvam has no partials** — only `FINAL` per VAD-segmented utterance. ElevenLabs emits `PARTIAL` then `FINAL`; gate with `interim_results`.
- **`TranscriptionResponse`** includes token/usage where supported — surface it to the caller for cost tracking.
- **Async.** Transcription calls are async; never block. Send and receive run concurrently on the same session.
- **Optional deps**: `google-genai` (Gemini), `httpx` (EkaCare), `sarvamai` (Sarvam), `elevenlabs` (ElevenLabs). Guard imports.

## Adding a new transcriber

→ `[[echo-sdk-adding-a-provider]]`. Subclass `BaseTranscriber` and/or `BaseStreamingTranscriber`, register in `factory.py`, add extra in `pyproject.toml`, add mapping tests in `tests/test_streaming_transcribers.py` with a fake socket.

## Common mistakes

- **Reading large audio files synchronously** before calling — use `aiofiles` or `asyncio.to_thread`.
- **Hardcoding mime types** — let the provider negotiate from `AudioInput`.
- **Sending compressed audio to a streaming session** — providers expect PCM; decode client-side.
- **Forgetting `flush()` before closing** — the last utterance stays uncommitted.
- **Sarvam streaming model**: the batch default `saarika:v2.5` is treated as unset; streaming resolves to `SARVAM_STT_STREAM_MODEL` or `saaras:v3`.

## See also

- `[[python-async-discipline]]`, `[[python-optional-deps]]`
- `examples/streaming_transcription_usage.py`
