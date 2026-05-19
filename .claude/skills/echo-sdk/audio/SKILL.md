---
name: echo-sdk-audio
description: Audio transcription provider abstraction (Gemini, EkaCare). Use when transcribing audio, adding a new transcriber, or shaping AudioInput.
---

# Audio

## What you're working with

- `BaseTranscriber` in `audio/transcription/base.py` — abstract.
- Providers: `audio/transcription/gemini.py`, `audio/transcription/ekacare.py`.
- `get_transcriber(TranscriberConfig) -> BaseTranscriber` in `audio/transcription/factory.py`.
- `AudioInput`, `TranscriptionResponse`, `TranscriberConfig` schemas.

## Rules

- **Factory only.** `get_transcriber()` is the single entry point; optional-deps handled there.
- **`AudioInput` carries** the audio data + metadata (mime type, sample rate where relevant). Don't pass raw bytes around.
- **`TranscriptionResponse`** includes token/usage where supported — surface it to the caller for cost tracking.
- **Async.** Transcription calls are async; never block.
- **Optional deps**: `google-generativeai` for Gemini, EkaCare SDK for EkaCare. Guard imports.

## Adding a new transcriber

→ `[[echo-sdk-adding-a-provider]]`. Subclass `BaseTranscriber`, register in `factory.py`, add extra in `pyproject.toml`.

## Common mistakes

- **Reading large audio files synchronously** before calling — use `aiofiles` or `asyncio.to_thread`.
- **Hardcoding mime types** — let the provider negotiate from `AudioInput`.

## See also

- `[[python-async-discipline]]`, `[[python-optional-deps]]`
