"""
Example: real-time speech-to-text over a provider websocket.

This demonstrates:
- get_streaming_transcriber() for Sarvam (saaras:v3) and ElevenLabs (Scribe v2)
- Feeding raw 16 kHz mono PCM in 100 ms frames, paced like a live microphone
- Consuming StreamingTranscriptEvent (PARTIAL / FINAL / ERROR / DONE)

Usage:
    uv run python examples/streaming_transcription_usage.py path/to/audio.wav [sarvam|elevenlabs] [lang]

The WAV must be 16-bit PCM mono at 16 kHz (convert with
`ffmpeg -i in.m4a -ac 1 -ar 16000 -sample_fmt s16 out.wav`).
Needs SARVAM_API_KEY or ELEVENLABS_API_KEY in the environment / .env.

This doubles as the live smoke test for the streaming providers.
"""

import asyncio
import sys
import time
import wave

from dotenv import load_dotenv

load_dotenv()

from echo.audio import (  # noqa: E402
    StreamingEventType,
    TranscriberConfig,
    get_streaming_transcriber,
)

FRAME_MS = 100


def read_pcm16k(path: str) -> bytes:
    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, "mono WAV required"
        assert wf.getsampwidth() == 2, "16-bit PCM WAV required"
        assert wf.getframerate() == 16000, "16 kHz WAV required"
        return wf.readframes(wf.getnframes())


async def main(path: str, provider: str, language: str | None) -> None:
    pcm = read_pcm16k(path)
    frame_bytes = 16000 * 2 * FRAME_MS // 1000
    cfg = TranscriberConfig(provider=provider, language=language, sample_rate=16000)
    transcriber = get_streaming_transcriber(cfg)

    finals: list[str] = []
    started = time.monotonic()

    async with transcriber.stream() as session:

        async def pump_audio() -> None:
            for i in range(0, len(pcm), frame_bytes):
                await session.send_audio(pcm[i : i + frame_bytes])
                await asyncio.sleep(FRAME_MS / 1000)  # real-time pacing
            await session.flush()
            # give the provider a moment to emit trailing finals, then hang up
            await asyncio.sleep(3)
            await session.close()

        async def consume() -> None:
            async for ev in session:
                t = time.monotonic() - started
                if ev.type is StreamingEventType.PARTIAL:
                    print(f"[{t:6.2f}s] …  {ev.text}")
                elif ev.type is StreamingEventType.FINAL:
                    finals.append(ev.text or "")
                    print(f"[{t:6.2f}s] ✔  {ev.text}  ({ev.language_detected})")
                elif ev.type is StreamingEventType.ERROR:
                    print(f"[{t:6.2f}s] !! {ev.error} recoverable={ev.recoverable}")
                elif ev.type is StreamingEventType.DONE:
                    print(f"[{t:6.2f}s] -- done")
                else:
                    print(f"[{t:6.2f}s] {ev.type.value} {ev.details}")

        await asyncio.gather(pump_audio(), consume())

    print("\nTRANSCRIPT:\n" + " ".join(finals))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    wav = sys.argv[1]
    prov = sys.argv[2] if len(sys.argv) > 2 else "sarvam"
    lang = sys.argv[3] if len(sys.argv) > 3 else None
    asyncio.run(main(wav, prov, lang))
