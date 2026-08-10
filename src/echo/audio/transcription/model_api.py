"""Self-hosted MODEL API transcription provider (ekascribe model).

The model is served behind an OpenAI-compatible **chat completions** API and
takes the audio as base64 ``input_audio`` content alongside a transcription
prompt:

    POST {base_url}/chat/completions
    {"model": ..., "messages": [{"role": "user", "content": [
        {"type": "text", "text": "<transcription prompt>"},
        {"type": "input_audio", "input_audio": {"data": "<b64>", "format": "wav"}}
    ]}], "temperature": 0.0, "top_p": 1.0, "max_completion_tokens": ...}

Config (env fallbacks):
- base_url:  TranscriberConfig.base_url or ``MODEL_API_BASE_URL`` (OpenAI-style
  base, e.g. ``http://ekascribe.orbi.orbi/v1``; legacy
  ``MODEL_API_TRANSCRIBE_URL`` is also honored)
- model:     TranscriberConfig.model / ``MODEL_API_STT_MODEL``
- api_key:   TranscriberConfig.api_key or ``MODEL_API_AUTH_TOKEN`` ("EMPTY"
  when unset — vLLM-style servers accept any placeholder)
- prompt:    caller-supplied > ``MODEL_API_TRANSCRIBE_PROMPT`` > built-in
  default (verbatim, any Indian language)
- max tokens: ``MODEL_API_MAX_TOKENS`` (default 1024 — the transcript is cut
  off at this many tokens, so keep it comfortably above chunk length)

The wire has no language parameter; the ``language`` argument is accepted and
echoed back as ``language_detected`` but not sent.
"""

import base64
import logging
import os
from typing import Any, Optional, Tuple

import orjson

from .base import BaseTranscriber
from .config import TranscriberConfig
from .schemas import AudioInput, TranscriptionResponse

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_PATH = "/chat/completions"

DEFAULT_PROMPT = (
    "You are transcriptionist. Transcribe the audio given to you in "
    "verbatim manner. It can be in any language in India.\n\n"
    "<|audio_bos|><audio><|audio_eos|> Transcribe this audio."
)

_MIME_FORMAT = {
    "audio/m4a": "m4a",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/mp3": "mp3",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/aac": "aac",
}


class ModelApiTranscriber(BaseTranscriber):
    """base64 input_audio over OpenAI chat completions wire."""

    def __init__(self, config: TranscriberConfig):
        super().__init__(config)
        self.base_url = (
            config.base_url
            or os.getenv("MODEL_API_BASE_URL")
            or os.getenv("MODEL_API_TRANSCRIBE_URL")
            or ""
        ).rstrip("/")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(timeout=self.config.request_timeout_s)
        return self._client

    async def transcribe(
        self,
        audio: AudioInput,
        prompt: Optional[str] = None,
        mime_type: Optional[str] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> TranscriptionResponse:
        try:
            if not self.base_url:
                raise ValueError(
                    "model_api transcriber needs the OpenAI-compatible base "
                    "URL — pass TranscriberConfig(base_url=...) or set "
                    "MODEL_API_BASE_URL (e.g. http://model-host/v1)."
                )
            audio_bytes, resolved_mime = self._resolve_audio(audio, mime_type)
            audio_format = _MIME_FORMAT.get((resolved_mime or "").lower(), "wav")
            text_prompt = (
                prompt or os.getenv("MODEL_API_TRANSCRIBE_PROMPT") or DEFAULT_PROMPT
            )

            body = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": text_prompt},
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64.b64encode(audio_bytes).decode(
                                        "utf-8"
                                    ),
                                    "format": audio_format,
                                },
                            },
                        ],
                    }
                ],
                "temperature": self.config.temperature,
                "top_p": 1.0,
                "max_completion_tokens": int(
                    os.getenv("MODEL_API_MAX_TOKENS", "1024")
                ),
            }
            api_key = (
                self.config.api_key or os.getenv("MODEL_API_AUTH_TOKEN") or "EMPTY"
            )
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }

            url = f"{self.base_url}{CHAT_COMPLETIONS_PATH}"
            response = await self.client.post(
                url, content=orjson.dumps(body), headers=headers
            )
            return self._parse_response(response, language)

        except Exception as e:
            logger.error("ModelApiTranscriber transcribe error: %s", e, exc_info=True)
            return TranscriptionResponse(text="", error=str(e))

    def _resolve_audio(
        self, audio: AudioInput, mime_type: Optional[str]
    ) -> Tuple[bytes, Optional[str]]:
        if isinstance(audio, bytes):
            return audio, mime_type
        raise TypeError(
            "model_api transcriber accepts raw audio bytes only; "
            f"got {type(audio).__name__}."
        )

    def _parse_response(
        self, response, language: Optional[str]
    ) -> TranscriptionResponse:
        if response.status_code != 200:
            body_preview = (response.text or "")[:300]
            return TranscriptionResponse(
                text="",
                error=f"model API HTTP {response.status_code}: {body_preview}",
            )
        try:
            data = orjson.loads(response.content)
        except Exception:
            return TranscriptionResponse(
                text="",
                error=(
                    "model API returned non-JSON chat completion: "
                    f"{(response.text or '')[:200]}"
                ),
            )
        choices = data.get("choices") or []
        if not choices:
            return TranscriptionResponse(
                text="",
                error=f"model API returned no choices: {str(data)[:200]}",
            )
        content = (choices[0].get("message") or {}).get("content") or ""
        return TranscriptionResponse(
            text=content.strip(), language_detected=language
        )
