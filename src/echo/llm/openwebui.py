"""Open WebUI LLM provider.

Open WebUI (https://github.com/open-webui/open-webui) fronts local models
(Ollama, vLLM, ...) with one OpenAI-compatible endpoint:

    POST {base}/api/chat/completions
    Authorization: Bearer <api key>

This provider is OpenAICompatibleLLM with Open WebUI's conventions baked in:
the ``/api`` path suffix is appended automatically (pass the plain instance
URL, e.g. ``http://openwebui.local:3000``) and the API key is resolved from
``OPENWEBUI_API_KEY`` (create one in Open WebUI -> Settings -> Account -> API
keys; a session JWT works too). Model IDs are whatever the instance lists at
``GET {base}/api/models`` (e.g. ``qwen3:14b``).

Config: LLMConfig(provider="openwebui", base_url=..., model=..., api_key=...)
with env fallbacks OPENWEBUI_BASE_URL / OPENWEBUI_API_KEY (then
ECHO_LLM_BASE_URL / ECHO_LLM_API_KEY).

Hybrid-reasoning open models (GLM, Qwen3, ...) served behind Open WebUI accept
``chat_template_kwargs`` (forwarded to vLLM/SGLang). Configure via env:

- ``OPENWEBUI_ENABLE_THINKING=false`` -> ``chat_template_kwargs:
  {"enable_thinking": false}`` (recommended for structuring: no <think>
  preamble, deterministic tool calls)
- ``OPENWEBUI_CHAT_TEMPLATE_KWARGS='{"enable_thinking": false}'`` -> raw JSON
  object for anything else; the boolean flag above wins on overlap.

Nothing is sent when neither is set.
"""

from __future__ import annotations

import json
import logging
import os

from .config import LLMConfig
from .openai_compatible import OpenAICompatibleLLM

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:3000"
API_SUFFIX = "/api"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _normalize_base_url(url: str) -> str:
    """``http://host:3000`` and ``http://host:3000/api`` both -> ``.../api``."""
    url = url.rstrip("/")
    if not url.endswith(API_SUFFIX):
        url += API_SUFFIX
    return url


class OpenWebUILLM(OpenAICompatibleLLM):
    """OpenAI wire format against an Open WebUI instance (.../api)."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = _normalize_base_url(
            getattr(config, "base_url", None)
            or os.getenv("OPENWEBUI_BASE_URL")
            or os.getenv("ECHO_LLM_BASE_URL")
            or DEFAULT_BASE_URL
        )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            api_key = (
                self.config.api_key
                or os.getenv("OPENWEBUI_API_KEY")
                or os.getenv("ECHO_LLM_API_KEY")
            )
            if not api_key:
                # Open WebUI normally requires a Bearer key; only deployments
                # running with auth disabled (WEBUI_AUTH=False) work without.
                logger.warning(
                    "No Open WebUI API key configured (OPENWEBUI_API_KEY / "
                    "ECHO_LLM_API_KEY / LLMConfig.api_key); requests will get "
                    "401 unless the instance runs with auth disabled."
                )
                api_key = "not-needed"
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)
        return self._client

    def _extra_body(self):
        template_kwargs = {}

        raw = os.getenv("OPENWEBUI_CHAT_TEMPLATE_KWARGS")
        if raw:
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                template_kwargs.update(parsed)
            else:
                logger.warning(
                    "OPENWEBUI_CHAT_TEMPLATE_KWARGS must be a JSON object; "
                    "ignoring %r",
                    raw,
                )

        enable_thinking = os.getenv("OPENWEBUI_ENABLE_THINKING")
        if enable_thinking is not None and enable_thinking.strip():
            template_kwargs["enable_thinking"] = _parse_bool(enable_thinking)

        if not template_kwargs:
            return None
        return {"chat_template_kwargs": template_kwargs}
