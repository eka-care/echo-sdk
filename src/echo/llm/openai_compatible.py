"""OpenAI-compatible LLM provider (plan decision #15).

Points the OpenAI client at ANY OpenAI-compatible endpoint — vLLM or Ollama
serving Qwen/MedGemma locally, LiteLLM proxies, or hosted services. Strips the
OpenAI-only request parameters (max_completion_tokens heuristics,
reasoning_effort) that open-model servers reject.

Config: LLMConfig(provider="openai_compatible", base_url=..., model=...,
api_key=...) with env fallbacks ECHO_LLM_BASE_URL / ECHO_LLM_API_KEY.
"""

from __future__ import annotations

import logging
import os
import ssl

from .config import LLMConfig
from .openai import OpenAILLM

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT_S = 600.0


def resolve_ssl_verify(verify_vars, ca_vars):
    """Resolve TLS verification for self-hosted endpoints.

    Self-hosted / on-prem endpoints often sit behind a private CA (or a plain
    self-signed cert). Returns one of:

    - ``False`` — verification disabled: the first env var in ``verify_vars``
      that is set to a falsy value (0/false/no/off) wins. Dev/testing only.
    - an ``ssl.SSLContext`` trusting the CA bundle at the first set env var in
      ``ca_vars`` (path to a PEM file) — the production-grade option.
    - ``True`` — default trust store (no env set).
    """
    for var in verify_vars:
        raw = os.getenv(var)
        if raw is not None and raw.strip():
            if raw.strip().lower() in ("0", "false", "no", "off"):
                logger.warning(
                    "TLS certificate verification DISABLED via %s — use a CA "
                    "bundle instead for anything beyond local testing.",
                    var,
                )
                return False
            break
    for var in ca_vars:
        path = os.getenv(var)
        if path:
            return ssl.create_default_context(cafile=path)
    return True


def build_custom_http_client(verify_vars, ca_vars, timeout_s=DEFAULT_HTTP_TIMEOUT_S):
    """httpx.Client honoring the TLS env vars, or None for SDK defaults."""
    verify = resolve_ssl_verify(verify_vars, ca_vars)
    if verify is True:
        return None
    import httpx

    return httpx.Client(verify=verify, timeout=timeout_s)


class OpenAICompatibleLLM(OpenAILLM):
    """OpenAI wire format against a configurable base_url."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.base_url = (
            getattr(config, "base_url", None)
            or os.getenv("ECHO_LLM_BASE_URL")
            or "http://localhost:11434/v1"  # Ollama default
        )

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            api_key = (
                self.config.api_key
                or os.getenv("ECHO_LLM_API_KEY")
                or "not-needed"  # local vLLM/Ollama don't check keys
            )
            client_kwargs = {"api_key": api_key, "base_url": self.base_url}
            http_client = build_custom_http_client(
                ("ECHO_LLM_VERIFY_SSL",), ("ECHO_LLM_CA_BUNDLE",)
            )
            if http_client is not None:
                client_kwargs["http_client"] = http_client
            self._client = OpenAI(**client_kwargs)
        return self._client

    # Open-model servers speak plain max_tokens and reject OpenAI-only params.
    def _uses_max_completion_tokens(self) -> bool:
        return False

    def _supports_reasoning_effort(self) -> bool:
        return False

    def _is_reasoning_model(self) -> bool:
        return False
