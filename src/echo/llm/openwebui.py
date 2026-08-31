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

``OPENWEBUI_DISABLE_TOOLS=true`` stops tool schemas from being sent at all —
temporary escape hatch for serving stacks launched without
``--enable-auto-tool-choice`` (vLLM 400s any request carrying tools). Agentic
flows degrade to plain-text generation while it is set.

TLS for instances behind a private/gov CA or self-signed cert:
``OPENWEBUI_CA_BUNDLE=/path/ca.pem`` verifies against that CA (recommended);
``OPENWEBUI_VERIFY_SSL=false`` disables verification (dev/testing only).
Generic ``ECHO_LLM_CA_BUNDLE`` / ``ECHO_LLM_VERIFY_SSL`` also work here and on
the openai_compatible provider.
"""

from __future__ import annotations

import json
import logging
import os

from .config import LLMConfig
from .openai_compatible import OpenAICompatibleLLM, build_custom_http_client

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
            client_kwargs = {"api_key": api_key, "base_url": self.base_url}
            http_client = build_custom_http_client(
                ("OPENWEBUI_VERIFY_SSL", "ECHO_LLM_VERIFY_SSL"),
                ("OPENWEBUI_CA_BUNDLE", "ECHO_LLM_CA_BUNDLE"),
            )
            if http_client is not None:
                client_kwargs["http_client"] = http_client
            self._client = OpenAI(**client_kwargs)
        return self._client

    def _tool_mode(self) -> str:
        """OPENWEBUI_TOOL_MODE: native (default) | prompted | off.
        Back-compat: OPENWEBUI_DISABLE_TOOLS=true means off when unset."""
        mode = (os.getenv("OPENWEBUI_TOOL_MODE") or "").strip().lower()
        if mode in ("native", "prompted", "off"):
            return mode
        raw = os.getenv("OPENWEBUI_DISABLE_TOOLS", "")
        if raw.strip() and _parse_bool(raw):
            return "off"
        return "native"

    def _tools_enabled(self) -> bool:
        mode = self._tool_mode()
        if mode == "off":
            logger.warning(
                "OpenWebUI tool mode is OFF — tool schemas are NOT sent and "
                "not emulated; agentic flows degrade to plain text. Use "
                "OPENWEBUI_TOOL_MODE=prompted for tool calling on serving "
                "stacks without native support."
            )
        return mode == "native"

    def _prompted_tools(self) -> bool:
        return self._tool_mode() == "prompted"

    def _augment_system_prompt(self, system_prompt, tools):
        from .prompted_tools import render_tool_protocol

        return (system_prompt or "") + render_tool_protocol(tools)

    def _parse_response(self, response, msg_id):
        msg = super()._parse_response(response, msg_id)
        if self._tool_mode() != "prompted":
            return msg
        from echo.models.user_conversation import TextMessage, ToolCall

        from .prompted_tools import parse_prompted_tool_calls

        new_content = []
        for item in msg.content:
            if isinstance(item, TextMessage):
                prose, calls = parse_prompted_tool_calls(item.text)
                if prose:
                    new_content.append(TextMessage(text=prose))
                for call in calls:
                    new_content.append(
                        ToolCall(
                            tool_id=call["id"],
                            tool_name=call["name"],
                            tool_input=call["arguments"],
                        )
                    )
            else:
                new_content.append(item)
        if new_content:
            msg.content = new_content
        return msg

    def _context_messages(self, context) -> list:
        if self._tool_mode() != "prompted":
            return context.to_openai_messages()
        wire = []
        for msg in context.messages:
            wire.extend(self._messages_for_wire(msg))
        return wire

    def _messages_for_wire(self, msg) -> list:
        if self._tool_mode() != "prompted":
            return msg.to_openai_messages()
        from echo.models.user_conversation import (
            MessageRole,
            TextMessage,
            ToolCall,
            ToolResult,
        )

        from .prompted_tools import tool_calls_as_text, tool_result_as_text

        if msg.role == MessageRole.ASSISTANT:
            calls = [i for i in msg.content if isinstance(i, ToolCall)]
            if calls:
                text = "".join(
                    i.text for i in msg.content if isinstance(i, TextMessage)
                )
                return [
                    {"role": "assistant", "content": tool_calls_as_text(text, calls)}
                ]
            return msg.to_openai_messages()
        if msg.role == MessageRole.TOOL:
            parts = [
                tool_result_as_text(item.tool_id, item.result)
                for item in msg.content
                if isinstance(item, ToolResult)
            ]
            if parts:
                return [{"role": "user", "content": "\n".join(parts)}]
        return msg.to_openai_messages()

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
