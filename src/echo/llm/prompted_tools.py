"""Prompted tool-calling: for serving stacks with NO native tool support.

Tool schemas travel in the system prompt and the model emits calls as
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` text — the
hermes-style format GLM/Qwen-class models are trained on. This module renders
the protocol block, parses calls back out of replies, and serializes tool
traffic as plain text (such servers also reject ``role:"tool"`` messages and
``tool_calls`` wire fields).
"""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List, Tuple

import orjson

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")

PROTOCOL_HEADER = """

# Tool calling protocol

You have the tools listed below. This API does not accept native tool calls —
instead, call a tool by writing EXACTLY this in your reply:

<tool_call>{"name": "<tool_name>", "arguments": { ... }}</tool_call>

Rules:
- "arguments" MUST be a single JSON object matching the tool's input schema.
- You may write several <tool_call> blocks in one reply.
- Text outside <tool_call> blocks is shown to the user; keep it minimal.
- Each call's result comes back in a <tool_result> block on the next turn. If
  a result reports a validation error, re-emit that call with a corrected
  payload.

## Available tools
"""


def render_tool_protocol(tools) -> str:
    """The system-prompt block: protocol rules + every tool's schema."""
    parts = [PROTOCOL_HEADER]
    for tool in tools:
        fn = tool.to_openai_schema().get("function", {})
        params = orjson.dumps(fn.get("parameters", {})).decode()
        parts.append(
            f"### {fn.get('name', tool.name)}\n"
            f"{fn.get('description', '')}\n"
            f"Input schema: {params}\n"
        )
    return "\n".join(parts)


def parse_prompted_tool_calls(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Split a reply into (prose, calls).

    Lenient: tolerates ```json fences inside the tags and stringified
    "arguments". Malformed blocks stay in the prose so failures are visible
    instead of silently dropped.
    """
    calls: List[Dict[str, Any]] = []
    prose_parts: List[str] = []
    cursor = 0
    for m in TOOL_CALL_RE.finditer(text):
        prose_parts.append(text[cursor : m.start()])
        cursor = m.end()
        raw = _JSON_FENCE_RE.sub("", m.group(1).strip()).strip()
        try:
            data = orjson.loads(raw)
        except Exception:
            prose_parts.append(m.group(0))
            continue
        if not isinstance(data, dict) or not data.get("name"):
            prose_parts.append(m.group(0))
            continue
        args = data.get("arguments", data.get("input", {}))
        if isinstance(args, str):
            try:
                args = orjson.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append(
            {
                "id": data.get("id") or f"ptc_{uuid.uuid4().hex[:12]}",
                "name": str(data["name"]),
                "arguments": args,
            }
        )
    prose_parts.append(text[cursor:])
    return "".join(prose_parts).strip(), calls


def tool_calls_as_text(text: str, tool_calls) -> str:
    """Re-serialize an assistant message's tool calls back into protocol text
    (ids included so results correlate on later turns)."""
    parts = [text] if text else []
    for tc in tool_calls:
        parts.append(
            "<tool_call>"
            + orjson.dumps(
                {
                    "id": tc.tool_id,
                    "name": tc.tool_name,
                    "arguments": tc.tool_input,
                }
            ).decode()
            + "</tool_call>"
        )
    return "\n".join(parts)


def tool_result_as_text(tool_id: str, result: Any) -> str:
    if not isinstance(result, str):
        try:
            result = orjson.dumps(result).decode()
        except Exception:
            result = str(result)
    return f'<tool_result id="{tool_id}">\n{result}\n</tool_result>'
