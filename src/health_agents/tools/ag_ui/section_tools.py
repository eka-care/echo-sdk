import logging
from typing import Any, ClassVar, Dict, Optional, Type

from pydantic import BaseModel, ValidationError

from echo.tools.base_tool import BaseTool

from .payloads import (
    KeyValuePayload,
    ListPayload,
    NarrativePayload,
    Section,
    SectionKind,
    SectionStatus,
    TablePayload,
)
from .state import DocumentState
from .state_ops import apply_section_to_state

logger = logging.getLogger(__name__)


# Shared input-schema fragments. Every add_* tool takes the same shell
# (key / display_name / order) and varies only on `payload`.
_KEY_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "pattern": r"^[a-z][a-z0-9_]*$",
    "description": (
        "Stable section identifier; slug(display_name) — lowercase "
        "letters, digits, and underscores only."
    ),
}

_DISPLAY_NAME_SCHEMA: Dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "description": "Heading text rendered verbatim as the section title.",
}

_ORDER_SCHEMA: Dict[str, Any] = {
    "type": "integer",
    "minimum": 0,
    "description": "0-indexed render order within the document.",
}


def _build_input_schema(payload_model: Type[BaseModel]) -> Dict[str, Any]:
    return {
        "type": "object",
        "required": ["key", "display_name", "payload", "order"],
        "additionalProperties": False,
        "properties": {
            "key": _KEY_SCHEMA,
            "display_name": _DISPLAY_NAME_SCHEMA,
            "payload": payload_model.model_json_schema(),
            "order": _ORDER_SCHEMA,
        },
    }


def _resolve_document_state(tool_context: Optional[Dict[str, Any]]):
    if tool_context is None:
        return "Error: tool_context missing (server bug)."
    state = tool_context.get("document_state")
    if not isinstance(state, DocumentState):
        return (
            f"Error: tool_context['document_state'] is "
            f"{type(state).__name__}, expected DocumentState."
        )
    return state


class _GenericEmitTool(BaseTool):
    KIND: ClassVar[SectionKind]
    PAYLOAD_MODEL: ClassVar[Type[BaseModel]]

    @property
    def input_schema(self) -> Dict[str, Any]:
        return _build_input_schema(self.PAYLOAD_MODEL)

    async def run(
        self,
        key: str,
        display_name: str,
        payload: Dict[str, Any],
        order: int,
        tool_context: Optional[Dict[str, Any]] = None,
        **_unused: Any,
    ) -> str:
        state = _resolve_document_state(tool_context)
        if isinstance(state, str):
            return state

        try:
            self.PAYLOAD_MODEL.model_validate(payload)
        except ValidationError as e:
            return (
                f"Error: payload does not match {self.KIND.value} schema. "
                f"Validation errors: {e.errors()}. Re-emit with the correct shape."
            )

        try:
            section = Section(
                key=key,
                display_name=display_name,
                kind=self.KIND,
                payload=payload,
                order=order,
                status=SectionStatus(state="ready"),
            )
        except ValidationError as e:
            return f"Error: invalid Section shell. Validation errors: {e.errors()}."

        apply_section_to_state(state, section)
        logger.info(
            "%s: section %r emitted (kind=%s, order=%s, sections_count=%s)",
            self.name,
            key,
            self.KIND.value,
            order,
            len(state.sections),
        )
        return f"ok — section {key!r} emitted via {self.name}"


class ListTool(_GenericEmitTool):
    name = "add_list"
    description = (
        "Emit a bulleted-list section. Each item is one self-contained markdown string. Use when the content is a flat enumeration "
        "where order matters but per-item column structure does not — key points, steps, action items, tags, references. Inline "
        "richer per-item detail in the markdown string. Do NOT use for repeated records that share the same columns (use add_table), "
        "for labelled single fields (use add_key_value), or for paragraph-shaped content (use add_narrative)."
    )
    KIND = SectionKind.LIST
    PAYLOAD_MODEL = ListPayload


class TableTool(_GenericEmitTool):
    name = "add_table"
    description = (
        "Emit a tabular section. Define `headers` once (each with key, label, and editor type: text | markdown | number | date), then "
        "`rows` as a list of dicts keyed by header.key. STREAM headers BEFORE rows so the UI can build the table skeleton before "
        "cells arrive. Use when the content is a set of records sharing the same columns. Do NOT use for one-off labelled fields "
        "(use add_key_value), a flat list of strings (use add_list), or prose (use add_narrative). If a section has only one row and "
        "no natural repetition, prefer add_key_value."
    )
    KIND = SectionKind.TABLE
    PAYLOAD_MODEL = TablePayload


class KeyValueTool(_GenericEmitTool):
    name = "add_key_value"
    description = (
        "Emit a key-value (definition-list) section. Each item is a {key, value} pair where value is markdown. Use when the "
        "content reads as labelled scalar fields top-to-bottom — metadata, properties, attributes, summary fields. Each label should "
        "appear at most once. Do NOT use when the same set of fields repeats for multiple records (use add_table), for an ordered "
        "enumeration without labels (use add_list), or for prose (use add_narrative)."
    )
    KIND = SectionKind.KEY_VALUE
    PAYLOAD_MODEL = KeyValuePayload


class NarrativeTool(_GenericEmitTool):
    name = "add_narrative"
    description = (
        "Emit a free-form markdown section. Use when the content is naturally prose — a summary, description, explanation, or notes. "
        "One coherent markdown block; use inline bullets only when the prose genuinely calls for them. Do NOT include the section "
        "heading inside the markdown — `display_name` is rendered as the title. Do NOT use this as a fallback for structured data; if "
        "a heading mixes shapes, split it into adjacent typed sections instead of flattening everything into prose."
    )
    KIND = SectionKind.NARRATIVE
    PAYLOAD_MODEL = NarrativePayload


ALL_SECTION_TOOLS: Dict[SectionKind, Type[_GenericEmitTool]] = {
    SectionKind.LIST: ListTool,
    SectionKind.TABLE: TableTool,
    SectionKind.KEY_VALUE: KeyValueTool,
    SectionKind.NARRATIVE: NarrativeTool,
}


def build_section_tools() -> list:
    return [cls() for cls in ALL_SECTION_TOOLS.values()]
