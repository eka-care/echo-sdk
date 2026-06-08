from typing import Any, Dict, List

from echo.ag_ui import AgUiState

# Re-export the payload-layer symbols so callers can `from .state import …`.
from .payloads import (  # noqa: F401
    KIND_TO_PAYLOAD,
    KeyValueItem,
    KeyValuePayload,
    ListPayload,
    NarrativePayload,
    Section,
    SectionKind,
    SectionStatus,
    TableColumn,
    TablePayload,
    validate_section_payload,
)


class DocumentState(AgUiState):
    sections: List[Section] = []
    metadata: Dict[str, Any] = {}


class ChatState(AgUiState):
    document_markdown: str = ""
