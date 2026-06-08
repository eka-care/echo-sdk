from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SectionKind(str, Enum):
    LIST = "LIST"
    TABLE = "TABLE"
    KEY_VALUE = "KEY_VALUE"
    NARRATIVE = "NARRATIVE"


class ListPayload(StrictModel):
    items: List[str] = []


ColumnType = Literal["text", "markdown", "number", "date"]


class TableColumn(StrictModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    label: str = Field(min_length=1)
    type: ColumnType = "markdown"


class TablePayload(StrictModel):
    headers: List[TableColumn] = []
    rows: List[Dict[str, str]] = []


class KeyValueItem(StrictModel):
    key: str = Field(min_length=1)
    value: str = ""


class KeyValuePayload(StrictModel):
    items: List[KeyValueItem] = []


class NarrativePayload(StrictModel):
    markdown: str = ""


class SectionStatus(BaseModel):
    state: Literal[
        "pending", "extracting", "awaiting_input", "ready", "saved", "error"
    ] = "pending"
    error: Optional[str] = None


class Section(BaseModel):
    key: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    display_name: str = Field(min_length=1)
    kind: SectionKind
    payload: Dict[str, Any] = {}
    order: int = Field(ge=0)
    status: SectionStatus = SectionStatus()
    edited_by_user: bool = False


KIND_TO_PAYLOAD: Dict[SectionKind, type[BaseModel]] = {
    SectionKind.LIST: ListPayload,
    SectionKind.TABLE: TablePayload,
    SectionKind.KEY_VALUE: KeyValuePayload,
    SectionKind.NARRATIVE: NarrativePayload,
}


def validate_section_payload(kind: SectionKind, payload: Dict[str, Any]) -> BaseModel:
    model_cls = KIND_TO_PAYLOAD.get(kind)
    if model_cls is None:
        raise ValueError(f"No payload model registered for kind: {kind!r}")
    return model_cls.model_validate(payload)
