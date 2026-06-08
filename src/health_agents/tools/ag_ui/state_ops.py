from typing import Optional

from .payloads import Section
from .state import DocumentState


def find_section_index(state: DocumentState, key: str) -> Optional[int]:
    for i, s in enumerate(state.sections):
        if s.key == key:
            return i
    return None


def apply_section_to_state(state: DocumentState, section: Section) -> None:
    if section.status.state == "pending":
        section.status.state = "ready"

    idx = find_section_index(state, section.key)
    if idx is None:
        state.sections.append(section)
    else:
        state.sections[idx] = section
