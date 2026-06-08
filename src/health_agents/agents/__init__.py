from .base import HealthAgentMixin, build_clinical_context, new_message_id
from .docassist import DocAssistAgent
from .transcript_to_clinical_notes import TranscriptToClinicalNotesAgent
from .transcript_to_clinical_notes_streaming import (
    TranscriptToClinicalNotesStreamingAgent,
)

__all__ = [
    "HealthAgentMixin",
    "build_clinical_context",
    "new_message_id",
    "TranscriptToClinicalNotesAgent",
    "TranscriptToClinicalNotesStreamingAgent",
    "DocAssistAgent",
]

try:  # pragma: no cover - import guard
    from .transcript_to_clinical_notes_ag_ui import TranscriptToClinicalNotesAgUiAgent

    __all__.append("TranscriptToClinicalNotesAgUiAgent")
except ImportError:  # pragma: no cover
    pass
