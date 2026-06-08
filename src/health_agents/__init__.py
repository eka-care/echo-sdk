from .agents import (
    DocAssistAgent,
    TranscriptToClinicalNotesAgent,
    TranscriptToClinicalNotesStreamingAgent,
    build_clinical_context,
)
from .prompts import (
    HealthPrompts,
    UserPrompt,
    build_agent_config,
    load_system_prompt,
    load_user_prompt,
)
from .tools import (
    EKA_CLINICAL_MCP_URL,
    EKA_EMR_MCP_URL,
    EkaClinicalMcpClient,
    EkaEmrMcpClient,
    EkaMcpClient,
    get_eka_clinical_tools,
    get_eka_emr_tools,
)

__all__ = [
    # agents
    "TranscriptToClinicalNotesAgent",
    "TranscriptToClinicalNotesStreamingAgent",
    "DocAssistAgent",
    "build_clinical_context",
    # prompts
    "HealthPrompts",
    "UserPrompt",
    "build_agent_config",
    "load_system_prompt",
    "load_user_prompt",
    # mcp tools
    "EkaMcpClient",
    "EkaClinicalMcpClient",
    "EkaEmrMcpClient",
    "get_eka_clinical_tools",
    "get_eka_emr_tools",
    "EKA_CLINICAL_MCP_URL",
    "EKA_EMR_MCP_URL",
]

# AG-UI agent — optional (`ag_ui` dependency).
try:  # pragma: no cover - import guard
    from .agents import TranscriptToClinicalNotesAgUiAgent

    __all__.append("TranscriptToClinicalNotesAgUiAgent")
except ImportError:  # pragma: no cover
    pass
