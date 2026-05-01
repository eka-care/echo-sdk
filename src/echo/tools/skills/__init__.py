"""Skill-related tools for Echo SDK.

Currently houses the LLM-driven activation meta-tools (`LoadSkillTool`,
`UnloadSkillTool`) that `BaseAgent` auto-injects when configured with
`skill_activation="llm"`. Future skill-related tools (e.g., a tool exposing
the active set to the LLM, or richer activation verbs) belong here too.
"""

from .meta_tools import LoadSkillTool, UnloadSkillTool

__all__ = ["LoadSkillTool", "UnloadSkillTool"]
