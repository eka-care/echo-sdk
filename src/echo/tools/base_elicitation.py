"""
Base elicitation tool interface for Echo SDK.

Elicitation tools return UI component specifications that tell the frontend
what to render to collect structured user input.
"""

from abc import abstractmethod
from typing import Any, Dict, List, Optional

from .base_tool import BaseTool
from .schemas import ElicitationComponent, ElicitationDetails


class BaseElicitationTool(BaseTool):
    """
    Base class for elicitation tools.

    Elicitation tools return UI component specs that tell the frontend
    what to render to collect structured user input.
    """

    name = "elicit_base"
    description = "Use this tool to collect structured user input from the user."

    @property
    def is_elicitation(self) -> bool:
        """
        Whether the tool is an elicitation tool.
        """
        return True

    @property
    @abstractmethod
    def elicitation_components(self) -> List[ElicitationComponent]:
        """
        The type of elicitation tool.
        """
        pass

    async def run(
        self, component: str, _meta: Optional[Dict[str, Any]] = None, **kwargs
    ) -> ElicitationDetails:
        """Return JSON string of ElicitationResponse."""
        # Find the matching enum for the component string
        component_enum = None
        for c in self.elicitation_components:
            if c.value == component:
                component_enum = c
                break

        if component_enum is None:
            valid_components = [c.value for c in self.elicitation_components]
            raise ValueError(
                f"Invalid component: {component}. Valid: {valid_components}"
            )

        return ElicitationDetails(
            component=component_enum.value,
            input=kwargs,
            _meta=_meta,
        )
