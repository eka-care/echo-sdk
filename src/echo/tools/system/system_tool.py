"""SystemTool — base type for loop/agent-managed tools (echo-internal only).

A `SystemTool` is a tool whose execution can change the agent's *loaded
state* (active skills, conversation history, ...) and therefore may direct
the loop to interrupt-and-recompute. Skill load/unload and (future) summary
or computer-use tools are system tools.

These are NOT for external extension. The `__init_subclass__` guard rejects
any subclass defined outside the `echo` package, which makes the invariant
"only our tools can interrupt the loop" enforced by construction rather than
by convention: external/user/MCP tools structurally cannot produce an
INTERRUPT directive because they cannot become a `SystemTool`.

Behavioral directives (control-flow, observability) are adopted in a later
step; for now this defines the type and the extension boundary.
"""

from echo.tools.base_tool import BaseTool


class SystemTool(BaseTool):
    """Base class for echo-internal, loop-managed tools.

    Subclassing is restricted to modules inside the `echo` package.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        module = cls.__module__ or ""
        if not (module == "echo" or module.startswith("echo.")):
            raise TypeError(
                f"{cls.__name__!r} cannot subclass SystemTool: SystemTool may "
                f"only be extended inside the echo-sdk package (got module "
                f"{module!r}). System tools are loop-managed and not part of the "
                f"public extension surface."
            )
