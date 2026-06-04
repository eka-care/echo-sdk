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

A system tool declares its directive by overriding the inherited class
attributes (`control_flow`, `observability`); `BaseLLM.invoke_tool` honors
`control_flow = INTERRUPT` ONLY from `SystemTool` instances.
"""

from echo.tools.core import BaseTool


class SystemTool(BaseTool):
    """Base class for echo-internal, loop-managed tools.

    Subclassing is restricted to modules inside the `echo` package.

    `SystemTool` is the ONLY tool type whose ``control_flow = INTERRUPT`` is
    honored by `BaseLLM.invoke_tool` (the unfakeable marker, thanks to the
    `__init_subclass__` guard). Concrete system tools declare their directive
    by overriding the inherited class attributes, e.g.::

        class LoadSkillTool(SystemTool):
            control_flow = ControlFlow.INTERRUPT   # changes loaded state → recompute
            observability = Observability.SILENT   # user need not see it

    Defaults are inherited from `BaseTool` (CONTINUE/VISIBLE), so a system tool
    that doesn't change loaded state behaves like a normal tool.
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
