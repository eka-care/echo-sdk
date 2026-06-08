import asyncio
import uuid

import orjson
from dotenv import load_dotenv

load_dotenv()

from ag_ui.core import EventType, RunAgentInput  # noqa: E402

from echo.ag_ui import AgUiAgent  # noqa: E402
from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig  # noqa: E402
from health_agents.tools.ag_ui import DocumentState, build_section_tools  # noqa: E402
from echo.models.user_conversation import (  # noqa: E402
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)

from _harness import banner, describe_config, example_llm_config, rule, short  # noqa: E402

# Arbitrary, non-clinical input to structure.
RAW_NOTES = """
Project Atlas kickoff notes. Goal: ship a self-serve analytics dashboard by Q3.
Team: Priya (lead), Marco (backend), Lena (design). Stack is Postgres + FastAPI
+ React. Risks: data volume could blow past current warehouse limits; the
auth migration is on the critical path. Next steps: Marco to spike the query
layer this week, Lena to deliver wireframes by Friday, Priya to confirm scope
with the customer council. Decision: we will NOT build mobile in v1.
""".strip()

_SYSTEM = AgentConfig(
    persona=PersonaConfig(
        role="Document structurer",
        goal="Turn unstructured notes into a clean, well-organized document.",
    ),
    task=TaskConfig(
        description=(
            "Structure the user's notes into a document by calling the section "
            "tools. Do NOT write the document as free text in your reply — the "
            "only way content reaches the document is through tool calls. Pick "
            "the tool that best fits each part: add_narrative for prose, "
            "add_list for enumerations, add_table for repeated records sharing "
            "columns, add_key_value for labelled fields. Give each section a "
            "slug `key`, a human `display_name`, an integer `order`, and the "
            "typed `payload`. Cover all the content; do not invent facts."
        ),
        expected_output="A sequence of section tool calls covering the notes.",
    ),
)


def render_event(event) -> str:
    etype = event.type
    if etype == EventType.TEXT_MESSAGE_CHUNK:
        return f"TEXT_CHUNK  {short(event.delta, 80)!r}"
    if etype == EventType.TOOL_CALL_START:
        return f"TOOL_START  {event.tool_call_name} (id={event.tool_call_id})"
    if etype == EventType.TOOL_CALL_ARGS:
        return f"TOOL_ARGS   {short(event.delta, 100)}"
    if etype == EventType.TOOL_CALL_END:
        return f"TOOL_END    id={event.tool_call_id}"
    if etype == EventType.STATE_SNAPSHOT:
        return f"STATE_SNAPSHOT  sections={len(event.snapshot.get('sections', []))}"
    if etype == EventType.STATE_DELTA:
        return f"STATE_DELTA  ops={len(event.delta)}"
    if etype == EventType.RUN_STARTED:
        return "RUN_STARTED"
    if etype == EventType.RUN_FINISHED:
        return "RUN_FINISHED"
    if etype == EventType.RUN_ERROR:
        return f"RUN_ERROR   {event.message}"
    return str(etype)


async def main() -> None:
    banner("Generic AG-UI structured-document building (non-clinical)")
    llm_config = example_llm_config()
    describe_config(llm_config)

    agent = AgUiAgent(
        agent_config=_SYSTEM,
        llm_config=llm_config,
        tools=build_section_tools(),
    )

    state = DocumentState()
    context = ConversationContext()
    context.system_context["tool_context"] = {"document_state": state}
    context.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text=RAW_NOTES)])
    )
    run_input = RunAgentInput(
        thread_id=str(uuid.uuid4()),
        run_id=str(uuid.uuid4()),
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={},
    )

    rule("input notes")
    print(RAW_NOTES)
    print()

    rule("AG-UI event stream")
    async for event in agent.ag_ui_stream(
        context=context,
        run_input=run_input,
        state=state,
        out_msg_id=str(uuid.uuid4()),
    ):
        print(render_event(event))

    print()
    rule("final DocumentState sections")
    for section in state.sections:
        print(f"[{section.order}] {section.display_name}  ({section.kind.value})")
        print(orjson.dumps(section.payload, option=orjson.OPT_INDENT_2).decode())
        print()


if __name__ == "__main__":
    asyncio.run(main())
