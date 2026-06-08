import asyncio

import orjson
from dotenv import load_dotenv

load_dotenv()

from ag_ui.core import EventType  # noqa: E402

from health_agents.agents import (  # noqa: E402
    TranscriptToClinicalNotesAgUiAgent,
)
from health_agents.tools.ag_ui import DocumentState  # noqa: E402

from _harness import (  # noqa: E402
    DEFAULT_TEMPLATE,
    banner,
    describe_config,
    example_llm_config,
    load_transcript,
    rule,
    short,
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
    banner("Transcript -> Clinical Notes (AG-UI events)")
    llm_config = example_llm_config()
    describe_config(llm_config)

    transcript = load_transcript()
    state = DocumentState()
    agent = TranscriptToClinicalNotesAgUiAgent(
        user_prompt=DEFAULT_TEMPLATE,
        llm_config=llm_config,
    )

    rule("transcript")
    print(transcript)
    print()

    rule("AG-UI event stream")
    async for event in agent.stream(transcript, document_state=state):
        print(render_event(event))

    print()
    rule("final DocumentState sections")
    for section in state.sections:
        print(f"[{section.order}] {section.display_name}  ({section.kind.value})")
        print(orjson.dumps(section.payload, option=orjson.OPT_INDENT_2).decode())
        print()


if __name__ == "__main__":
    asyncio.run(main())
