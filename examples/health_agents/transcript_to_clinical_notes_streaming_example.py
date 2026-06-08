import asyncio

from dotenv import load_dotenv

load_dotenv()

from health_agents import TranscriptToClinicalNotesStreamingAgent  # noqa: E402
from echo.llm.schemas import StreamEventType  # noqa: E402

from _harness import (  # noqa: E402
    DEFAULT_TEMPLATE,
    banner,
    describe_config,
    example_llm_config,
    load_transcript,
    rule,
)


async def main() -> None:
    banner("Transcript -> Clinical Notes (streaming)")
    llm_config = example_llm_config()
    describe_config(llm_config)

    agent = TranscriptToClinicalNotesStreamingAgent(
        user_prompt=DEFAULT_TEMPLATE,
        llm_config=llm_config,
    )

    transcript = load_transcript()
    rule("transcript")
    print(transcript)
    print()

    rule("clinical note (streaming)")
    final_response = None
    async for event in agent.generate_stream(transcript):
        if event.type == StreamEventType.TEXT:
            print(event.text, end="", flush=True)
        elif event.type == StreamEventType.DONE:
            print()
            final_response = event.llm_response
        elif event.type == StreamEventType.ERROR:
            print(f"\n[error] {event.error}")
            return

    print()
    rule("done")
    if final_response:
        print(f"verbose items: {len(final_response.verbose)}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
