import asyncio

from dotenv import load_dotenv

load_dotenv()

from health_agents import TranscriptToClinicalNotesAgent  # noqa: E402

from _harness import (  # noqa: E402
    DEFAULT_TEMPLATE,
    banner,
    describe_config,
    example_llm_config,
    load_transcript,
    rule,
)


async def main() -> None:
    banner("Transcript -> Clinical Notes (non-streaming)")
    llm_config = example_llm_config()
    describe_config(llm_config)

    agent = TranscriptToClinicalNotesAgent(
        user_prompt=DEFAULT_TEMPLATE,
        llm_config=llm_config,
    )

    transcript = load_transcript()
    rule("transcript")
    print(transcript)
    print()

    result = await agent.generate(transcript)

    rule("clinical note")
    if result.error:
        print(f"[error] {result.error}")
    elif result.llm_response:
        print(result.llm_response.text)
    else:
        print("[no response]")
    print()


if __name__ == "__main__":
    asyncio.run(main())
