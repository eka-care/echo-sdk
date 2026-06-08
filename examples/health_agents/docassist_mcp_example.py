import asyncio

from dotenv import load_dotenv

load_dotenv()

from health_agents.tools import (  # noqa: E402
    EKA_CLINICAL_MCP_URL,
    EKA_EMR_MCP_URL,
    get_eka_clinical_tools,
    get_eka_emr_tools,
)

from _harness import banner, env_token, rule  # noqa: E402


async def list_tools(label: str, url: str, fetch, token: str) -> None:
    rule(f"{label}  ({url})")
    try:
        tools = await fetch(token)
    except Exception as exc:  # noqa: BLE001 - example: surface any connection error
        print(f"[error fetching tools] {exc}")
        return
    if not tools:
        print("(no tools returned)")
        return
    for tool in tools:
        print(f"- {tool.name}: {tool.description[:80]}")
    print()


async def main() -> None:
    banner("Eka Care MCP tools + DocAssist (scaffold)")

    token = env_token("EKA_BEARER_TOKEN")
    if not token:
        print(
            "EKA_BEARER_TOKEN is not set — skipping.\n"
            "Set it to a valid Eka Care token to fetch live MCP tools, e.g.:\n"
            "    EKA_BEARER_TOKEN=... python examples/health_agents/"
            "docassist_mcp_example.py"
        )
        return

    await list_tools("EMR MCP", EKA_EMR_MCP_URL, get_eka_emr_tools, token)
    await list_tools(
        "Clinical MCP", EKA_CLINICAL_MCP_URL, get_eka_clinical_tools, token
    )

    rule("DocAssist")
    print(
        "DocAssistAgent is scaffolded only. Once implemented it will fetch the "
        "EMR tools above via setup_tools() and answer/act on patient-record "
        "queries. See health_agents.agents.docassist."
    )


if __name__ == "__main__":
    asyncio.run(main())
