"""
Example: Using GenericAgent with MCP tools.

This demonstrates:
- Connecting to multiple MCP servers via SSE
- Discovering and using MCP tools with GenericAgent
- Running a multi-turn conversation with user input
"""

import asyncio
import os

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.agents.generic_agent import GenericAgent
from echo.llm import LLMConfig
from echo.models.user_conversation import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
)
from echo.tools import MCPServerConfig, MCPToolProvider, MCPTransport

# MCP Server configurations
print(os.getenv("EK_JWT_PAYLOAD"))
MCP_SERVERS = [
    MCPServerConfig(
        transport=MCPTransport.STREAMABLE_HTTP,
        url="http://remote-mcp-internal.orbi.orbi/mcp",
        headers={
            # "Authorization": "Bearer <Eka Access Token>"
            "x-eka-jwt-payload": os.getenv("EKA_JWT_PAYLOAD")
        },
    ),
]

# llm_config = LLMConfig(
#     provider="anthropic",
#     model="claude-haiku-4-5-20251001",
#     temperature=0.2,
#     max_tokens=2000,
#     max_iterations=5,  # Allow multiple iterations for tool call + response
# )

# Agent configuration
AGENT_CONFIG = AgentConfig(
    persona=PersonaConfig(
        role="Medical Assistant chatbot called Ekabot",
        goal="Help users with medical queries using available tools. Use tools when appropriate to provide accurate information.",
        backstory="You are a helpful medical assistant with access to various tools to look up information and assist users.",
    ),
    task=TaskConfig(
        description="You are a helpful medical assistant with access to various tools to look up information and assist users.",
        expected_output="A helpful response that addresses the user's query, using tool results when appropriate.",
    ),
)


async def discover_tools_from_servers():
    """Discover and display available tools from all MCP servers."""
    print("Discovering tools from MCP servers...\n")

    all_tools = []

    for server_config in MCP_SERVERS:
        try:
            provider = MCPToolProvider(server_config)
            print(f"Connecting to: {server_config.url}")

            async with provider.connect() as tools:
                print(f"  Found {len(tools)} tools:")
                for tool in tools:
                    print(f"    - {tool.name}: {tool.description}")
                all_tools.extend(tools)

        except Exception as e:
            print(f"  Error connecting to {server_config.url}: {e}")

    print(f"\nTotal tools discovered: {len(all_tools)}")
    return all_tools


async def run_conversation():
    """Run a 10-message conversation with user input."""
    print("=" * 60)
    print("MCP Agent Conversation Example")
    print("=" * 60)
    print()

    # Collect tools from all MCP servers
    all_tools = []
    provider_contexts = []

    for server_config in MCP_SERVERS:
        try:
            provider = MCPToolProvider(server_config)
            print(f"Connecting to: {server_config.url}")

            # Enter the context and keep it open
            ctx = provider.connect()
            tools = await ctx.__aenter__()
            provider_contexts.append(ctx)

            print(f"  Connected! Found {len(tools)} tools")
            all_tools.extend(tools)

        except Exception as e:
            print(f"  Error connecting to {server_config.url}: {e}")

    if not all_tools:
        print("\nNo tools available. Exiting.")
        return

    print(f"\nTotal tools available: {len(all_tools)}")
    print("Tools:", ", ".join(t.name for t in all_tools))
    print()

    try:
        # Create agent with all MCP tools
        agent = GenericAgent(
            agent_config=AGENT_CONFIG,
            tools=all_tools,
            llm_config=LLMConfig(),
        )

        # Create conversation context
        context = ConversationContext()

        print("=" * 60)
        print("Starting conversation (10 messages)")
        print("Type 'quit' to exit early")
        print("=" * 60)
        print()

        for i in range(10):
            # Get user input
            try:
                user_input = input(f"[{i+1}/10] You: ").strip()
            except EOFError:
                print("\nEnd of input. Exiting.")
                break

            if user_input.lower() == "quit":
                print("Exiting conversation.")
                break

            if not user_input:
                print("Empty input, skipping...")
                continue

            # Add user message to context
            context.add_message(
                Message(
                    role=MessageRole.USER,
                    content=[TextMessage(text=user_input)],
                )
            )

            # Run agent
            print("Agent thinking...")
            try:
                result = await agent.run(context)

                if result.error:
                    print(f"Agent Error: {result.error}")
                elif result.llm_response:
                    # Display verbose output (text and tool calls from all iterations)
                    if result.llm_response.verbose:
                        print("\n--- Verbose Output ---")
                        for item in result.llm_response.verbose:
                            if item.type == "text":
                                print(f"[Text] {item.text}")
                            elif item.type == "tool":
                                print(f"[Tool] Called: {item.tool_name}")
                        print("--- End Verbose ---")

                    # Display final response
                    print(f"\nAgent: {result.llm_response.text}")

                    # Check if there are elicitations to display
                    if result.llm_response.elicitations:
                        print("\n--- Elicitation UI ---")
                        for elicitation in result.llm_response.elicitations:
                            print(f"Component: {elicitation.details.component}")
                            print(
                                f"Options: {elicitation.details.input.get('options', [])}"
                            )
                            print(f"Text: {elicitation.details.input.get('text', '')}")
                        print("--- End Elicitation ---")
                else:
                    print(f"Agent: {result}")

                # Update context with the result's context if available
                if result.context:
                    context = result.context

            except Exception as e:
                print(f"Error running agent: {e}")

            print()

        print("=" * 60)
        print("Conversation ended")
        print("=" * 60)

    finally:
        # Clean up all provider contexts
        for ctx in provider_contexts:
            try:
                await ctx.__aexit__(None, None, None)
            except Exception:
                pass


async def main():
    """Main entry point."""
    print("Choose an option:")
    print("1. Discover tools from MCP servers")
    print("2. Run conversation")
    print()

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        await discover_tools_from_servers()
    elif choice == "2":
        await run_conversation()
    else:
        print("Invalid choice. Running conversation by default.")
        await run_conversation()


if __name__ == "__main__":
    asyncio.run(main())

""" 
Examples
Server Config Examples:
    For SSE transport:
        config = MCPServerConfig(
            transport=MCPTransport.SSE,
            url="http://localhost:8000/sse",
            headers={"Authorization": "Bearer token"}
        )

    For Streamable HTTP transport (HTTP POST with JSON-RPC):
        config = MCPServerConfig(
            transport=MCPTransport.STREAMABLE_HTTP,
            url="http://localhost:8000/mcp/",
            headers={"Authorization": "Bearer token"}
        )

    For stdio transport:
        config = MCPServerConfig(
            transport=MCPTransport.STDIO,
            command="python",
            args=["my_mcp_server.py"],
            env={"API_KEY": "secret"}
        )

MCP Tool Provider Examples:
    Example:
        # SSE with custom headers
        config = MCPServerConfig(
            transport=MCPTransport.SSE,
            url="http://mcp-server/sse",
            headers={"X-Tenant-ID": "123"}
        )
        provider = MCPToolProvider(config)

        async with provider.connect() as tools:
            # tools is List[MCPTool]
            agent = MyAgent(tools=tools)
            result = agent.run("query")

        # With filtering
        async with provider.connect(filter_fn=lambda t: t.name.startswith("search_")) as tools:
            # Only tools matching the filter
            ...
"""
