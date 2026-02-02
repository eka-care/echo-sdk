"""
Example: Using GenericAgent with MCP auth tool.

This demonstrates:
- Connecting to multiple MCP servers via SSE
- Discovering and using MCP tools with GenericAgent
- Authenticating user using authenticate tool from MCP servers
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
    ToolResult,
)
from echo.tools import MCPConnectionManager, MCPServerConfig, MCPTransport


# MCP Server configurations
MCP_SERVERS = [
    MCPServerConfig(
        transport=MCPTransport.STDIO,
        command="/Users/surabhi.vatsa/Documents/Github/eka-mcp-sdk/.venv/bin/python",
        args=["-m", "eka_mcp_sdk.server"],
        env= {
            "EKA_CLIENT_ID": "",
            "EKA_CLIENT_SECRET": "",
            "EKA_API_KEY": ""
        }
    ),
    # MCPServerConfig(
    #     transport=MCPTransport.STREAMABLE_HTTP,
    #     url="http://remote-mcp-internal.orbi.orbi/mcp",
    #     headers={"x-eka-jwt-payload": os.getenv("EKA_JWT_PAYLOAD")},
    # ),
]

llm_config = LLMConfig(
    provider="bedrock",
    model="anthropic.claude-3-haiku-20240307-v1:0",
    temperature=0.2,
    max_tokens=2000,
    max_iterations=5,
)

# Agent configuration
AUTH_AGENT_CONFIG = AgentConfig(
    persona=PersonaConfig(
        role="",
        goal="Help users to log into the system using available tools",
        backstory="You are a helpful assistant with access to various tools including authenticate_user to assist users.",
    ),
    task=TaskConfig(
        description="Help the user with authentication using the available tools.",
        expected_output="Authenticated user and tool call details when using the tool.",
    ),
)


async def discover_tools_from_servers():
    """Discover and display available tools from all MCP servers."""
    print("Discovering tools from MCP servers...\n")

    all_tools = []

    for server_config in MCP_SERVERS:
        try:
            manager = MCPConnectionManager(server_config)
            print(f"Connecting to: {server_config.url}")

            tools = await manager.get_tools()
            print(f"  Found {len(tools)} tools:")
            for tool in tools:
                print(f"    - {tool.name}: {tool.description}")
            all_tools.extend(tools)

        except Exception as e:
            print(f"  Error connecting to {server_config.url}: {e}")

    print(f"\nTotal tools discovered: {len(all_tools)}")

    # Cleanup connections
    await MCPConnectionManager.cleanup_all()
    return all_tools


async def user_authentication():
    """Authenticate user using the tool in MCP server"""
    print("=" * 60)
    print("MCP User Authentication Example")
    print("=" * 60)
    print()

    # Collect tools from all MCP servers
    all_tools = []

    for server_config in MCP_SERVERS:
        try:
            manager = MCPConnectionManager(server_config)
            print(f"Connecting to: {server_config.url}")

            tools = await manager.get_tools(filter_fn=lambda t: t.name.startswith("authenticate_"))
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
            agent_config=AUTH_AGENT_CONFIG,
            tools=all_tools,
            llm_config=llm_config,
        )

        # Create conversation context
        context = ConversationContext()

        print("=" * 60)
        print("Starting conversation (10 messages)")
        print("Type 'quit' to exit early")
        print("=" * 60)
        print()

        # Initial greeting - trigger the agent to start
        initial_message = "Hi, I need to login to the system"
        print(f"You: {initial_message}")

        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text=initial_message)],
            )
        )

        active_elicitation_tool_id = None
        for i in range(10):
            # Run agent
            print("Agent thinking...")
            try:
                result = await agent.run(context, "hgrsdfa"+str(i))

                if result.error:
                    print(f"Agent Error: {result.error}")
                elif result.llm_response:
                    # Display verbose output
                    if result.llm_response.verbose:
                        print("\n--- Verbose Output ---")
                        for item in result.llm_response.verbose:
                            if item.type == "text":
                                print(f"[Text] {item.text}")
                            elif item.type == "tool":
                                print(f"[Tool] Called: {item.tool_name}")
                                print(result)
                        print("--- End Verbose ---")

                    # Display final response
                    print(f"\nAgent: {result.llm_response.text}")

                    # Check if there are elicitations
                    if result.llm_response.elicitations:
                        print("\n--- Elicitation UI ---")
                        for elicitation in result.llm_response.elicitations:
                            print(f"Component: {elicitation.details.component}")
                            print(
                                f"Additional Info: {elicitation.details.input.get('additional_info', {})}"
                            )
                            print(f"Text: {elicitation.details.input.get('text', '')}")
                            # Store the tool_id for the response
                            active_elicitation_tool_id = elicitation.tool_id
                        print("--- End Elicitation ---")
                    else:
                        active_elicitation_tool_id = None
                else:
                    print(f"Agent: {result}")

                # Update context
                if result.context:
                    context = result.context

            except Exception as e:
                print(f"Error running agent: {e}")
                import traceback

                traceback.print_exc()
                break

            # Get user input
            print()
            try:
                user_input = input("You: ").strip()
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
            if active_elicitation_tool_id:
                # Respond to elicitation with ToolResult
                context.add_message(
                    Message(
                        role=MessageRole.TOOL,
                        content=[
                            ToolResult(
                                tool_id=active_elicitation_tool_id,
                                result=user_input,
                            )
                        ],
                    )
                )
                active_elicitation_tool_id = None
            else:
                context.add_message(
                    Message(
                        role=MessageRole.USER,
                        content=[TextMessage(text=user_input)],
                    )
                )

            print()

        print("=" * 60)
        print("Conversation ended")
        print("=" * 60)

    finally:
        # Clean up all provider contexts
        await MCPConnectionManager.cleanup_all()


async def main():
    """Main entry point."""
    print("Choose an option:")
    print("1. Discover tools from MCP servers")
    print("2. User authentication")
    print()

    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        await discover_tools_from_servers()
    elif choice == "2":
        await user_authentication()
    else:
        print("Invalid choice. Running conversation by default.")
        await user_authentication()


if __name__ == "__main__":
    asyncio.run(main())
