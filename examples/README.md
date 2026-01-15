# Echo SDK Examples

## Available Examples

| File | Description | Requires LLM |
|------|-------------|--------------|
| `streaming_usage.py` | Streaming responses with elicitation tools | Yes |
| `mcp_agent_usage.py` | GenericAgent with MCP tools integration | Yes |

## Running Examples

```bash
# Set up environment (copy .env.sample to .env and fill in credentials)
cp .env.sample .env
source .env

# Run streaming example
uv run python examples/streaming_usage.py

# Run MCP agent example
uv run python examples/mcp_agent_usage.py
```

## Streaming Usage

Demonstrates streaming responses with real-time output and elicitation tools:

```python
import asyncio
from echo.llm import LLMConfig, get_llm
from echo.llm.schemas import StreamEventType
from echo.models import ConversationContext, Message, MessageRole, TextMessage

async def main():
    # Create LLM (supports bedrock, anthropic, openai)
    llm = get_llm(LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514"))

    context = ConversationContext()
    context.add_message(
        Message(
            role=MessageRole.USER,
            content=[TextMessage(text="Tell me a joke")]
        )
    )

    # Stream response
    async for event in llm.invoke_stream(context):
        if event.type == StreamEventType.TEXT:
            print(event.text, end="", flush=True)
        elif event.type == StreamEventType.TOOL_CALL_START:
            print(f"\n[Tool: {event.json.get('tool_name')}]")
        elif event.type == StreamEventType.TOOL_CALL_END:
            print(" Done")
        elif event.type == StreamEventType.DONE:
            # Access final response and updated context
            response = event.llm_response
            context = event.context

            # Check for elicitations
            if response.elicitations:
                print(f"Elicitation: {response.elicitations[0]}")
        elif event.type == StreamEventType.ERROR:
            print(f"Error: {event.error}")

asyncio.run(main())
```

### Stream Event Types

- `TEXT` - Streamed text chunk (`event.text`)
- `TOOL_CALL_START` - Tool call started (`event.json` has tool info)
- `TOOL_CALL_END` - Tool call completed
- `DONE` - Stream complete (`event.llm_response`, `event.context`)
- `ERROR` - Error occurred (`event.error`)

## MCP Agent Usage

Demonstrates using GenericAgent with MCP (Model Context Protocol) tools:

```python
import asyncio
from echo.agents import GenericAgent
from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.llm import LLMConfig
from echo.models import ConversationContext, Message, MessageRole, TextMessage
from echo.tools import MCPServerConfig, MCPToolProvider, MCPTransport

async def main():
    # Configure MCP server
    mcp_config = MCPServerConfig(
        transport=MCPTransport.STREAMABLE_HTTP,
        url="http://your-mcp-server/mcp",
        headers={"x-auth-token": "your_token"},
    )

    # Define agent configuration
    agent_config = AgentConfig(
        persona=PersonaConfig(
            role="Medical Assistant",
            goal="Help users with medical queries using available tools",
            backstory="You are a helpful medical assistant.",
        ),
        task=TaskConfig(
            description="Assist the user with their medical query.",
            expected_output="A helpful response using available tools.",
        ),
    )

    # Discover tools from MCP server
    async with MCPToolProvider([mcp_config]) as provider:
        tools = await provider.get_tools()
        print(f"Discovered {len(tools)} tools")

        # Create agent with MCP tools
        agent = GenericAgent(
            agent_config=agent_config,
            llm_config=LLMConfig(
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                max_iterations=5,
            ),
            tools=tools,
        )

        # Create conversation
        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text="Book an appointment with Dr. Smith")]
            )
        )

        # Run agent
        result = await agent.run(context)
        print(result.llm_response.text)

        # Or stream the response
        async for event in agent.run_stream(context):
            if event.type == StreamEventType.TEXT:
                print(event.text, end="", flush=True)

asyncio.run(main())
```

### MCP Transport Types

- `STREAMABLE_HTTP` - HTTP with streaming support (recommended)
- `HTTP` - Standard HTTP transport

## Creating Custom Tools

```python
from echo.tools import BaseTool

class MyCustomTool(BaseTool):
    name = "my_tool"
    description = "Description of what this tool does"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                }
            },
            "required": ["query"]
        }

    async def run(self, query: str, **kwargs) -> str:
        # Tool implementation
        return f"Result for: {query}"

# Use with agent
agent = GenericAgent(
    agent_config=config,
    tools=[MyCustomTool()],
)
```

## Prerequisites

Set up LLM credentials:

```bash
# AWS Bedrock (default)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=ap-south-1

# Or OpenAI
export OPENAI_API_KEY=your_key

# Or Anthropic
export ANTHROPIC_API_KEY=your_key
```
