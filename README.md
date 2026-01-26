# Echo SDK

A framework-agnostic AI agent SDK for building LLM-powered applications. Echo provides a unified interface for working with multiple LLM providers (AWS Bedrock, OpenAI, Anthropic) and supports both standalone execution and integration with frameworks like CrewAI and LangGraph.

## Features

- **Multi-Provider LLM Support**: Seamlessly switch between AWS Bedrock, OpenAI, and Anthropic
- **Framework Agnostic**: Use standalone or integrate with CrewAI/LangGraph via adapters
- **Tool System**: Build custom tools with automatic schema conversion for all providers
- **Streaming Support**: Real-time response streaming with event-based handling
- **MCP Integration**: Connect to Model Context Protocol servers for external tools
- **Conversation Management**: Multi-turn conversations with provider-agnostic message handling

## Installation

### From Git

```bash
# Basic installation
pip install git+https://github.com/eka-care/echo-sdk.git

# With uv
uv add git+https://github.com/eka-care/echo-sdk.git
```

### With Optional Dependencies

```bash
# AWS Bedrock support (default provider)
pip install "echo[bedrock] @ git+https://github.com/eka-care/echo-sdk.git"

# OpenAI support
pip install "echo[openai] @ git+https://github.com/eka-care/echo-sdk.git"

# Anthropic support
pip install "echo[anthropic] @ git+https://github.com/eka-care/echo-sdk.git"

# All providers
pip install "echo[all] @ git+https://github.com/eka-care/echo-sdk.git"

# MCP tools support
pip install "echo[mcp] @ git+https://github.com/eka-care/echo-sdk.git"
```

### From Local Build

```bash
# Clone and install
git clone https://github.com/eka-care/echo-sdk.git
cd echo-sdk
pip install .

# Or with uv
uv sync
```

## Quick Start

### 1. Basic LLM Usage

```python
import asyncio
from echo.llm import LLMConfig, get_llm
from echo.models import ConversationContext, Message, MessageRole, TextMessage

async def main():
    # Create LLM instance (defaults to AWS Bedrock)
    llm = get_llm(LLMConfig())

    # Create conversation context
    context = ConversationContext()
    context.add_message(
        Message(
            role=MessageRole.USER,
            content=[TextMessage(text="Hello! What can you help me with?")]
        )
    )

    # Get response
    response, updated_context = await llm.invoke(
        context=context,
        system_prompt="You are a helpful assistant."
    )

    print(response.text)

asyncio.run(main())
```

### 2. Using GenericAgent

```python
import asyncio
from echo.agents import GenericAgent
from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.llm import LLMConfig
from echo.models import ConversationContext, Message, MessageRole, TextMessage

async def main():
    # Configure the agent
    config = AgentConfig(
        persona=PersonaConfig(
            role="Helpful Assistant",
            goal="Help users with their questions",
            backstory="You are a knowledgeable and friendly assistant."
        ),
        task=TaskConfig(
            description="Answer user questions helpfully and accurately.",
            expected_output="A clear and helpful response."
        )
    )

    # Create agent
    agent = GenericAgent(
        agent_config=config,
        llm_config=LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
    )

    # Create conversation
    context = ConversationContext()
    context.add_message(
        Message(
            role=MessageRole.USER,
            content=[TextMessage(text="What is Python?")]
        )
    )

    # Run agent
    result = await agent.run(context)
    print(result.llm_response.text)

asyncio.run(main())
```

## Creating Custom Agents

Extend `BaseAgent` to create your own specialized agents:

```python
from typing import AsyncGenerator
from echo.agents import BaseAgent, AgentResult
from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.llm import LLMConfig
from echo.llm.schemas import StreamEvent
from echo.models import ConversationContext

class MyCustomAgent(BaseAgent):
    """A custom agent for specific tasks."""

    @property
    def name(self) -> str:
        return "my_custom_agent"

    def __init__(self, llm_config: LLMConfig = None, tools: list = None):
        # Define your agent's configuration
        config = AgentConfig(
            persona=PersonaConfig(
                role="Specialist Agent",
                goal="Perform specialized tasks with expertise",
                backstory="You are an expert in your domain."
            ),
            task=TaskConfig(
                description="Handle specialized queries with domain knowledge.",
                expected_output="Expert-level responses."
            )
        )
        super().__init__(
            agent_config=config,
            llm_config=llm_config,
            tools=tools
        )

    async def run(self, context: ConversationContext) -> AgentResult:
        """Execute the agent's task."""
        # Add custom preprocessing if needed
        # ...

        # Use the base implementation
        return await self._run_agent(context)

    async def run_stream(self, context: ConversationContext) -> AsyncGenerator[StreamEvent, None]:
        """Stream the agent's response."""
        async for event in self._run_agent_stream(context):
            yield event


# Usage
async def main():
    agent = MyCustomAgent(
        llm_config=LLMConfig(provider="openai", model="gpt-4o-mini")
    )

    context = ConversationContext()
    context.add_message(
        Message(
            role=MessageRole.USER,
            content=[TextMessage(text="Help me with a specialized task")]
        )
    )

    result = await agent.run(context)
    print(result.llm_response.text)
```

## Creating Custom Tools

Extend `BaseTool` to create tools that your agents can use:

```python
from typing import Any, Dict
from echo.tools import BaseTool

class WeatherTool(BaseTool):
    """Tool to get weather information."""

    name = "get_weather"
    description = "Get current weather for a location"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature units"
                }
            },
            "required": ["location"]
        }

    async def run(self, location: str, units: str = "celsius", **kwargs) -> str:
        # Implement your tool logic here
        # This could call an external API, database, etc.
        return f"Weather in {location}: 22 degrees {units}"


class SearchTool(BaseTool):
    """Tool to search a knowledge base."""

    name = "search_knowledge"
    description = "Search the knowledge base for information"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    async def run(self, query: str, max_results: int = 5, **kwargs) -> str:
        # Implement search logic
        results = [f"Result {i+1} for '{query}'" for i in range(max_results)]
        return "\n".join(results)


# Using tools with an agent
async def main():
    tools = [WeatherTool(), SearchTool()]

    agent = GenericAgent(
        agent_config=AgentConfig(
            persona=PersonaConfig(
                role="Research Assistant",
                goal="Help users find information",
                backstory="You have access to weather data and a knowledge base."
            ),
            task=TaskConfig(
                description="Answer questions using available tools.",
                expected_output="Accurate information from tools."
            )
        ),
        llm_config=LLMConfig(),
        tools=tools
    )

    context = ConversationContext()
    context.add_message(
        Message(
            role=MessageRole.USER,
            content=[TextMessage(text="What's the weather in Tokyo?")]
        )
    )

    result = await agent.run(context)
    print(result.llm_response.text)
```

## LLM Providers

### AWS Bedrock (Default)

```python
from echo.llm import LLMConfig, get_llm

# Default: Claude Haiku on Bedrock
llm = get_llm(LLMConfig())

# Custom Bedrock model
llm = get_llm(LLMConfig(
    provider="bedrock",
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0.7,
    max_tokens=2000
))
```

Required environment variables:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=ap-south-1
```

### OpenAI

```python
llm = get_llm(LLMConfig(
    provider="openai",
    model="gpt-4o-mini",
    temperature=0.7
))
```

Required: `export OPENAI_API_KEY=your_key`

### Anthropic

```python
llm = get_llm(LLMConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    temperature=0.7
))
```

Required: `export ANTHROPIC_API_KEY=your_key`

## Streaming Responses

```python
from echo.llm.schemas import StreamEventType

async def stream_example():
    llm = get_llm(LLMConfig())
    context = ConversationContext()
    context.add_message(
        Message(role=MessageRole.USER, content=[TextMessage(text="Tell me a story")])
    )

    async for event in llm.invoke_stream(context, system_prompt="You are a storyteller."):
        if event.type == StreamEventType.TEXT:
            print(event.text, end="", flush=True)
        elif event.type == StreamEventType.TOOL_CALL_START:
            print(f"\n[Calling tool: {event.json.get('tool_name')}]")
        elif event.type == StreamEventType.TOOL_CALL_END:
            print(" Done")
        elif event.type == StreamEventType.DONE:
            print("\n--- Complete ---")
            final_response = event.llm_response
        elif event.type == StreamEventType.ERROR:
            print(f"\nError: {event.error}")
```

## MCP Tool Integration

Connect to Model Context Protocol servers:

```python
from echo.tools import MCPServerConfig, MCPConnectionManager, MCPTransport

async def use_mcp_tools():
    # Configure MCP server
    server_config = MCPServerConfig(
        transport=MCPTransport.STREAMABLE_HTTP,
        url="http://your-mcp-server/mcp",
        headers={"Authorization": "Bearer token"}
    )

    # Discover and use tools (with automatic connection management)
    manager = MCPConnectionManager(server_config)
    tools = await manager.get_tools()

    print(f"Found {len(tools)} tools")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    # Use tools with an agent
    agent = GenericAgent(
        agent_config=config,
        llm_config=LLMConfig(),
        tools=tools
    )
    result = await agent.run(context)
```

## Conversation Context

Manage multi-turn conversations:

```python
from echo.models import (
    ConversationContext,
    Message,
    MessageRole,
    TextMessage,
    ToolCall,
    ToolResult
)

# Create context
context = ConversationContext()

# Add user message
context.add_message(
    Message(
        role=MessageRole.USER,
        content=[TextMessage(text="Hello!")]
    )
)

# After LLM response, context is automatically updated
response, context = await llm.invoke(context)

# Continue the conversation
context.add_message(
    Message(
        role=MessageRole.USER,
        content=[TextMessage(text="Tell me more")]
    )
)

# Get next response with full history
response, context = await llm.invoke(context)
```

## Creating Examples in Your Codebase

Here's how to set up examples in your own project:

### Project Structure

```
your-project/
├── pyproject.toml
├── src/
│   └── your_app/
│       ├── __init__.py
│       ├── agents/
│       │   ├── __init__.py
│       │   └── my_agent.py
│       └── tools/
│           ├── __init__.py
│           └── my_tools.py
└── examples/
    ├── __init__.py
    ├── basic_usage.py
    ├── custom_agent.py
    └── with_tools.py
```

### Example: `examples/basic_usage.py`

```python
"""
Basic usage example for Echo SDK.

Run with: python examples/basic_usage.py
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()  # Load .env file for API keys

from echo.llm import LLMConfig, get_llm
from echo.models import ConversationContext, Message, MessageRole, TextMessage


async def main():
    # Initialize LLM
    llm = get_llm(LLMConfig(
        provider="anthropic",  # or "bedrock", "openai"
        model="claude-haiku-4-5-20251001",
        temperature=0.7
    ))

    # Create conversation
    context = ConversationContext()

    print("Echo SDK Basic Example")
    print("=" * 40)
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # Add user message
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text=user_input)]
            )
        )

        # Get response
        response, context = await llm.invoke(
            context=context,
            system_prompt="You are a helpful assistant. Be concise."
        )

        print(f"Assistant: {response.text}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

### Example: `examples/custom_agent.py`

```python
"""
Custom agent example.

Run with: python examples/custom_agent.py
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from echo.agents import GenericAgent
from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.llm import LLMConfig
from echo.models import ConversationContext, Message, MessageRole, TextMessage


# Define your agent configuration
MY_AGENT_CONFIG = AgentConfig(
    persona=PersonaConfig(
        role="Code Review Assistant",
        goal="Help developers improve their code quality",
        backstory="You are an experienced software engineer who specializes in code review and best practices."
    ),
    task=TaskConfig(
        description="""Review code snippets provided by users and give constructive feedback on:
        - Code quality and readability
        - Potential bugs or issues
        - Performance improvements
        - Best practices""",
        expected_output="Detailed code review with specific suggestions for improvement."
    )
)


async def main():
    # Create agent
    agent = GenericAgent(
        agent_config=MY_AGENT_CONFIG,
        llm_config=LLMConfig(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            temperature=0.3  # Lower for more focused responses
        )
    )

    # Create conversation
    context = ConversationContext()

    print("Code Review Agent")
    print("=" * 40)
    print("Paste your code and get feedback!")
    print("Type 'quit' to exit\n")

    while True:
        print("Enter your code (type 'END' on a new line to submit):")
        lines = []
        while True:
            line = input()
            if line == "END":
                break
            if line.lower() == "quit":
                return
            lines.append(line)

        code = "\n".join(lines)
        if not code.strip():
            continue

        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text=f"Please review this code:\n\n```\n{code}\n```")]
            )
        )

        print("\nReviewing...")
        result = await agent.run(context)

        if result.llm_response:
            print(f"\nReview:\n{result.llm_response.text}\n")
            context = result.context  # Update context for follow-up
        else:
            print(f"Error: {result.error}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Example: `examples/with_tools.py`

```python
"""
Agent with custom tools example.

Run with: python examples/with_tools.py
"""
import asyncio
from typing import Any, Dict
from dotenv import load_dotenv

load_dotenv()

from echo.agents import GenericAgent
from echo.agents.config import AgentConfig, PersonaConfig, TaskConfig
from echo.llm import LLMConfig
from echo.models import ConversationContext, Message, MessageRole, TextMessage
from echo.tools import BaseTool


# Define custom tools
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Perform mathematical calculations. Supports +, -, *, /, and ** (power)."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g., '2 + 2' or '10 ** 2'"
                }
            },
            "required": ["expression"]
        }

    async def run(self, expression: str, **kwargs) -> str:
        try:
            # Safety: only allow basic math operations
            allowed = set("0123456789+-*/(). ")
            if not all(c in allowed for c in expression):
                return "Error: Invalid characters in expression"
            result = eval(expression)
            return f"{expression} = {result}"
        except Exception as e:
            return f"Error calculating: {e}"


class DateTimeTool(BaseTool):
    name = "get_datetime"
    description = "Get the current date and time."

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "description": "Output format: 'full', 'date', or 'time'",
                    "enum": ["full", "date", "time"]
                }
            },
            "required": []
        }

    async def run(self, format: str = "full", **kwargs) -> str:
        from datetime import datetime
        now = datetime.now()
        if format == "date":
            return now.strftime("%Y-%m-%d")
        elif format == "time":
            return now.strftime("%H:%M:%S")
        return now.strftime("%Y-%m-%d %H:%M:%S")


async def main():
    # Create tools
    tools = [CalculatorTool(), DateTimeTool()]

    # Create agent with tools
    agent = GenericAgent(
        agent_config=AgentConfig(
            persona=PersonaConfig(
                role="Helpful Assistant with Tools",
                goal="Help users by using available tools when appropriate",
                backstory="You are an assistant with access to a calculator and datetime tool."
            ),
            task=TaskConfig(
                description="Answer user questions. Use tools when calculations or date/time info is needed.",
                expected_output="Helpful responses using tools when appropriate."
            )
        ),
        llm_config=LLMConfig(provider="anthropic", model="claude-haiku-4-5-20251001"),
        tools=tools
    )

    context = ConversationContext()

    print("Agent with Tools")
    print("=" * 40)
    print("Available tools: calculator, get_datetime")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text=user_input)]
            )
        )

        result = await agent.run(context)

        if result.llm_response:
            # Show tool calls if any
            if result.llm_response.verbose:
                tool_calls = [v for v in result.llm_response.verbose if v.type == "tool"]
                if tool_calls:
                    print(f"[Tools used: {', '.join(t.tool_name for t in tool_calls)}]")

            print(f"Assistant: {result.llm_response.text}\n")
            context = result.context
        else:
            print(f"Error: {result.error}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

### Running Examples

```bash
# Set up environment variables
export ANTHROPIC_API_KEY=your_key
# or
export OPENAI_API_KEY=your_key
# or for Bedrock
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=ap-south-1

# Run examples
python examples/basic_usage.py
python examples/custom_agent.py
python examples/with_tools.py
```

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `BaseAgent` | Abstract base class for agents |
| `GenericAgent` | Flexible agent for any task |
| `BaseTool` | Abstract base class for tools |
| `LLMConfig` | LLM provider configuration |
| `ConversationContext` | Multi-turn conversation state |

### Key Methods

| Method | Description |
|--------|-------------|
| `agent.run(context)` | Execute agent and get result |
| `agent.run_stream(context)` | Stream agent response |
| `llm.invoke(context)` | Get LLM response |
| `llm.invoke_stream(context)` | Stream LLM response |
| `tool.run(**kwargs)` | Execute tool logic |

## License

MIT License - see [LICENSE](LICENSE) for details.
