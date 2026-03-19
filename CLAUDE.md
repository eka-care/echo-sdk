# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Echo SDK is a framework-agnostic / LLM agnostic AI agent SDK for Multi Agent Creation. It provides multi-provider LLM support (AWS Bedrock, OpenAI, Anthropic), agent orchestration with tool support, and MCP (Model Context Protocol) integration.

## Development Commands

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Run unit tests only (no LLM credentials required)
uv run pytest -k "not Integration"

# Run integration tests (requires LLM credentials)
uv run pytest -k "Integration"

# Run specific test file
uv run pytest tests/test_llm_response.py -v

# Run examples (requires .env configured)
uv run python examples/streaming_usage.py
uv run python examples/mcp_agent_usage.py
```

## Architecture

### Core Design
- **Async-first**: All LLM calls, tools, and agents use async/await
- **Provider abstraction**: Base classes (BaseAgent, BaseLLM, BaseTool) with provider-specific implementations
- **Configuration-driven**: YAML support for agent/task config, environment variable defaults, Pydantic models

### Main Components

**Agents** (`src/echo/agents/`):
- `BaseAgent` - Abstract agent interface with `run()` and `run_stream()` methods
- `GenericAgent` - Flexible implementation configured via `AgentConfig` (code or YAML)
- Supports persona (role, goal, backstory) and task (description, expected_output)

**LLM Module** (`src/echo/llm/`):
- `BaseLLM` - Abstract interface with `invoke()` and `invoke_stream()` methods
- Providers: `BedrockLLM` (default), `OpenAILLM`, `AnthropicLLM`
- `get_llm(config)` - Factory function for provider instantiation
- `LLMConfig` - Configuration with env var defaults (ECHO_DEFAULT_LLM_*)

**Models** (`src/echo/models/`):
- `ConversationContext` - Multi-turn conversation manager with provider-specific message formatters
- `Message`, `ToolCall`, `ToolResult` - Conversation building blocks

**Tools** (`src/echo/tools/`):
- `BaseTool` - Abstract tool interface with schema adapters for each provider
- `MCPToolProvider` - Discovers and wraps MCP server tools
- `BaseElicitationTool` - UI elicitation tools for structured input

### Agentic Loop Flow
1. Generate LLM response
2. Extract tool calls from response
3. Execute tools and collect results
4. Feed results back to LLM
5. Repeat until no tool calls or max_iterations reached

### Streaming
- `StreamEvent` types: TEXT, TOOL_CALL_START, TOOL_CALL_END, DONE, ERROR
- Access final `LLMResponse` from DONE event

## Environment Variables

```bash
# LLM provider selection (bedrock|openai|anthropic)
ECHO_DEFAULT_LLM_PROVIDER=bedrock
ECHO_DEFAULT_LLM_MODEL=anthropic.claude-3-haiku-20240307-v1:0
ECHO_DEFAULT_LLM_TEMPERATURE=0.2
ECHO_DEFAULT_LLM_MAX_TOKENS=<int>
ECHO_DEFAULT_LLM_MAX_ITERATIONS=10

# Provider credentials
AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION  # Bedrock
OPENAI_API_KEY  # OpenAI
ANTHROPIC_API_KEY  # Anthropic
```

## Optional Dependencies

Install via extras: `uv add echo[openai]`, `uv add echo[anthropic]`, `uv add echo[bedrock]`, `uv add echo[crewai]`, `uv add echo[mcp]`, or `uv add echo[all]`
