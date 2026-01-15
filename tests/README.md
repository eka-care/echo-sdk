# Echo SDK Tests

## Running Tests

```bash
# All tests
uv run pytest

# Unit tests only (no LLM calls required)
uv run pytest -k "not Integration"

# Integration tests only (requires LLM credentials)
uv run pytest -k "Integration"

# Specific test file
uv run pytest tests/test_llm_response.py -v

# Verbose with print output
uv run pytest -v -s

# With coverage
uv run pytest --cov=echo
```

## Test Files

| File | Description | Requires LLM |
|------|-------------|--------------|
| `test_llm_response.py` | LLMResponse and VerboseResponseItem model tests | No |
| `test_llm_providers_integration.py` | Integration tests for Bedrock, OpenAI, Anthropic providers | Yes |
| `conftest.py` | Pytest fixtures and path configuration | - |

## Test Categories

### Unit Tests
Fast tests that don't require LLM calls:
```bash
uv run pytest -k "not Integration"
```

These include:
- `TestVerboseResponseItem` - VerboseResponseItem model creation and serialization
- `TestLLMResponse` - LLMResponse model with text, verbose, json, error fields
- `TestVerboseOutputScenarios` - Realistic verbose output scenarios (tool calls, elicitations)

### Integration Tests
Tests that call actual LLM APIs (require credentials):
```bash
uv run pytest -k "Integration"
```

These include:
- `TestBedrockLLMIntegration` - AWS Bedrock Claude provider
- `TestOpenAILLMIntegration` - OpenAI GPT provider
- `TestAnthropicLLMIntegration` - Anthropic Claude provider

## Prerequisites for Integration Tests

Set up LLM credentials for your preferred provider:

```bash
# AWS Bedrock (default)
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_REGION=ap-south-1

# OpenAI
export OPENAI_API_KEY=your_key

# Anthropic
export ANTHROPIC_API_KEY=your_key
```

## Test Structure

```python
class TestLLMResponse:
    """Unit tests - no LLM calls"""
    def test_create_empty_response(self): ...
    def test_create_response_with_verbose(self): ...

class TestBedrockLLMIntegration:
    """Integration tests - requires AWS credentials"""
    async def test_simple_text_response_has_verbose(self): ...
    async def test_tool_call_response_has_verbose(self): ...
```

## Writing New Tests

```python
import pytest
from echo.llm import LLMConfig, get_llm
from echo.llm.schemas import LLMResponse, VerboseResponseItem
from echo.models import ConversationContext, Message, MessageRole, TextMessage

class TestMyFeature:
    def test_unit_test(self):
        """No LLM call needed"""
        response = LLMResponse(text="Hello")
        assert response.text == "Hello"
        assert response.verbose == []

    @pytest.mark.asyncio
    async def test_integration_test(self):
        """Requires LLM - will fail without credentials"""
        llm = get_llm(LLMConfig(provider="bedrock"))
        context = ConversationContext()
        context.add_message(
            Message(
                role=MessageRole.USER,
                content=[TextMessage(text="Hello")]
            )
        )
        response, _ = await llm.invoke(context)
        assert response.text != ""
```

## Common Issues

**AWS Token Expired:**
```
ExpiredTokenException: The security token included in the request is expired
```
Refresh your AWS credentials.

**Model Access:**
```
Request validation failed: Invocation of model ID...
```
Enable the model in AWS Bedrock console or use a different provider.

**Missing API Key:**
```
AuthenticationError: No API key provided
```
Set the appropriate environment variable for your provider.
