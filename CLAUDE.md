# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Claude Code Model Router - a proxy server that intelligently routes Claude Code CLI traffic between Anthropic's API and OpenAI's API based on configurable rules. It translates between Anthropic Messages API and OpenAI Chat Completions API formats.

## Development Commands

```bash
# Install dependencies
uv sync --dev

# Run the router server
uv run claude-router --config router.yaml

# Run tests
uv run pytest

# Run specific test file
uv run pytest tests/test_router.py -v

# Type checking
uv run mypy src/

# Linting and formatting
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Fix auto-fixable linting issues
uv run ruff check src/ tests/ --fix
```

## Architecture

### Core Components

- **ProxyRouter** (`src/router/server.py`): Main FastAPI application that handles request routing
- **ModelRouter** (`src/router/router.py`): Decision engine that determines routing based on headers, request content, and configuration rules
- **Request Adapters** (`src/router/adapters/`): Translate between API formats
  - `AnthropicOpenAIRequestAdapter`: Converts Anthropic Messages API � OpenAI Chat Completions API
  - `OpenAIAnthropicResponseAdapter`: Converts OpenAI responses � Anthropic format
  - `PassthroughAdapter`: Direct forwarding to Anthropic API

### Routing Logic

The router makes decisions based on:
1. **Override rules** (highest precedence) - custom header/request conditions including:
   - Tool call detection (`has_tool: "ExitPlanMode"` for plan mode)
   - Model name patterns (`model_contains: "haiku"`)
   - Header matching
   - Request data conditions
2. **Default passthrough** - everything else goes to Anthropic

Override rules are processed in order, with the first matching rule taking precedence.

### Configuration

All routing behavior is controlled by `router.yaml`:
- **Model mappings**: `plan_model`, `background_model` for OpenAI models
- **Override rules**: Flexible routing conditions with `when` clauses
- **OpenAI settings**: API key, reasoning effort thresholds, summary inclusion
- **Timeouts**: Connection and read timeout configuration
- **Logging**: Configurable log levels

#### Override Rule Conditions

Override rules support various `when` conditions:
- `request.has_tool: "ToolName"` - matches requests containing specific tool calls
- `request.model_contains: "pattern"` - matches model names containing substring
- `request.model: "exact-name"` - exact model name matching
- `header.HeaderName: "value"` - HTTP header matching

### Key Features

- **Hot reload**: Configuration changes are picked up automatically
- **Streaming support**: Handles both streaming and non-streaming responses
- **Request translation**: Full conversion between Anthropic and OpenAI API formats
- **Reasoning effort mapping**: Maps token budgets to OpenAI reasoning effort levels
- **Error handling**: Proper HTTP status codes and error propagation

## Important Notes

- Plan mode detection is configured via override rules using `has_tool: "ExitPlanMode"` 
- Haiku/mini model requests are routed to cost-efficient OpenAI models via override rules
- All requests include request IDs for tracing and debugging
- Configuration supports environment variable substitution for API keys
- The router supports both streaming and non-streaming responses with proper error handling

## Code Organization

- `src/router/router.py` - Core routing decision logic and override rule processing
- `src/router/server.py` - FastAPI application and HTTP request handling  
- `src/router/adapters/` - API format translation between Anthropic and OpenAI
- `src/router/config/` - Configuration loading, validation, and hot reload
- `tests/` - Unit tests for routing logic and configuration

The `ModelRouter.decide_route()` method is the main entry point that processes override rules and returns routing decisions. Override rule matching happens in `_matches_override_condition()` with support for various condition types including tool call detection via `_has_tool_call()`.