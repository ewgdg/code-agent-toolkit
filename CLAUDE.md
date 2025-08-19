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
   - System prompt regex patterns (`system_regex: r"\bplan mode is (activated|triggered|on)\b"` for plan mode detection)
   - Model name regex patterns (`model_regex: r"haiku|mini"` for model matching)
   - Tool call detection (`has_tool: "ToolName"` for specific tool availability)
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
- `request.system_regex: "regex_pattern"` - matches system prompts using regex patterns (case-insensitive)
- `request.user_regex: "regex_pattern"` - matches user messages using regex patterns (case-insensitive)
- `request.model_regex: "regex_pattern"` - matches model names using regex patterns (case-insensitive)
- `request.has_tool: "ToolName"` - matches requests containing specific tool calls
- `header.HeaderName: "value"` - HTTP header matching

**Regex Examples:**
- `system_regex: r"\bplan mode is (activated|triggered|on)\b"` - matches plan mode activation
- `user_regex: r"<system-reminder>[\s\S]*\b(?:plan mode is (?:activated|active|triggered|on)|in plan mode)\b[\s\S]*</system-reminder>"` - matches system reminders with plan mode patterns
- `model_regex: r"^claude-3\.5-(sonnet|haiku)"` - matches Claude 3.5 models
- `model_regex: "opus"` - matches any model containing "opus"

### Key Features

- **Hot reload**: Configuration changes are picked up automatically
- **Streaming support**: Handles both streaming and non-streaming responses
- **Request translation**: Full conversion between Anthropic and OpenAI API formats
- **Reasoning effort mapping**: Maps token budgets to OpenAI reasoning effort levels
- **Error handling**: Proper HTTP status codes and error propagation

## Important Notes

- Plan mode detection is configured via override rules using `system_regex: r"\bplan mode is (activated|triggered|on)\b"` to detect when Claude Code enters plan mode 
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

The `ModelRouter.decide_route()` method is the main entry point that processes override rules and returns routing decisions. Override rule matching happens in `_matches_override_condition()` with support for various condition types including:
- System prompt regex analysis via `_extract_system_content()`
- User message regex analysis via `_extract_user_content()` (matches last user message only)
- Model regex matching with case-insensitive search and error handling
- Tool detection via `_has_tool()`