# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Claude Code Model Router - a proxy server that intelligently routes Claude Code CLI traffic between Anthropic's API and OpenAI's API based on configurable rules. It translates between Anthropic Messages API and OpenAI Chat Completions API formats.

## Development Commands

```bash
# Install dependencies
uv sync --dev

# Run the router server
claude-router --config router.yaml

# Run tests
pytest

# Type checking
mypy src/

# Linting and formatting
ruff check src/ tests/
ruff format src/ tests/
```

## Architecture

### Core Components

- **ProxyRouter** (`src/router/server.py`): Main FastAPI application that handles request routing
- **ModelRouter** (`src/router/router.py`): Decision engine that determines routing based on headers, request content, and configuration rules
- **Request Adapters** (`src/router/adapters/`): Translate between API formats
  - `AnthropicOpenAIRequestAdapter`: Converts Anthropic Messages API ’ OpenAI Chat Completions API
  - `OpenAIAnthropicResponseAdapter`: Converts OpenAI responses ’ Anthropic format
  - `PassthroughAdapter`: Direct forwarding to Anthropic API

### Routing Logic

The router makes decisions based on:
1. **Override rules** (highest precedence) - custom header/request conditions
2. **Plan mode detection** - ExitPlanMode tool calls or `X-Claude-Code-Mode: plan` header
3. **Model name matching** - haiku patterns route to background models
4. **Default passthrough** - everything else goes to Anthropic

### Configuration

All routing behavior is controlled by `router.yaml`:
- Model mappings (plan_model, background_model)
- Mode detection settings
- OpenAI reasoning effort thresholds
- Timeout configuration
- Override rules for custom routing

### Key Features

- **Hot reload**: Configuration changes are picked up automatically
- **Streaming support**: Handles both streaming and non-streaming responses
- **Request translation**: Full conversion between Anthropic and OpenAI API formats
- **Reasoning effort mapping**: Maps token budgets to OpenAI reasoning effort levels
- **Error handling**: Proper HTTP status codes and error propagation

## Important Notes

- The router detects plan mode by looking for `ExitPlanMode` tool calls in assistant messages
- Haiku model requests are automatically routed to cost-efficient OpenAI models
- All requests include request IDs for tracing
- Configuration supports environment variable substitution for API keys