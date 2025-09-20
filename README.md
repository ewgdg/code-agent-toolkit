# Claude Code Model Router

Intelligent proxy that routes Claude Code CLI traffic between Anthropic and OpenAI APIs with automatic format translation.

## Features

- Smart routing based on configurable rules (plan mode, model patterns, etc.)
- Seamless API translation between Anthropic Messages and OpenAI Chat Completions
- Hot configuration reload and streaming support
- **Optimized for reasoning models** with automatic Responses API usage and reasoning workflow support

## Reasoning Model Optimization

This router is specifically optimized for OpenAI's reasoning models (o1, o3, etc.) and follows the best practices from the [OpenAI Cookbook](https://cookbook.openai.com/examples/responses_api/reasoning_items?utm_source=openai):

- **Responses API Integration**: Automatically uses OpenAI's Responses API for official OpenAI reasoning models to capture reasoning traces
- **Reasoning Workflow Support**: For third-party APIs that only support Chat Completions endpoints, the router feeds reasoning items through the standard chat interface. Following the cookbook pattern, only reasoning items from the final round of tool calling are included as assistant messages, ensuring efficient reasoning trace integration without overwhelming the context window
- **Automatic Reasoning Conversion**: Converts Claude Code requests to reasoning-optimized formats with configurable effort levels and summary inclusion
- **Effort Level Mapping**: Intelligently maps token budgets and request complexity to appropriate reasoning effort levels (low, medium, high)

The router seamlessly handles both official OpenAI APIs (with full Responses API support) and third-party reasoning-compatible models through unified reasoning workflows.

## Quick Start

```bash
# Install
uv sync

# Configure API key
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run
uv run claude-router --config config/router.yaml
```

Point Claude Code CLI to `http://localhost:8787` to use the router.

```bash
claudecode() {
    ANTHROPIC_BASE_URL="http://localhost:8787" \
    claude "$@"
}
```

## Configuration

Edit `config/router.yaml` to customize routing rules and providers:

### Provider Configuration

```yaml
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    adapter: "openai"  # Official OpenAI API with Responses API support
    api_key_env: "OPENAI_API_KEY"
  llama:
    base_url: "http://127.0.0.1:8080/v1"
    adapter: "openai-compatible"  # Third-party OpenAI-compatible API
```

**Adapter Types:**
- `anthropic-passthrough`: Direct forwarding to Anthropic API
- `openai`: Official OpenAI API with Responses API support
- `openai-compatible`: Third-party APIs (llama.cpp, Ollama, LocalAI, etc.)

### Override Rules

```yaml
overrides:
  # Route plan mode to reasoning model with custom config
  - when:
      request:
        user_regex: '<system-reminder>[\s\S]*plan mode[\s\S]*</system-reminder>'
    provider: "openai"
    model: "gpt-5"
    config:
      reasoning:
        effort: "high"
        summary: "auto"
```

## Default Behavior

### Reasoning Conversion

If no reasoning config is specified for a request and the selected OpenAI model supports reasoning, the router applies minimal reasoning conversion by default.

## Development

```bash
# Tests
uv run pytest
# Type check
uv run pyright src/
# Lint
uv run ruff format src/
uv run ruff check src/
```
