# Claude Code Model Router

Intelligent proxy that routes Claude Code CLI traffic between Anthropic and OpenAI APIs with automatic format translation.

## Features

- Smart routing based on configurable rules (plan mode, model patterns, etc.)
- Seamless API translation between Anthropic Messages and OpenAI Chat Completions
- Hot configuration reload and streaming support

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
