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
uv run claude-router
```

Point Claude Code CLI to `http://localhost:8787` to use the router.

```bash
claudecode() {
    ANTHROPIC_BASE_URL="http://localhost:8787" \
    claude "$@"
}
```

## Configuration

Edit `config/router.yaml` to customize routing rules:

```yaml
overrides:
  # Route plan mode to reasoning model with custom config
  - when:
      request:
        user_regex: '<system-reminder>[\s\S]*plan mode[\s\S]*</system-reminder>'
    model: "openai/gpt-5?reasoning.effort=high&reasoning.summary=auto"
```

## Development

```bash
uv run pytest              # Tests
uv run mypy src/          # Type check  
uv run ruff check src/    # Lint
```
