# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Claude Code Model Router - a proxy server that intelligently routes Claude Code CLI traffic between Anthropic's API and OpenAI's API based on configurable rules. It translates between Anthropic Messages API and OpenAI Chat Completions API formats.

Default server runs on `http://localhost:8787`. Point Claude Code CLI to this URL:

```bash
claudecode() {
    ANTHROPIC_BASE_URL="http://localhost:8787" \
    claude "$@"
}
```

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
uv run pyright src/

# Linting and formatting, try formatting first
uv run ruff format src/ tests/
uv run ruff check src/ tests/

# Fix auto-fixable linting issues
uv run ruff check src/ tests/ --fix
```

## Architecture

### Core Components

- **ProxyRouter** (`src/claude_router/server.py`): Main FastAPI application that handles request routing and adapter dispatching
- **ModelRouter** (`src/claude_router/router.py`): Decision engine that determines routing based on headers, request content, and configuration rules
- **Provider System** (`src/claude_router/config/schema.py`): Configurable provider definitions with base URLs, adapters, and API keys
- **Request/Response Adapters** (`src/claude_router/adapters/`): Translate between API formats
  - `PassthroughAdapter`: Direct forwarding to Anthropic API
  - `UnifiedLangChainAdapter`: Unified LangChain-based adapters for OpenAI compatibility

### Routing Logic

The router makes decisions based on:

1. **Override rules** (highest precedence) - custom header/request conditions including:
   - System prompt regex patterns (`system_regex: r"\bplan mode is (activated|triggered|on)\b"` for plan mode detection)
   - Model name regex patterns (`model_regex: r"haiku|mini"` for model matching)
   - Tool call detection (`has_tool: "ToolName"` for specific tool availability)
   - Header matching
   - Request data conditions
   - **Provider specification**: Optional `provider` field in rules to explicitly route to configured providers
2. **Provider prefix parsing** - Models with prefixes like `openai/gpt-4` or `anthropic/claude-3` are routed to matching providers
3. **Default passthrough** - everything else goes to Anthropic

Override rules are processed in order, with the first matching rule taking precedence. Provider resolution follows: explicit rule provider > parsed provider prefix > "anthropic" default.

### Configuration

All routing behavior is controlled by `router.yaml`:

- **Provider definitions**: `providers` section defining custom API endpoints with adapters (OpenAI, llama.cpp, Ollama, etc.)
- **Override rules**: Flexible routing conditions with `when` clauses and optional `provider` targeting
- **OpenAI settings**: API key, reasoning effort thresholds, summary inclusion
- **Timeouts**: Global and provider-specific connection and read timeout configuration
- **Logging**: Configurable log levels

#### Override Rule Conditions

Override rules support various `when` conditions:

- `request.system_regex: "regex_pattern"` - matches system prompts using regex patterns (case-insensitive)
- `request.user_regex: "regex_pattern"` - matches user messages using regex patterns (case-insensitive)
- `request.model_regex: "regex_pattern"` - matches model names using regex patterns (case-insensitive)
- `request.has_tool: "ToolName"` - matches requests containing specific tool calls
- `header.HeaderName: "value"` - HTTP header matching

**Provider Configuration Example:**

```yaml
providers:
  openai:
    base_url: "https://api.openai.com/v1"
    adapter: "openai"
    api_key_env: "OPENAI_API_KEY"
  llama-local:
    base_url: "http://localhost:8080/v1"
    adapter: "openai"
    api_key_env: "LLAMA_API_KEY"  # optional
    timeouts_ms:
      connect: 3000
      read: 120000
```

**Regex Examples:**

- `system_regex: r"\bplan mode is (activated|triggered|on)\b"` - matches plan mode activation
- `user_regex: r"<system-reminder>[\s\S]*\b(?:plan mode is (?:activated|active|triggered|on)|in plan mode)\b[\s\S]*</system-reminder>"` - matches system reminders with plan mode patterns
- `model_regex: r"^claude-3\.5-(sonnet|haiku)"` - matches Claude 3.5 models
- `model_regex: "opus"` - matches any model containing "opus"

#### Model Configuration Overrides

Override rules can include a `config` section to modify model parameters with conditional logic. This uses the `ModelConfigEntry` system with `when` conditions to determine when overrides should be applied.

**When Condition Types:**

- `current_in: [val1, val2, null]` - Apply if current value is in the specified list (supports `null` for None/unset values)
- `current_not_in: [val1, val2]` - Apply if current value is NOT in the specified list  
- `current_equals: value` - Apply if current value equals the specified value
- `current_not_equals: value` - Apply if current value does NOT equal the specified value

If no `when` conditions are specified, the override is always applied.

**Configuration Override Examples:**

```yaml
overrides:
  # Route plan mode with conditional reasoning effort
  - when:
      request:
        user_regex: '<system-reminder>.*plan mode.*</system-reminder>'
    provider: "openai"
    model: "gpt-5"
    config:
      reasoning:
        effort: 
          value: "medium"
          when:
            current_in: [null, "low", "minimum"]  # Only upgrade if unset, low, or minimum
        summary: "auto"  # Always apply (no conditions)
      temperature: 0.7  # Always apply (no conditions)

  # High-performance model with conditional overrides  
  - when:
      request:
        model_regex: '^gpt-high$'
    model: "gpt-5"
    provider: "openai"
    config:
      reasoning:
        effort: 
          value: "high"
          when:
            current_not_in: ["high", "maximum"]  # Apply unless already high/maximum
      max_tokens:
        value: 4000
        when:
          current_equals: null  # Only set if unset
```

**YAML Dict Format (Alternative):**

```yaml
config:
  temperature:
    value: 0.8
    when:
      current_not_equals: 0.8  # Apply unless already 0.8
  reasoning:
    effort: "medium"  # No conditions = always apply
```

### Key Features

- **Hot reload**: Configuration changes are picked up automatically
- **Multi-adapter support**: Two adapter types for different API compatibility levels
- **Streaming support**: Handles both streaming and non-streaming responses
- **Request translation**: Full conversion between Anthropic and OpenAI API formats
- **Reasoning effort mapping**: Maps token budgets to OpenAI reasoning effort levels
- **Provider flexibility**: Support for any OpenAI-compatible API (llama.cpp, Ollama, LocalAI)
- **Error handling**: Proper HTTP status codes and error propagation

## Important Notes

- **Adapter Types**: Available adapter types:
  - `anthropic-passthrough`: Direct forwarding to Anthropic API
  - `openai`: Unified LangChain-based adapter for OpenAI compatibility
- Plan mode detection is configured via override rules using `system_regex: r"\bplan mode is (activated|triggered|on)\b"` to detect when Claude Code enters plan mode
- Provider resolution precedence: explicit rule provider > model prefix parsing > "anthropic" default
- All requests include request IDs for tracing and debugging
- Configuration supports environment variable substitution for API keys
- The router supports both streaming and non-streaming responses with proper error handling

## Code Organization

- `src/claude_router/router.py` - Core routing decision logic and override rule processing
- `src/claude_router/server.py` - FastAPI application and HTTP request handling  
- `src/claude_router/adapters/` - API format translation between Anthropic and OpenAI
- `src/claude_router/config/` - Configuration loading, validation, and hot reload
- `tests/` - Unit tests for routing logic and configuration

The `ModelRouter.decide_route()` method is the main entry point that processes override rules and returns `RouterDecision` objects with provider, adapter, and model information. Override rule matching happens in `_matches_override_condition()` with support for various condition types including:

- System prompt regex analysis via `_extract_system_content()`
- User message regex analysis via `_extract_user_content()` (matches last user message only)
- Model regex matching with case-insensitive search and error handling
- Tool detection via `_has_tool()`
- Provider prefix parsing via `_parse_provider_model()`

The server dispatches requests to different handlers based on the adapter type:

- `_handle_passthrough_request()` for Anthropic passthrough
- `_handle_langchain_openai_request()` for LangChain-based OpenAI compatibility

Model configuration overrides are processed by:

- `_apply_granular_config_overrides()` applies configuration overrides with when condition evaluation
- `_should_apply_config_override()` evaluates when conditions (current_in, current_not_in, current_equals, current_not_equals)
- `WhenCondition` schema class defines the available condition types for ModelConfigEntry objects

## Typing Rules
- Always do typing.
- Prefer `T1|T2` over `Union[T1,T2]`
- Avoid `Any` if possible
