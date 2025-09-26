# Tool Filter Policy

This gateway can strip specific tools (for example, `WebSearch` or `WebFetch`) before routing a request downstream. The policy applies uniformly to every provider using it.

## Configuration

You can set policy globally via `Config.tools`, or override per provider using `ProviderConfig.tools`. `ToolPolicyConfig` currently supports a single field:

- `restricted_tool_names`: tool names to remove from the request (case-insensitive). Defaults to `["WebSearch", "WebFetch"]`.

Example `config/router.yaml` snippet with a per-provider override:

```
providers:
  openai:
    base_url: https://api.openai.com/v1
    adapter: openai
    tools:
      restricted_tool_names: ["WebSearch", "WebFetch"]

  anthropic:
    base_url: https://api.anthropic.com
    adapter: anthropic-passthrough
```

## How It Works

- The server applies tool filtering in `server.py` before routing to adapters. It uses the per-provider override (`ProviderConfig.tools`) when present, otherwise the global `Config.tools`.
- Any tool whose `name` matches (case-insensitive) an entry in `restricted_tool_names` is removed from the request prior to contacting the downstream provider.
- The official OpenAI adapter automatically appends the platform's built-in `web_search` tool when sending Requests API payloads. This ensures the model can launch first-party search even if user-supplied tool lists excluded it upstream.

## Testing

- Run: `uv run pytest -q`.
- See `tests/test_tool_filter.py` for unit coverage.
