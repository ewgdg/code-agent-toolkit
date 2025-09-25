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

- Requests handled by `UnifiedLangChainAdapter` call `filter_tools_in_request(...)` before adapting or invoking the model.
- Any tool whose `name` matches (case-insensitive) an entry in `restricted_tool_names` is removed from the request prior to contacting the downstream provider.

## Testing

- Run: `uv run pytest -q`.
- See `tests/test_tool_filter.py` for unit coverage.
