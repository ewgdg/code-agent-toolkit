# Server-Side Filters

This project now applies request filtering centrally in `server.py` before routing
to any adapter. Two filter types are enforced:

- System prompt clause removal (`system_prompt_filters`)
- Tool restriction policy (`tools.restricted_tool_names` with provider overrides)

Key notes:
- Both OpenAI and Anthropic passthrough paths receive sanitized requests.
- For passthrough, the server re-encodes the request body after filtering.
- Adapters no longer perform this filtering; tests cover the server behavior.

## Tests

- `tests/test_server_filters.py::test_server_filters_tools_for_openai_adapter` — asserts
  the OpenAI route receives tool-stripped JSON at the adapter boundary.
- `tests/test_server_filters.py::test_server_filters_tools_for_passthrough_adapter` —
  asserts the passthrough path receives a re-encoded body with restricted tools removed.

See also: `docs/system-clause-filter.md` and `docs/tool-filter.md`.

