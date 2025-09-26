# System Clause Filter

The router now supports removing provider-specific clauses from the top-level Anthropic
system prompt before forwarding a request downstream. This is useful when you need to
strip organizational policies or sensitive instructions that should not be visible to
the target model.

## Configuration

Add one or more clause filters under `system_prompt_filters` in `config/router.yaml`.
Filters are applied in order; each filter removes every match from the system prompt.

```yaml
system_prompt_filters:
  clause_filters:
    - pattern: '(?:\s*[,;])?\s*[^.;,]*\bdefensive security\b[^.;,]*'
      is_regex: true
    - pattern: '(?:\s*[,;])?\s*[^.;,]*\brefuse to\b[^.;,]*'
      is_regex: true
```

**Fields**

- `pattern` (required): literal text or regular expression to remove.
- `is_regex` (default `false`): when `true`, `pattern` is treated as a Python regex.
- `case_sensitive` (default `false`): perform case-sensitive matching when enabled.

If all text blocks in the system prompt are removed, the adapter omits the system
message entirely. Regex filters use `re.IGNORECASE` unless `case_sensitive` is set.

## Runtime Behavior

- Filters apply only to the top-level `system` field in Anthropic-style requests.
- Text content inside list blocks is filtered safely without mutating the original payload.
- Filtering now occurs in `server.py` before routing, ensuring both passthrough and LangChain/OpenAI paths receive sanitized input.
- Clause removal is conservative: the helper leaves surrounding whitespace untouched so callers can decide how to format the remaining text.

See `tests/test_system_clause_filter.py` for usage examples.
