Reasoning Blocks and Message IDs
================================

Overview
--------
- For OpenAI Responses, we surface the OpenAI reasoning item id (rs_*) inside Anthropic thinking blocks under a unique field: `extracted_openai_rs_id`.
- On the next turn, the request adapter reads that field and emits an input reasoning reference for the Responses API.

Behavior
--------
- Request adapter (`LangChainOpenAIRequestAdapter`):
  - If the provider adapter is `openai` and the Responses API is used, an Anthropic `thinking` block is converted into an OpenAI Responses reasoning reference ONLY when `extracted_openai_rs_id` is present:
    - `{ "type": "reasoning", "id": <rs_id> }`
  - If no `extracted_openai_rs_id` is present, or for other providers / Chat Completions, the block degrades to visible text `<think>…</think>`.

- Response adapter (`LangChainOpenAIResponseAdapter`):
  - Top‑level `id` is generated (`msg_*`) for the Anthropic message envelope (not the OpenAI response id).
  - Any `thinking` blocks derived from OpenAI Responses reasoning output include `extracted_openai_rs_id: "rs_*"` when available; we do not set a plain `id`.
  - In streaming, thinking block starts also include `extracted_openai_rs_id` when available.

Notes
-----
- This behavior only applies for OpenAI Responses API requests (adapter: `openai`).
- If the underlying client does not surface an rs_* id, the thinking block will have no `extracted_openai_rs_id` and the next request will not emit a reasoning reference.
