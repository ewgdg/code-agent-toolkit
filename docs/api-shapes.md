API Shapes Reference (Anthropic ↔ OpenAI Responses ↔ LangChain)
==============================================================

Purpose
-------
Quick reference for request/response shapes relevant to: Anthropic Messages API thinking blocks, OpenAI Responses API reasoning items, and LangChain ChatOpenAI when `output_version='responses/v1'`.

Anthropic Messages API (thinking)
---------------------------------
- Request (enable thinking): top-level
  - `thinking: { "type": "enabled", "budget_tokens": <int> }`
- Response (assistant content blocks, ordered):
  - `{ "type": "thinking", "thinking": <string>, "signature": <opaque-string> }`
  - `{ "type": "text", "text": <string> }`
- Passing back thinking blocks in a later request (for tool-use/multi-turn):
  - Include the original `thinking` blocks exactly as returned (including `signature`).

OpenAI Responses API (core)
---------------------------
- Request (selected fields):
  - `model: <string>`
  - `input: string | [ { role: "user"|"assistant", content: [ { type: "input_text"|"input_image"|... } ] } ]`
  - `reasoning: { effort: "minimal"|"low"|"medium"|"high", summary?: "auto"|"concise"|"detailed" }` (o‑series)
  - `previous_response_id?: "resp_..."` (conversation continuity)
- Response (selected fields):
  - `id: "resp_..."`
  - `output: [ OutputMessage | FunctionToolCall | ... | ReasoningItem ]`
  - `usage: { input_tokens, output_tokens, output_tokens_details: { reasoning_tokens }, total_tokens }`
- Reasoning item (output):
  - Item type: `ReasoningItem` → commonly exposed with structure like:
    - `{ type: "reasoning", id: "rs_...", summary: [ { type: "output_text", text: <string> }, ... ], encrypted_content?: <opaque-string> }`
- Reasoning item (input):
  - You can carry forward prior reasoning via an input item. Preferred and fallback patterns:
    1) Preferred: `{ type: "reasoning", encrypted_content: <opaque-string> }`
    2) Fallback: `{ type: "reasoning", id: "rs_..." }`
  - These patterns are only valid for models that support reasoning and are sensitive to item ordering; malformed orders produce 400 errors.

LangChain ChatOpenAI (Responses mode)
------------------------------------
- Configure:
  - `ChatOpenAI(model=..., reasoning={...}, output_version="responses/v1")`
  - `output_version='responses/v1'` formats Responses API output items into AIMessage content blocks.
- Output (AIMessage.content, when `responses/v1`):
  - List of content blocks including:
    - `{ "type": "text", "text": <string> }`
    - `{ "type": "reasoning", "summary": [ { "text": <string> }, ... ] }`
  - Note: LangChain may not expose `id` for reasoning blocks by default.

Mapping Notes
-------------
- Anthropic `thinking` (private chain-of-thought) ≈ OpenAI `reasoning` (summary) in spirit, but:
  - Anthropic returns `thinking` blocks inside `content` (with `signature`).
  - OpenAI returns `reasoning` as separate output items (not part of message text), summarized.
- When targeting OpenAI Responses via LangChain, we attach the OpenAI reasoning id to the Anthropic thinking block under `extracted_openai_rs_id`. The plain `id` field is not set.
- For conversation continuity with OpenAI Responses, prefer `previous_response_id`.
- If emitting reasoning items in request input, ensure the target model supports reasoning and that item ordering/requirements are met.
