Streaming Reasoning: Preserving Encrypted Payload
=================================================

Problem
-------
When streaming Responses reasoning as Anthropic-style `thinking` blocks, attaching `extracted_openai_rs_encrypted_content` only inside a `content_block_delta` caused many SSE aggregators/clients to drop it. These clients typically merge only the documented delta fields (e.g., `thinking`) and ignore unknown keys in deltas.

Resolution
----------
- The response adapter now includes `extracted_openai_rs_encrypted_content` directly in the initial `content_block_start` for `thinking` blocks when present.
- To avoid duplication and maximize compatibility with conservative reconcilers, thinking deltas carry only the `thinking` text; the encrypted payload is omitted from deltas.

Implications
------------
- Non‑stream responses are unchanged; they already included the field in the final JSON.
- For streaming, any client that reconstructs the final message object from SSE events will retain the encrypted payload without custom merge logic.

Notes
-----
- This only applies when using the `openai` adapter with Responses API (`use_responses_api=true`).
- The request adapter reads `extracted_openai_rs_encrypted_content` from the prior assistant `thinking` block to prefer `{ type: "reasoning", encrypted_content }` on the next turn.
