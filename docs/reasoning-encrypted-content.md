Reasoning Encrypted Content – Test Update

Context
- The OpenAI Responses API flags `include=["reasoning.encrypted_content"]` and `store=False` are now configured when instantiating the LangChain `ChatOpenAI` model inside the request adapter, not as per‑call params.

Changes
- Updated `tests/test_reasoning_encrypted_content.py::test_request_adapter_prefers_encrypted_over_id_for_reasoning_item` to:
  - Stop passing `provider_config` into `_convert_to_langchain_messages` (method doesn’t accept it).
  - Retrieve the model via `_get_langchain_model` and assert the model carries `include=["reasoning.encrypted_content"]` and `store=False`.

Rationale
- These flags are connection/model configuration, not per‑request knobs in the current design; asserting at model level better reflects behavior and avoids duplication.

Result
- Test suite passes: 58 passed.
