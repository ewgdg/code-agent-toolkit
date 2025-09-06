from langchain_core.messages import AIMessage

from src.claude_router.adapters.langchain_openai_response_adapter import (
    LangChainOpenAIResponseAdapter,
)
from src.claude_router.adapters.langchain_openai_request_adapter import (
    LangChainOpenAIRequestAdapter,
)
from src.claude_router.config.schema import Config, ProviderConfig
from src.claude_router.router import ModelRouter


def test_response_adapter_surfaces_encrypted_reasoning_content():
    adapter = LangChainOpenAIResponseAdapter()

    enc = "ENC_REASONING_PAYLOAD_0123456789"
    message = AIMessage(
        content=[
            {
                "type": "reasoning",
                "id": "rs_123",
                "encrypted_content": enc,
                "summary": [{"type": "output_text", "text": "hidden chain"}],
            },
            {"type": "text", "text": "final answer"},
        ]
    )

    resp = adapter._non_stream_response(message, headers=None, use_responses_api=True)
    thinking = [
        b
        for b in resp["content"]
        if isinstance(b, dict) and b.get("type") == "thinking"
    ]
    assert thinking, "expected thinking block"
    assert thinking[0].get("extracted_openai_rs_encrypted_content") == enc


def test_request_adapter_prefers_encrypted_over_id_for_reasoning_item():
    cfg = Config()
    router = ModelRouter(cfg)
    req_adapter = LangChainOpenAIRequestAdapter(cfg, router)

    anthropic_request = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "id": "asst_1",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "",
                        "extracted_openai_rs_id": "rs_OLD",
                        "extracted_openai_rs_encrypted_content": "ENC_NEW",
                    },
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "doit",
                        "input": {"x": 1},
                    },
                ],
            },
        ],
        "stream": False,
    }

    provider = ProviderConfig(
        base_url="https://api.openai.com/v1", adapter="openai"
    )

    openai_req = req_adapter._prepare_openai_request(
        messages=req_adapter._convert_to_langchain_messages(
            anthropic_request, use_responses_api=True
        ),
        anthropic_request=anthropic_request,
        model="o4-mini",
        model_config=None,
        use_responses_api=True,
        support_reasoning=True,
        provider_config=provider,
    )

    # With the current design, `include` and `store` are bound on the model
    # instance. Assert the model carries those flags.
    lc_model = req_adapter._get_langchain_model(provider, "o4-mini")
    assert getattr(lc_model, "include", None) == ["reasoning.encrypted_content"]
    assert getattr(lc_model, "store", None) is False
