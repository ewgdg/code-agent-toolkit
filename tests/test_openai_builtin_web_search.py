from src.claude_router.adapters.langchain_openai_request_adapter import (
    LangChainOpenAIRequestAdapter,
)
from src.claude_router.config.schema import Config, ProviderConfig
from src.claude_router.router import ModelRouter


def _basic_anthropic_request() -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello"}],
            }
        ],
        "stream": False,
    }


def test_openai_adapter_appends_builtin_web_search_tool():
    cfg = Config()
    router = ModelRouter(cfg)
    adapter = LangChainOpenAIRequestAdapter(cfg, router)

    anthropic_request = _basic_anthropic_request()
    provider = ProviderConfig(
        base_url="https://api.openai.com/v1",
        adapter="openai",
    )

    messages = adapter._convert_to_langchain_messages(
        anthropic_request, use_responses_api=True
    )

    prepared = adapter._prepare_openai_request(
        messages=messages,
        anthropic_request=anthropic_request,
        model="gpt-4.1",
        model_config=None,
        use_responses_api=True,
        support_reasoning=False,
        provider_config=provider,
    )

    assert {"type": "web_search"} in prepared["tools"]


def test_openai_compatible_adapter_does_not_append_builtin_web_search_tool():
    cfg = Config()
    router = ModelRouter(cfg)
    adapter = LangChainOpenAIRequestAdapter(cfg, router)

    anthropic_request = _basic_anthropic_request()
    provider = ProviderConfig(
        base_url="https://example.com/v1",
        adapter="openai-compatible",
    )

    messages = adapter._convert_to_langchain_messages(
        anthropic_request, use_responses_api=False
    )

    prepared = adapter._prepare_openai_request(
        messages=messages,
        anthropic_request=anthropic_request,
        model="gpt-4.1",
        model_config=None,
        use_responses_api=False,
        support_reasoning=False,
        provider_config=provider,
    )

    assert prepared.get("tools") == []
