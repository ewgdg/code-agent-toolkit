import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.claude_router.adapters.langchain_openai_request_adapter import (
    LangChainOpenAIRequestAdapter,
)
from src.claude_router.adapters.prompt_filter import filter_system_prompt_in_request
from src.claude_router.config.schema import (
    ClauseFilterRule,
    Config,
    SystemPromptFilterConfig,
)
from src.claude_router.router import ModelRouter


def _build_adapter_with_filters(*filters: ClauseFilterRule) -> LangChainOpenAIRequestAdapter:
    cfg = Config(
        system_prompt_filters=SystemPromptFilterConfig(clause_filters=filters)
    )
    router = ModelRouter(cfg)
    return LangChainOpenAIRequestAdapter(cfg, router)


def test_clause_filter_removes_sentence_from_string_system_prompt() -> None:
    adapter = _build_adapter_with_filters(
        ClauseFilterRule(pattern="Remove this sentence.")
    )

    anthropic_request = {
        "system": "Keep this. Remove this sentence.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            }
        ],
    }

    filter_system_prompt_in_request(
        anthropic_request, adapter.config.system_prompt_filters
    )

    assert anthropic_request["system"].strip().startswith("Keep this.")
    messages = adapter._convert_to_langchain_messages(
        anthropic_request, use_responses_api=True
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content[0]["text"].strip() == "Keep this."


def test_clause_filter_removes_system_message_when_empty() -> None:
    adapter = _build_adapter_with_filters(ClauseFilterRule(pattern="Remove me"))

    anthropic_request = {
        "system": "Remove me",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            }
        ],
    }

    filter_system_prompt_in_request(
        anthropic_request, adapter.config.system_prompt_filters
    )
    assert "system" not in anthropic_request
    messages = adapter._convert_to_langchain_messages(
        anthropic_request, use_responses_api=True
    )

    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == [{"type": "text", "text": "hi"}]


def test_clause_filter_supports_regex_on_structured_system_prompt() -> None:
    adapter = _build_adapter_with_filters(
        ClauseFilterRule(pattern=r";?\s*drop please\.", is_regex=True)
    )

    anthropic_request = {
        "system": [
            {
                "type": "text",
                "text": "Keep this clause; drop please.",
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
            }
        ],
    }

    filter_system_prompt_in_request(
        anthropic_request, adapter.config.system_prompt_filters
    )
    assert isinstance(anthropic_request["system"], list)
    messages = adapter._convert_to_langchain_messages(
        anthropic_request, use_responses_api=True
    )

    assert isinstance(messages[0], SystemMessage)
    assert messages[0].content[0]["text"].strip() == "Keep this clause"


def test_clause_filter_precompiles_regex_pattern() -> None:
    regex_rule = ClauseFilterRule(pattern=r"foo|bar", is_regex=True)
    compiled = regex_rule.compiled_pattern
    assert isinstance(compiled, re.Pattern)

    adapter = _build_adapter_with_filters(regex_rule)
    rule = adapter.config.system_prompt_filters.clause_filters[0]
    compiled_in_config = rule.compiled_pattern
    assert isinstance(compiled_in_config, re.Pattern)
    assert compiled_in_config.pattern == compiled.pattern
    assert compiled_in_config.flags == compiled.flags
