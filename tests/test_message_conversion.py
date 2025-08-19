import json

import pytest

from src.router.adapters.anthropic_openai_request_adapter import (
    AnthropicOpenAIRequestAdapter,
)
from src.router.config import Config
from src.router.router import ModelRouter


class TestMessageConversion:
    def setup_method(self):
        self.config = Config()
        self.router = ModelRouter(self.config)
        self.adapter = AnthropicOpenAIRequestAdapter(self.config, self.router)

    def test_convert_tool_use_to_function_call(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_123",
                        "name": "Search",
                        "input": {"query": "hello world"},
                    }
                ],
            }
        ]

        converted = self.adapter._convert_messages(messages)
        assert isinstance(converted, list) and len(converted) == 1

        # The adapter directly appends tool messages, not wrapped in role/content
        item = converted[0]
        assert item["type"] == "function_call"
        assert item["name"] == "Search"
        # arguments should be a JSON string
        args = item["arguments"]
        assert isinstance(args, str | dict)
        if isinstance(args, str):
            parsed = json.loads(args)
            assert parsed == {"query": "hello world"}
        else:
            # If passed through as dict in some edge, still equivalent
            assert args == {"query": "hello world"}
        # call_id should be carried over
        assert item.get("call_id") == "call_123"

    def test_convert_tool_result_to_function_call_output(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc",
                        "content": {"results": [1, 2, 3]},
                    }
                ],
            }
        ]

        converted = self.adapter._convert_messages(messages)
        assert len(converted) == 1

        # The adapter directly appends tool result messages, not wrapped in role/content
        item = converted[0]
        assert item["type"] == "function_call_output"
        assert item["call_id"] == "call_abc"
        # output should be a string (JSON-serialized)
        assert isinstance(item["output"], str)
        assert json.loads(item["output"]) == {"results": [1, 2, 3]}

    def test_mixed_text_and_tool_blocks(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will search now."},
                    {
                        "type": "tool_use",
                        "id": "call_tool",
                        "name": "Search",
                        "input": {"query": "docs"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here is the tool output:"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_tool",
                        "content": "Done",
                    },
                ],
            },
        ]

        converted = self.adapter._convert_messages(messages)
        # With the new policy, tool_use/tool_result each start a new message unless
        # the current one is empty. Thus we expect 4 items: assistant text message,
        # function_call, user text message, function_call_output.
        assert len(converted) == 4

        # First item: assistant text message
        m0 = converted[0]
        assert m0["role"] == "assistant"
        assert len(m0["content"]) == 1
        assert m0["content"][0]["type"] == "output_text"
        assert m0["content"][0]["text"] == "I will search now."

        # Second item: function_call (direct, not wrapped in message)
        m1 = converted[1]
        assert m1["type"] == "function_call"
        assert m1["name"] == "Search"

        # Third item: user text message
        m2 = converted[2]
        assert m2["role"] == "user"
        assert len(m2["content"]) == 1
        assert m2["content"][0]["type"] == "input_text"
        assert m2["content"][0]["text"] == "Here is the tool output:"

        # Fourth item: function_call_output (direct, not wrapped in message)
        m3 = converted[3]
        assert m3["type"] == "function_call_output"
        assert m3["call_id"] == "call_tool"
        assert m3["output"] == "Done"


@pytest.mark.asyncio
async def test_adapt_request_integration_includes_function_items():
    config = Config()
    router = ModelRouter(config)
    adapter = AnthropicOpenAIRequestAdapter(config, router)

    anthropic_request = {
        "model": "claude-3-sonnet",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "Lookup",
                        "input": {"k": "v"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_1", "content": 42}
                ],
            },
        ],
    }

    openai_request = await adapter.adapt_request(
        anthropic_request, target_model="gpt-4o"
    )
    assert "input" in openai_request
    items = openai_request["input"]
    # Ensure function_call and function_call_output are present
    # Items are direct tool calls/outputs, not wrapped in messages
    assert any(i.get("type") == "function_call" for i in items)
    assert any(i.get("type") == "function_call_output" for i in items)
