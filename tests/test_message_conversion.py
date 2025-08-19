import json

import pytest

from src.router.config import Config
from src.router.router import ModelRouter
from src.router.adapters.anthropic_openai_request_adapter import (
    AnthropicOpenAIRequestAdapter,
)


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
        msg = converted[0]

        # Role should be forced to assistant for tool_use
        assert msg["role"] == "assistant"
        assert isinstance(msg["content"], list) and len(msg["content"]) == 1

        item = msg["content"][0]
        assert item["type"] == "function_call"
        assert item["name"] == "Search"
        # arguments should be a JSON string
        args = item["arguments"]
        assert isinstance(args, (str, dict))
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
        msg = converted[0]

        # Role should be set to user for tool_result
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1

        item = msg["content"][0]
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
        # With the new policy, tool_use/tool_result each start a new message unless the current one is empty.
        # Thus we expect 4 messages: assistant text, assistant function_call, user text, user function_call_output.
        assert len(converted) == 4

        # First message: assistant text only
        m0 = converted[0]
        assert m0["role"] == "assistant"
        assert len(m0["content"]) == 1
        assert m0["content"][0]["type"] == "output_text"
        assert m0["content"][0]["text"] == "I will search now."

        # Second message: assistant function_call only
        m1 = converted[1]
        assert m1["role"] == "assistant"
        assert len(m1["content"]) == 1
        assert m1["content"][0]["type"] == "function_call"
        assert m1["content"][0]["name"] == "Search"

        # Third message: user text only
        m2 = converted[2]
        assert m2["role"] == "user"
        assert len(m2["content"]) == 1
        assert m2["content"][0]["type"] == "input_text"
        assert m2["content"][0]["text"] == "Here is the tool output:"

        # Fourth message: user function_call_output only
        m3 = converted[3]
        assert m3["role"] == "user"
        assert len(m3["content"]) == 1
        assert m3["content"][0]["type"] == "function_call_output"
        assert m3["content"][0]["call_id"] == "call_tool"
        assert m3["content"][0]["output"] == "Done"


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
    assert any(i for i in items[0]["content"] if i["type"] == "function_call")
    assert any(i for i in items[1]["content"] if i["type"] == "function_call_output")
