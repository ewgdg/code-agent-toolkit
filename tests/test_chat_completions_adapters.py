from unittest.mock import Mock

import pytest

from claude_router.adapters import (
    ChatCompletionsRequestAdapter,
    ChatCompletionsResponseAdapter,
)
from claude_router.config.schema import Config, ProviderConfig, TimeoutsConfig
from claude_router.router import ModelRouter


@pytest.fixture
def mock_config():
    """Create mock config for testing."""
    return Config()


@pytest.fixture
def provider_config():
    """Create provider config for testing."""
    return ProviderConfig(
        base_url="http://localhost:8080/v1",
        adapter="openai",
        api_key_env="TEST_API_KEY",
        timeouts_ms=TimeoutsConfig(connect=5000, read=30000),
    )


@pytest.fixture
def mock_router(mock_config):
    """Create mock router for testing."""
    return ModelRouter(mock_config)


class TestChatCompletionsRequestAdapter:
    """Test OpenAI Chat Completions request adapter."""

    def test_simple_message_conversion(self, mock_config, provider_config, mock_router):
        """Test conversion of simple Anthropic message to OpenAI format."""
        adapter = ChatCompletionsRequestAdapter(mock_config, mock_router)

        anthropic_request = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Hello, how are you?"}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }

        result = adapter.adapt_request(anthropic_request, provider_config, "llama3.1")

        assert result["model"] == "llama3.1"
        assert result["messages"] == [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1000

    def test_system_message_conversion(self, mock_config, provider_config, mock_router):
        """Test conversion of system messages."""
        adapter = ChatCompletionsRequestAdapter(mock_config, mock_router)

        anthropic_request = {
            "model": "claude-3-sonnet",
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        result = adapter.adapt_request(anthropic_request, provider_config, "llama3.1")

        assert len(result["messages"]) == 2
        assert result["messages"][0] == {
            "role": "system",
            "content": "You are a helpful assistant.",
        }
        assert result["messages"][1] == {"role": "user", "content": "Hello!"}

    def test_tool_use_conversion(self, mock_config, provider_config, mock_router):
        """Test conversion of tool use messages."""
        adapter = ChatCompletionsRequestAdapter(mock_config, mock_router)

        anthropic_request = {
            "model": "claude-3-sonnet",
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_123",
                            "name": "get_weather",
                            "input": {"city": "New York"},
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        }

        result = adapter.adapt_request(anthropic_request, provider_config, "llama3.1")

        # Check tools conversion
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0] == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }

        # Check message conversion
        assistant_message = result["messages"][1]
        assert "tool_calls" in assistant_message
        assert len(assistant_message["tool_calls"]) == 1
        assert assistant_message["tool_calls"][0] == {
            "id": "tool_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "New York"}'},
        }

    def test_reasoning_effort_conversion(
        self, mock_config, provider_config, mock_router
    ):
        """Test conversion of reasoning effort for supported models."""
        adapter = ChatCompletionsRequestAdapter(mock_config, mock_router)

        # Test with reasoning-capable model
        anthropic_request = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Hello!"}],
            "thinking": {"budget_tokens": 5000},
        }

        result = adapter.adapt_request(
            anthropic_request,
            provider_config,
            "o4",  # o4 supports reasoning
        )

        # Should include reasoning config
        assert "reasoning" in result
        # 5000 tokens maps to low (5000 is low_max)
        assert result["reasoning"]["effort"] == "low"
        assert result["reasoning"]["summary"] == "auto"  # non-minimal includes summary

    def test_reasoning_effort_not_added_for_non_reasoning_models(
        self, mock_config, provider_config, mock_router
    ):
        """Test that reasoning effort is not added for non-reasoning models."""
        adapter = ChatCompletionsRequestAdapter(mock_config, mock_router)

        anthropic_request = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Hello!"}],
            "thinking": {"budget_tokens": 5000},
        }

        result = adapter.adapt_request(
            anthropic_request,
            provider_config,
            "gpt-4",  # gpt-4 doesn't support reasoning
        )

        # Should not include reasoning config
        assert "reasoning" not in result

    def test_model_config_overrides(self, mock_config, provider_config, mock_router):
        """Test that model config overrides are applied."""
        adapter = ChatCompletionsRequestAdapter(mock_config, mock_router)

        anthropic_request = {
            "model": "claude-3-sonnet",
            "messages": [{"role": "user", "content": "Hello!"}],
            "temperature": 0.7,
        }

        model_config = {
            "temperature": 0.9,  # Override the request temperature
            "top_p": 0.8,  # Add new parameter
            "max_tokens": 2000,  # Add new parameter
        }

        result = adapter.adapt_request(
            anthropic_request, provider_config, "gpt-4", model_config
        )

        # Config overrides should be applied
        assert result["temperature"] == 0.9  # Overridden
        assert result["top_p"] == 0.8  # Added
        assert result["max_tokens"] == 2000  # Added


class TestChatCompletionsResponseAdapter:
    """Test Chat Completions to Anthropic response adapter."""

    def test_simple_response_conversion(self):
        """Test conversion of simple OpenAI response to Anthropic format."""
        adapter = ChatCompletionsResponseAdapter()

        # Create mock OpenAI ChatCompletion response
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 25

        mock_message = Mock()
        mock_message.content = "Hello! I'm doing well, thank you for asking."
        mock_message.tool_calls = None

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"

        openai_response = Mock()
        openai_response.id = "chatcmpl-123"
        openai_response.model = "llama3.1"
        openai_response.choices = [mock_choice]
        openai_response.usage = mock_usage

        result = adapter.adapt_response(openai_response)

        assert result["id"] == "chatcmpl-123"
        assert result["model"] == "llama3.1"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0] == {
            "type": "text",
            "text": "Hello! I'm doing well, thank you for asking.",
        }
        assert result["usage"] == {
            "input_tokens": 10,
            "output_tokens": 15,
            "total_tokens": 25,
        }

    def test_tool_call_response_conversion(self):
        """Test conversion of tool call responses."""
        adapter = ChatCompletionsResponseAdapter()

        # Create mock tool call
        mock_function = Mock()
        mock_function.name = "get_weather"
        mock_function.arguments = '{"city": "New York"}'

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = mock_function

        mock_usage = Mock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 25

        mock_message = Mock()
        mock_message.content = None
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        openai_response = Mock()
        openai_response.id = "chatcmpl-456"
        openai_response.model = "llama3.1"
        openai_response.choices = [mock_choice]
        openai_response.usage = mock_usage

        result = adapter.adapt_response(openai_response)

        assert result["stop_reason"] == "tool_use"
        assert len(result["content"]) == 1
        assert result["content"][0] == {
            "type": "tool_use",
            "id": "call_123",
            "name": "get_weather",
            "input": {"city": "New York"},
        }

    def test_finish_reason_mapping(self):
        """Test mapping of finish reasons."""
        adapter = ChatCompletionsResponseAdapter()

        test_cases = [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("content_filter", "stop_sequence"),
            ("tool_calls", "tool_use"),
            (None, "end_turn"),
            ("unknown", "end_turn"),
        ]

        for openai_reason, expected_anthropic in test_cases:
            result = adapter._map_stop_reason(openai_reason)
            assert result == expected_anthropic

    def test_usage_mapping(self):
        """Test usage statistics mapping."""
        adapter = ChatCompletionsResponseAdapter()

        usage = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

        result = adapter._map_usage(usage)

        assert result == {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}

    def test_empty_usage_mapping(self):
        """Test handling of empty usage."""
        adapter = ChatCompletionsResponseAdapter()

        result = adapter._map_usage({})

        assert result == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
