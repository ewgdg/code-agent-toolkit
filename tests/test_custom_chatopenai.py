"""Tests for the custom ChatOpenAI class with custom field extraction."""

from src.claude_router.adapters.custom_chatopenai import (
    STANDARD_OPENAI_FIELDS,
    ChatOpenAIWithCustomFields,
)


class TestChatOpenAIWithCustomFields:
    """Test custom field extraction functionality."""

    def test_standard_fields_configuration(self):
        """Test that standard OpenAI fields are properly defined for exclusion."""
        # Ensure basic standard fields are excluded
        assert "content" in STANDARD_OPENAI_FIELDS
        assert "role" in STANDARD_OPENAI_FIELDS
        assert "tool_calls" in STANDARD_OPENAI_FIELDS
        assert "finish_reason" in STANDARD_OPENAI_FIELDS

        # Ensure the set is not empty
        assert len(STANDARD_OPENAI_FIELDS) > 0

    def test_extract_custom_fields_success(self):
        """Test successful extraction of custom fields while excluding standard fields."""
        chat_model = ChatOpenAIWithCustomFields(model="gpt-4", api_key="test")

        # Mock response with both standard and custom fields
        mock_response = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "This is the reasoning content",
                        "custom_metadata": {"model_temp": 0.7},
                        "content": "This is the main response",  # Standard field - should be excluded
                        "role": "assistant",  # Standard field - should be excluded
                    }
                }
            ]
        }

        custom_fields = chat_model._extract_custom_fields_from_dict(mock_response)

        # Should extract custom fields
        assert "reasoning_content" in custom_fields
        assert custom_fields["reasoning_content"] == "This is the reasoning content"
        assert "custom_metadata" in custom_fields
        assert custom_fields["custom_metadata"] == {"model_temp": 0.7}

        # Should NOT extract standard fields
        assert "content" not in custom_fields
        assert "role" not in custom_fields

    def test_extract_custom_fields_missing_field(self):
        """Test extraction when custom field is missing."""
        chat_model = ChatOpenAIWithCustomFields(model="gpt-4", api_key="test")

        # Mock response with only standard fields (no custom fields)
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "This is the main response",
                        "role": "assistant",
                        "finish_reason": "stop",
                    }
                }
            ]
        }

        custom_fields = chat_model._extract_custom_fields_from_dict(mock_response)

        assert "reasoning_content" not in custom_fields
        assert len(custom_fields) == 0

    def test_extract_custom_fields_invalid_structure(self):
        """Test extraction with invalid response structure."""
        chat_model = ChatOpenAIWithCustomFields(model="gpt-4", api_key="test")

        # Mock response with invalid structure
        mock_response = {"invalid": "structure"}

        custom_fields = chat_model._extract_custom_fields_from_dict(mock_response)

        assert len(custom_fields) == 0

    def test_create_chat_result_adds_custom_fields(self):
        """Test that _create_chat_result properly adds custom fields to additional_kwargs."""
        chat_model = ChatOpenAIWithCustomFields(model="gpt-4", api_key="test")

        # Create a valid OpenAI response format with custom fields
        response = {
            "id": "test-id",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "This is the main response",
                        "reasoning_content": "This is custom reasoning",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }

        result = chat_model._create_chat_result(response)

        # Check that custom fields were added to additional_kwargs
        assert "reasoning_content" in result.generations[0].message.additional_kwargs
        assert (
            result.generations[0].message.additional_kwargs["reasoning_content"]
            == "This is custom reasoning"
        )

    def test_extract_custom_fields_from_chunk_success(self):
        """Test extraction of custom fields from streaming chunk."""
        chat_model = ChatOpenAIWithCustomFields(model="gpt-4", api_key="test")

        # Mock streaming chunk with reasoning_content in delta (realistic format)
        mock_chunk = {
            "choices": [{"delta": {"reasoning_content": "Streaming reasoning content"}}]
        }

        custom_fields = chat_model._extract_custom_fields_from_dict(mock_chunk)

        assert "reasoning_content" in custom_fields
        assert custom_fields["reasoning_content"] == "Streaming reasoning content"

    def test_extract_custom_fields_multiple_fields(self):
        """Test extraction of multiple custom fields."""
        chat_model = ChatOpenAIWithCustomFields(model="gpt-4", api_key="test")

        # Mock response with multiple custom fields
        mock_response = {
            "choices": [
                {
                    "message": {
                        "reasoning_content": "This is reasoning",
                        "thinking_content": "This is thinking",
                        "metadata": "This is metadata",
                    }
                }
            ]
        }

        custom_fields = chat_model._extract_custom_fields_from_dict(mock_response)

        assert "reasoning_content" in custom_fields
        assert "thinking_content" in custom_fields
        assert "metadata" in custom_fields
        assert custom_fields["reasoning_content"] == "This is reasoning"
        assert custom_fields["thinking_content"] == "This is thinking"
        assert custom_fields["metadata"] == "This is metadata"
