"""Custom ChatOpenAI implementation that extracts custom fields like reasoning_content."""

from collections.abc import Mapping
from typing import Any, override

import openai
import structlog
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessageChunk,
)
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI

log = structlog.get_logger(__name__)

# Known standard OpenAI fields that are already handled by LangChain
# We skip these to avoid conflicts with standard processing
STANDARD_OPENAI_FIELDS = {
    # Standard message fields
    "content",
    "role",
    "name",
    "refusal",
    # Tool calling fields
    "tool_calls",
    "tool_call_id",
    "function_call",
    # Metadata fields that LangChain already processes
    "finish_reason",
    "index",
    "logprobs",
    # Delta-specific fields for streaming
    "delta",
    "usage",
}


class ChatOpenAIWithCustomFields(ChatOpenAI):
    """
    Custom ChatOpenAI that extracts custom fields from OpenAI responses.

    This class extends the standard ChatOpenAI to extract fields like reasoning_content
    from the raw OpenAI response and make them available in additional_kwargs.
    """

    def _extract_custom_fields_from_dict(
        self, data: dict[str, Any] | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Dynamically extract any custom fields from OpenAI response data."""
        custom_fields = {}

        # Try both streaming (delta) and non-streaming (message) paths
        candidate_paths = [
            ["choices", 0, "message"],  # Non-streaming path
            ["choices", 0, "delta"],  # Streaming path
        ]

        for path in candidate_paths:
            try:
                # Navigate to the target object (message or delta)
                target_obj: Any = data
                for key in path:
                    if isinstance(target_obj, dict) and key in target_obj:
                        target_obj = target_obj[key]
                    elif (
                        isinstance(target_obj, list)
                        and isinstance(key, int)
                        and key < len(target_obj)
                    ):
                        target_obj = target_obj[key]
                    else:
                        target_obj = None
                        break

                # If we found a valid target object, extract custom fields from it
                if isinstance(target_obj, dict):
                    for field_name, field_value in target_obj.items():
                        # Skip standard fields that LangChain already handles
                        if field_name in STANDARD_OPENAI_FIELDS:
                            continue

                        # Skip fields that are None or empty
                        if field_value is None or field_value == "":
                            continue

                        # Add the custom field if we haven't already found it
                        if field_name not in custom_fields:
                            custom_fields[field_name] = field_value

            except Exception as e:
                log.debug(
                    "Failed to explore path for custom fields",
                    path=path,
                    error=str(e),
                )
                continue

        return custom_fields

    def _add_custom_fields_to_message(
        self, chat_result: ChatResult, custom_fields: dict[str, Any], context: str
    ) -> None:
        """Add custom fields to the message's additional_kwargs."""
        if not custom_fields or not chat_result.generations:
            return

        generation = chat_result.generations[0]
        if isinstance(generation.message, AIMessage):
            # Merge custom fields into additional_kwargs
            if generation.message.additional_kwargs is None:
                generation.message.additional_kwargs = {}

            generation.message.additional_kwargs.update(custom_fields)

            log.info(
                "Added custom fields to message additional_kwargs",
                context=context,
                custom_fields=list(custom_fields.keys()),
                field_preview={k: str(v)[:50] for k, v in custom_fields.items()},
            )

    def _add_custom_fields_to_chunk(
        self, chunk: BaseMessageChunk, custom_fields: dict[str, Any], context: str
    ) -> None:
        """Add custom fields to a message chunk's additional_kwargs."""
        if not custom_fields or not isinstance(chunk, AIMessageChunk):
            return

        if chunk.additional_kwargs is None:
            chunk.additional_kwargs = {}

        chunk.additional_kwargs.update(custom_fields)

    @override
    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        """Override to extract custom fields and add them to additional_kwargs."""
        # First call the parent method to get the standard result
        chat_result = super()._create_chat_result(response, generation_info)

        # Extract custom fields from the raw response using shared method
        # Convert BaseModel to dict if needed
        response_dict = (
            response.model_dump()
            if isinstance(response, openai.BaseModel)
            else response
        )
        custom_fields = self._extract_custom_fields_from_dict(response_dict)

        # Add custom fields to the message's additional_kwargs if any were found
        self._add_custom_fields_to_message(
            chat_result, custom_fields, "non-streaming response"
        )

        return chat_result

    @override
    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        """Override to extract custom fields from Chat Completions API streaming chunks."""
        # Call parent method first to get the standard generation chunk
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )

        # Extract custom fields from the raw chunk using shared method
        if generation_chunk:
            custom_fields = self._extract_custom_fields_from_dict(chunk)

            # Add custom fields to the generation chunk using shared method
            if custom_fields and isinstance(generation_chunk.message, AIMessageChunk):
                self._add_custom_fields_to_chunk(
                    generation_chunk.message,
                    custom_fields,
                    "Chat Completions streaming",
                )

        return generation_chunk
