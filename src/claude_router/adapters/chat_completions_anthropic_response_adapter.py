import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.completion_usage import CompletionUsage

logger = structlog.get_logger(__name__)


class ChatCompletionsAnthropicResponseAdapter:
    """Adapter that translates OpenAI Chat Completions API responses to Anthropic format."""

    def __init__(self) -> None:
        pass

    def adapt_response(self, openai_response: ChatCompletion) -> dict[str, Any]:
        """Translate OpenAI Chat Completions response to Anthropic format."""

        # Build base Anthropic response
        anthropic_response: dict[str, Any] = {
            "id": openai_response.id,
            "type": "message",
            "role": "assistant",
            "model": openai_response.model,
            "content": [],
            "stop_reason": "end_turn",
            "usage": self._map_usage_from_sdk(openai_response.usage),
        }

        # Process choices (typically only one choice)
        if openai_response.choices:
            choice = openai_response.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            # Map finish reason to Anthropic format
            anthropic_response["stop_reason"] = self._map_stop_reason(finish_reason)

            # Process message content
            if message.content:
                anthropic_response["content"].append({
                    "type": "text",
                    "text": message.content
                })

            # Process tool calls if present
            if message.tool_calls:
                anthropic_response["stop_reason"] = "tool_use"
                for tool_call in message.tool_calls:
                    if tool_call.type == "function":
                        function = tool_call.function

                        # Parse arguments
                        try:
                            arguments = json.loads(function.arguments) if function.arguments else {}
                        except (json.JSONDecodeError, TypeError):
                            arguments = {"raw_arguments": function.arguments}

                        anthropic_response["content"].append({
                            "type": "tool_use",
                            "id": tool_call.id,
                            "name": function.name,
                            "input": arguments
                        })

        return anthropic_response

    def _map_stop_reason(self, finish_reason: str | None) -> str:
        """Map OpenAI finish reason to Anthropic format."""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "stop_sequence",
            "tool_calls": "tool_use",
            None: "end_turn",
        }
        return mapping.get(finish_reason, "end_turn")

    def _map_usage_from_sdk(self, usage: CompletionUsage | None) -> dict[str, int]:
        """Map OpenAI SDK usage object to Anthropic format."""
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        prompt_tokens = usage.prompt_tokens or 0
        completion_tokens = usage.completion_tokens or 0
        total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)

        return {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _map_usage(self, usage: dict[str, Any]) -> dict[str, int]:
        """Map OpenAI usage dict to Anthropic format (for backward compatibility)."""
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", prompt_tokens + completion_tokens)

        return {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    async def adapt_stream(
        self,
        openai_stream: AsyncStream[ChatCompletionChunk]
    ) -> AsyncIterator[str]:
        """Convert OpenAI Chat Completions streaming response to Anthropic format."""

        message_started = False
        content_block_index = 0
        current_block_type = None
        accumulated_tool_calls = {}

        async for chunk in openai_stream:
            # Extract metadata from first chunk
            if not message_started:
                message_id = chunk.id or ""
                model = chunk.model or ""

                # Send message start event
                message_start = {
                    "type": "message_start",
                    "message": {
                        "id": message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
                yield f"data: {json.dumps(message_start)}\n\n"
                message_started = True

            # Process choices
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason

            # Handle content delta
            if delta.content:
                # Start text content block if not already started
                if current_block_type != "text":
                    content_start = {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"data: {json.dumps(content_start)}\n\n"
                    current_block_type = "text"

                # Send content block delta
                delta_event = {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "text_delta", "text": delta.content},
                }
                yield f"data: {json.dumps(delta_event)}\n\n"

            # Handle tool calls
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    tool_id = tool_call.id

                    # Initialize tool call if needed
                    if tool_id and tool_id not in accumulated_tool_calls:
                        # End current content block if needed
                        if current_block_type == "text":
                            content_stop = {
                                "type": "content_block_stop",
                                "index": content_block_index,
                            }
                            yield f"data: {json.dumps(content_stop)}\n\n"
                            content_block_index += 1

                        function = tool_call.function
                        if function:
                            accumulated_tool_calls[tool_id] = {
                                "name": function.name or "",
                                "arguments": function.arguments or ""
                            }

                            # Start tool use content block
                            content_start = {
                                "type": "content_block_start",
                                "index": content_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": function.name or "",
                                    "input": {}
                                },
                            }
                            yield f"data: {json.dumps(content_start)}\n\n"
                            current_block_type = "tool_use"

                    # Update arguments if present
                    elif tool_id and tool_id in accumulated_tool_calls:
                        function = tool_call.function
                        if function and function.arguments:
                            accumulated_tool_calls[tool_id]["arguments"] += function.arguments

            # Handle finish reason
            if finish_reason:
                # End current content block
                if current_block_type == "text":
                    content_stop = {
                        "type": "content_block_stop",
                        "index": content_block_index,
                    }
                    yield f"data: {json.dumps(content_stop)}\n\n"
                elif current_block_type == "tool_use":
                    # Process accumulated tool calls
                    for tool_id, tool_data in accumulated_tool_calls.items():
                        try:
                            arguments = json.loads(tool_data["arguments"])
                        except json.JSONDecodeError:
                            arguments = {"raw_arguments": tool_data["arguments"]}

                        # Send tool use delta with final arguments
                        delta_event = {
                            "type": "content_block_delta",
                            "index": content_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(arguments)
                            },
                        }
                        yield f"data: {json.dumps(delta_event)}\n\n"

                    content_stop = {
                        "type": "content_block_stop",
                        "index": content_block_index,
                    }
                    yield f"data: {json.dumps(content_stop)}\n\n"

                # Send message delta with stop reason and usage
                message_delta = {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self._map_stop_reason(finish_reason),
                    },
                }
                if chunk.usage:
                    message_delta["usage"] = self._map_usage_from_sdk(chunk.usage)

                yield f"data: {json.dumps(message_delta)}\n\n"

                # Send message stop
                message_stop = {
                    "type": "message_stop"
                }
                yield f"data: {json.dumps(message_stop)}\n\n"
                break
