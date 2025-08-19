import json
from collections.abc import AsyncIterator
from typing import Any

import structlog

from ..config import Config

logger = structlog.get_logger(__name__)


class OpenAIAnthropicResponseAdapter:
    def __init__(self, config: Config):
        self.config = config

    async def adapt_response(self, openai_response: dict[str, Any]) -> dict[str, Any]:
        """Translate OpenAI Responses API response to Anthropic format."""

        # OpenAI Responses API format: response has an 'output' array with items
        output_items = openai_response.get("output", [])

        # Build base anthropic response
        anthropic_response = {
            "id": openai_response.get("id", ""),
            "type": "message",
            "role": "assistant",
            "model": openai_response.get("model", ""),
            "content": [],
            "stop_reason": "end_turn",
            "usage": self._map_usage(openai_response.get("usage", {})),
        }

        # Process each output item
        for item in output_items:
            if not isinstance(item, dict):
                continue

            item_type = item.get("type")

            if item_type == "message":
                # Update response metadata from message item
                anthropic_response["id"] = item.get("id", anthropic_response["id"])
                status = item.get("status")
                anthropic_response["stop_reason"] = self._map_stop_reason(status)

                # Add annotations (web search results) if present
                annotations = item.get("annotations", [])
                if annotations:
                    annotation_id = f"srvtoolu_{item.get('id', 'unknown')}"
                    anthropic_response["content"].append({
                        "type": "server_tool_use",
                        "id": annotation_id,
                        "name": "web_search",
                        "input": {"query": ""},
                    })

                    search_results = []
                    for annotation in annotations:
                        url_citation = annotation.get("url_citation", {})
                        if url_citation:
                            search_results.append({
                                "type": "web_search_result",
                                "url": url_citation.get("url", ""),
                                "title": url_citation.get("title", ""),
                            })

                    if search_results:
                        anthropic_response["content"].append({
                            "type": "web_search_tool_result",
                            "tool_use_id": annotation_id,
                            "content": search_results,
                        })

                # Process message content
                for content_item in item.get("content", []):
                    content_type = content_item.get("type")

                    if content_type == "output_text":
                        anthropic_response["content"].append({
                            "type": "text",
                            "text": content_item.get("text", "")
                        })
                    elif content_type == "function_call":
                        # Convert OpenAI function_call to Anthropic tool_use
                        try:
                            arguments_str = content_item.get("arguments", "{}")
                            if isinstance(arguments_str, dict):
                                arguments = arguments_str
                            else:
                                arguments = json.loads(arguments_str)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            raw_args = content_item.get("arguments", "")
                            arguments = {"raw_arguments": raw_args}

                        anthropic_response["content"].append({
                            "type": "tool_use",
                            "id": content_item.get("call_id", ""),
                            "name": content_item.get("name", ""),
                            "input": arguments,
                        })
                        anthropic_response["stop_reason"] = "tool_use"

            elif item_type == "reasoning":
                # Add reasoning summary content (user-facing summary)
                for summary_part in item.get("summary", []):
                    is_dict = isinstance(summary_part, dict)
                    is_summary = summary_part.get("type") == "summary_text"
                    if is_dict and is_summary:
                        anthropic_response["content"].append({
                            "type": "text",
                            "text": summary_part.get("text", ""),
                        })

            elif item_type == "function_call":
                # Handle top-level function call items
                try:
                    arguments_str = item.get("arguments", "{}")
                    if isinstance(arguments_str, dict):
                        arguments = arguments_str
                    else:
                        arguments = json.loads(arguments_str)
                except (json.JSONDecodeError, TypeError, ValueError):
                    raw_args = item.get("arguments", "")
                    arguments = {"raw_arguments": raw_args}

                anthropic_response["content"].append({
                    "type": "tool_use",
                    "id": item.get("call_id", ""),
                    "name": item.get("name", ""),
                    "input": arguments,
                })
                anthropic_response["stop_reason"] = "tool_use"

        return anthropic_response

    def _map_stop_reason(self, openai_stop_reason: str | None) -> str:
        """Map OpenAI stop reason to Anthropic format."""

        mapping = {
            "completed": "end_turn",
            "stop": "end_turn",
            "length": "max_tokens",
            "content_filter": "stop_sequence",
            None: "end_turn",
        }

        return mapping.get(openai_stop_reason, "end_turn")

    def _map_usage(self, openai_usage: dict[str, Any]) -> dict[str, int]:
        """Map OpenAI usage to Anthropic format."""

        input_tokens = openai_usage.get("input_tokens", 0)
        output_tokens = openai_usage.get("output_tokens", 0)

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    async def adapt_stream(
        self, openai_stream: AsyncIterator[str]
    ) -> AsyncIterator[str]:
        """Convert OpenAI SSE stream to Anthropic format."""

        message_started = False
        content_block_index = 0
        current_block_type = None  # Track current block type: 'text', 'tool_use', etc.

        async for line in openai_stream:
            if not line.strip() or not line.startswith("data: "):
                continue

            data_str = line[6:]  # Remove "data: " prefix
            if data_str.strip() == "[DONE]":
                # Send message stop event
                yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
                break

            try:
                data = json.loads(data_str)
                event_type = data.get("type")

                if event_type == "response.created":
                    # Send message start event
                    if not message_started:
                        message_start = {
                            "type": "message_start",
                            "message": {
                                "id": data.get("id", ""),
                                "type": "message",
                                "role": "assistant",
                                "model": data.get("model", ""),
                                "content": [],
                                "stop_reason": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        }
                        yield f"data: {json.dumps(message_start)}\n\n"
                        message_started = True

                elif event_type == "response.output_text.delta":
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
                        "delta": {"type": "text_delta", "text": data.get("delta", "")},
                    }
                    yield f"data: {json.dumps(delta_event)}\n\n"

                elif event_type == "response.output_text.done":
                    # Send content block stop
                    if current_block_type == "text":
                        stop_event = {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        yield f"data: {json.dumps(stop_event)}\n\n"
                        content_block_index += 1
                        current_block_type = None

                elif event_type == "response.function_call.delta":
                    call_id = data.get("function_call", {}).get("call_id")

                    # Start new tool use block if not already started
                    if current_block_type != "tool_use":
                        tool_start = {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call_id,
                                "name": data.get("function_call", {}).get("name", ""),
                                "input": {},
                            },
                        }
                        yield f"data: {json.dumps(tool_start)}\n\n"
                        current_block_type = "tool_use"

                    # Send function call delta with proper error handling
                    partial_json = data.get("delta", "")
                    try:
                        delta_event = {
                            "type": "content_block_delta",
                            "index": content_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": partial_json,
                            },
                        }
                        yield f"data: {json.dumps(delta_event)}\n\n"
                    except (TypeError, UnicodeEncodeError):
                        # Fallback for malformed JSON
                        cleaned_json = partial_json.replace("\x00", "")
                        delta_event = {
                            "type": "content_block_delta",
                            "index": content_block_index,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": cleaned_json,
                            },
                        }
                        yield f"data: {json.dumps(delta_event)}\n\n"

                elif event_type == "response.function_call.done":
                    # Complete the tool call content block
                    if current_block_type == "tool_use":
                        stop_event = {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        yield f"data: {json.dumps(stop_event)}\n\n"
                        content_block_index += 1
                        current_block_type = None

                elif event_type == "reasoning.summary_text.delta":
                    # Start reasoning block if not already started
                    if current_block_type != "reasoning":
                        content_start = {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        yield f"data: {json.dumps(content_start)}\n\n"
                        current_block_type = "reasoning"

                    # Send reasoning summary delta
                    delta_event = {
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": {"type": "text_delta", "text": data.get("delta", "")},
                    }
                    yield f"data: {json.dumps(delta_event)}\n\n"

                elif event_type == "reasoning.summary_text.done":
                    # Send reasoning summary block stop
                    if current_block_type == "reasoning":
                        stop_event = {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        yield f"data: {json.dumps(stop_event)}\n\n"
                        content_block_index += 1
                        current_block_type = None

                elif event_type == "response.annotations.delta":
                    # Handle web search annotations
                    annotations = data.get("annotations", [])
                    for annotation in annotations:
                        url_citation = annotation.get("url_citation", {})
                        if url_citation:
                            # Add web search result content block
                            message_id = data.get("id", "unknown")
                            annotation_id = f"srvtoolu_{message_id}"
                            web_search_start = {
                                "type": "content_block_start",
                                "index": content_block_index,
                                "content_block": {
                                    "type": "web_search_tool_result",
                                    "tool_use_id": annotation_id,
                                    "content": [
                                        {
                                            "type": "web_search_result",
                                            "title": url_citation.get("title", ""),
                                            "url": url_citation.get("url", ""),
                                        }
                                    ],
                                },
                            }
                            yield f"data: {json.dumps(web_search_start)}\n\n"

                            # Immediately close the web search block
                            stop_event = {
                                "type": "content_block_stop",
                                "index": content_block_index,
                            }
                            yield f"data: {json.dumps(stop_event)}\n\n"
                            content_block_index += 1

                elif event_type == "response.completed":
                    # Send final usage and message stop
                    usage = data.get("usage", {})
                    if usage:
                        usage_event = {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": self._map_stop_reason(
                                    data.get("stop_reason")
                                ),
                                "usage": self._map_usage(usage),
                            },
                        }
                        yield f"data: {json.dumps(usage_event)}\n\n"

            except json.JSONDecodeError:
                logger.warning("Failed to parse OpenAI stream data", data=data_str)
                continue
