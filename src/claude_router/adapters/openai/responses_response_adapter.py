import json
from collections.abc import AsyncIterator
from typing import Any

import structlog
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionCallArgumentsDoneEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputTextAnnotationAddedEvent,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
    ResponseTextDoneEvent,
    ResponseUsage,
)
from openai.types.responses.response_output_text import AnnotationURLCitation

from ...config import Config

logger = structlog.get_logger(__name__)


class ResponsesResponseAdapter:
    def __init__(self, config: Config):
        self.config = config

    async def adapt_response(self, openai_response: Response) -> dict[str, Any]:
        """Translate OpenAI Responses API response to Anthropic format."""

        # Build base anthropic response
        anthropic_response: dict[str, Any] = {
            "id": openai_response.id or "",
            "type": "message",
            "role": "assistant",
            "model": openai_response.model or "",
            "content": [],
            "stop_reason": "end_turn",
            "usage": self._map_usage(openai_response.usage),
        }

        # Process output items
        output_items = openai_response.output or []
        for item in output_items:
            if isinstance(item, ResponseOutputMessage):
                # Extract metadata
                anthropic_response["id"] = item.id or anthropic_response["id"]
                anthropic_response["stop_reason"] = self._map_stop_reason(item.status)

                # Process message content
                for content_item in item.content or []:
                    if content_item.type == "output_text":
                        anthropic_response["content"].append(
                            {"type": "text", "text": content_item.text or ""}
                        )

                        # Handle annotations from output_text
                        if content_item.annotations:
                            annotation_id = f"srvtoolu_{item.id or 'unknown'}"
                            anthropic_response["content"].append(
                                {
                                    "type": "server_tool_use",
                                    "id": annotation_id,
                                    "name": "web_search",
                                    "input": {"query": ""},
                                }
                            )

                            search_results = []
                            for annotation in content_item.annotations:
                                if isinstance(annotation, AnnotationURLCitation):
                                    search_results.append(
                                        {
                                            "type": "web_search_result",
                                            "url": annotation.url or "",
                                            "title": annotation.title or "",
                                        }
                                    )

                            if search_results:
                                anthropic_response["content"].append(
                                    {
                                        "type": "web_search_tool_result",
                                        "tool_use_id": annotation_id,
                                        "content": search_results,
                                    }
                                )

            elif isinstance(item, ResponseReasoningItem):
                # Add reasoning summary content (user-facing summary)
                for summary_part in item.summary or []:
                    anthropic_response["content"].append(
                        {
                            "type": "text",
                            "text": summary_part.text or "",
                        }
                    )

            elif isinstance(item, ResponseFunctionToolCall):
                # Handle function call items
                try:
                    arguments_str = item.arguments
                    if isinstance(arguments_str, dict):
                        arguments = arguments_str
                    else:
                        arguments = json.loads(arguments_str or "{}")
                except (json.JSONDecodeError, TypeError, ValueError):
                    raw_args = item.arguments or ""
                    arguments = {"raw_arguments": raw_args}

                anthropic_response["content"].append(
                    {
                        "type": "tool_use",
                        "id": item.call_id or "",
                        "name": item.name or "",
                        "input": arguments,
                    }
                )
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

    def _map_usage(self, openai_usage: ResponseUsage | None) -> dict[str, int]:
        """Map OpenAI usage to Anthropic format."""

        if openai_usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        input_tokens = openai_usage.input_tokens or 0
        output_tokens = openai_usage.output_tokens or 0

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    async def adapt_stream(
        self, openai_stream: AsyncIterator[ResponseStreamEvent]
    ) -> AsyncIterator[str]:
        """Convert OpenAI streaming response events to Anthropic format."""

        message_started = False
        content_block_index = 0
        current_block_type = None  # Track current block type: 'text', 'tool_use', etc.

        async for event in openai_stream:
            try:
                if isinstance(event, ResponseCreatedEvent):
                    # Send message start event
                    if not message_started:
                        message_start = {
                            "type": "message_start",
                            "message": {
                                "id": getattr(event, "id", ""),
                                "type": "message",
                                "role": "assistant",
                                "model": getattr(event, "model", ""),
                                "content": [],
                                "stop_reason": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0},
                            },
                        }
                        yield f"data: {json.dumps(message_start)}\n\n"
                        message_started = True

                elif isinstance(event, ResponseTextDeltaEvent):
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
                        "delta": {
                            "type": "text_delta",
                            "text": getattr(event, "delta", ""),
                        },
                    }
                    yield f"data: {json.dumps(delta_event)}\n\n"

                elif isinstance(event, ResponseTextDoneEvent):
                    # Send content block stop
                    if current_block_type == "text":
                        stop_event = {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        yield f"data: {json.dumps(stop_event)}\n\n"
                        content_block_index += 1
                        current_block_type = None

                elif isinstance(event, ResponseFunctionCallArgumentsDeltaEvent):
                    # Get function call information from event
                    call_id = getattr(event, "call_id", "")
                    function_name = getattr(event, "name", "")

                    # Start new tool use block if not already started
                    # todo: a new tool call id is not handled
                    if current_block_type != "tool_use":
                        tool_start = {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": call_id,
                                "name": function_name,
                                "input": {},
                            },
                        }
                        yield f"data: {json.dumps(tool_start)}\n\n"
                        current_block_type = "tool_use"

                    # Send function call delta with proper error handling
                    partial_json = getattr(event, "arguments_delta", "")
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

                elif isinstance(event, ResponseFunctionCallArgumentsDoneEvent):
                    # Complete the tool call content block
                    if current_block_type == "tool_use":
                        stop_event = {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        yield f"data: {json.dumps(stop_event)}\n\n"
                        content_block_index += 1
                        current_block_type = None

                elif isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
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
                        "delta": {
                            "type": "text_delta",
                            "text": event.delta or "",
                        },
                    }
                    yield f"data: {json.dumps(delta_event)}\n\n"

                elif isinstance(event, ResponseReasoningSummaryTextDoneEvent):
                    # Send reasoning summary block stop
                    if current_block_type == "reasoning":
                        stop_event = {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }
                        yield f"data: {json.dumps(stop_event)}\n\n"
                        content_block_index += 1
                        current_block_type = None

                elif isinstance(event, ResponseOutputTextAnnotationAddedEvent):
                    # Handle web search annotations
                    annotation = event.annotation
                    if isinstance(annotation, AnnotationURLCitation):
                        url_citation = annotation
                        # Add web search result content block
                        message_id = getattr(event, "id", "unknown")
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
                                        "title": url_citation.title or "",
                                        "url": url_citation.url or "",
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

                elif isinstance(event, ResponseCompletedEvent):
                    # Send final usage and message stop
                    usage = getattr(event, "usage", None)
                    if usage:
                        usage_event = {
                            "type": "message_delta",
                            "delta": {
                                "stop_reason": self._map_stop_reason(
                                    getattr(event, "status", None)
                                ),
                                "usage": self._map_usage(usage),
                            },
                        }
                        yield f"data: {json.dumps(usage_event)}\n\n"

                    # Send message stop event
                    yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
                    break

            except Exception as e:
                logger.warning(
                    "Failed to process OpenAI stream event",
                    event_type=type(event).__name__,
                    error=str(e),
                )
                continue
