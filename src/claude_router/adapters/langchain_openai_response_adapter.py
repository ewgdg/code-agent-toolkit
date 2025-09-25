# adapters/langchain_responses_response_adapter.py
"""
LangChain‑based Responses API response adapter.

Transforms LangChain AIMessage / AIMessageChunk (or an async iterator of them)
into the Anthropic‑compatible JSON payload that the gateway returns.

The implementation mirrors the conversion performed by the original
ResponsesRequestAdapter (which used the OpenAI SDK) so that the public API
and the JSON shape are 100 % compatible.
"""

import json
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import (
    Any,
    cast,
)

import structlog
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ToolMessage,
)

# Optional config / router – only needed for reasoning‑effort helpers.
from ..config import Config
from ..router import ModelRouter

log = structlog.get_logger(__name__)

# Configuration for custom field parsing
CUSTOM_FIELD_MAPPING = {
    "reasoning_content": {"block_type": "thinking", "field_name": "thinking"},
    "thinking_content": {"block_type": "thinking", "field_name": "thinking"},
    "reasoning": {"block_type": "thinking", "field_name": "thinking"},
    "thinking": {"block_type": "thinking", "field_name": "thinking"},
}


def _text_block(text: str) -> dict[str, Any]:
    """Anthropic text block."""
    return {"type": "text", "text": text}


def _image_block(url: str) -> dict[str, Any]:
    """Anthropic image block (url or data‑uri)."""
    if url.startswith("data:"):
        # Extract base64 data from data URI (e.g., "data:image/jpeg;base64,/9j/4AAQ...")
        if ";base64," in url:
            media_type, base64_data = url.split(";base64,", 1)
            return {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type.replace("data:", ""),
                    "data": base64_data,
                },
            }
    # Regular URL
    return {"type": "image", "source": {"type": "url", "url": url}}


def _function_call_block(name: str, args: Any, call_id: str) -> dict[str, Any]:
    """
    Anthropic representation of a tool call.
    ``args`` may be a JSON string or a Python dict – we convert to dict for Anthropic format.
    """
    if isinstance(args, str):
        try:
            input_data = json.loads(args)
        except (json.JSONDecodeError, TypeError, ValueError):
            input_data = {"raw_arguments": args}
    elif isinstance(args, dict):
        input_data = args
    else:
        input_data = {"raw_arguments": str(args)}

    result = {
        "type": "tool_use",
        "id": call_id,
        "name": name,
        "input": input_data,
    }

    return result


def _custom_field_block(
    field_key: str, field_value: Any
) -> tuple[dict[str, Any], str] | None:
    """Create a custom field block based on configuration mapping.

    Defensive against malformed entries in CUSTOM_FIELD_MAPPING. If an entry is
    missing required keys ("block_type" or "field_name") or is not a dict, we
    log and return None instead of raising.
    """
    config = CUSTOM_FIELD_MAPPING.get(field_key)
    if not isinstance(config, dict):
        return None

    block_type = config.get("block_type")
    field_name = config.get("field_name")
    if not isinstance(block_type, str) or not isinstance(field_name, str):
        log.warning(
            "Malformed CUSTOM_FIELD_MAPPING entry; skipping custom field",
            field_key=field_key,
            config=config,
        )
        return None

    block = {"type": block_type, field_name: str(field_value)}
    return block, field_name


# (removed helper; we now return field_name from _custom_field_block)


def _function_output_block(call_id: str, output: Any) -> dict[str, Any]:
    """Anthropic representation of a tool result."""
    if isinstance(output, str):
        out_str = output
    else:
        try:
            out_str = json.dumps(output, ensure_ascii=False)
        except Exception:
            out_str = str(output)

    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": out_str,
    }


def _content_blocks_from_message(
    message: AIMessage, use_responses_api: bool
) -> list[dict[str, Any]]:
    """
    Convert the ``content`` attribute of an AIMessage (or AIMessageChunk)
    into the exact list of Anthropic blocks.
    """
    blocks: list[dict[str, Any]] = []

    # ── Text / image / mixed list -----------------
    raw_content = message.content
    if isinstance(raw_content, str):
        # Simple string → single text block
        blocks.append(_text_block(raw_content))
    elif isinstance(raw_content, list):
        for item in raw_content:
            # LangChain may already give us dicts that follow the OpenAI schema.
            # We still normalise them to the Anthropic shape.
            if isinstance(item, dict):
                block_type = item.get("type")
                if block_type == "text":
                    blocks.append(_text_block(item.get("text", "")))
                elif block_type == "image_url":
                    # Handle LangChain/OpenAI format: {"type": "image_url", "image_url": {"url": "..."}}
                    image_url_obj = item.get("image_url", "")
                    if isinstance(image_url_obj, dict):
                        url = image_url_obj.get("url", "")
                    else:
                        url = str(image_url_obj)
                    blocks.append(_image_block(url))
                elif block_type == "reasoning":
                    # LangChain OpenAI reasoning format
                    # Assume summary is a list; concatenate all text parts into a single string.
                    item_id = item.get("id") if use_responses_api else None
                    # When using Responses API, the reasoning item may carry
                    # an opaque encrypted payload we should surface for
                    # round‑tripping on subsequent requests.
                    encrypted_content = (
                        item.get("encrypted_content") if use_responses_api else None
                    )
                    summary_list = item.get("summary", [])
                    thinking_text = ""
                    if isinstance(summary_list, list):
                        parts: list[str] = []
                        for summary_item in summary_list:
                            if isinstance(summary_item, dict):
                                parts.append(str(summary_item.get("text", "")))
                            else:
                                parts.append(str(summary_item))
                        thinking_text = "".join(parts)
                    else:
                        thinking_text = str(summary_list)

                    if thinking_text or item_id or encrypted_content:
                        tb: dict[str, Any] = {
                            "type": "thinking",
                            "thinking": thinking_text,
                        }
                        if isinstance(item_id, str) and item_id:
                            tb["extracted_openai_rs_id"] = item_id
                        if encrypted_content:
                            tb["extracted_openai_rs_encrypted_content"] = (
                                encrypted_content
                            )
                        blocks.append(tb)

                        # Info log for OpenAI reasoning summary (non-stream)
                        try:
                            preview = thinking_text[:200] if thinking_text else ""
                            log.debug(
                                "OpenAI reasoning summary extracted",
                                mode="non_stream",
                                preview=preview,
                                length=len(thinking_text),
                                item_id=item_id,
                                has_encrypted=bool(encrypted_content),
                            )
                        except Exception:
                            # Never fail the request because of logging
                            pass
                elif block_type == "web_search_call":
                    blocks.append(
                        {
                            "type": "thinking",
                            "thinking": "web_search has been performed.",
                        }
                    )
                else:
                    # Unknown dict – fall back to JSON stringified version.
                    blocks.append(_text_block(json.dumps(item, ensure_ascii=False)))
            else:
                # Primitive (unlikely) – string‑ify.
                blocks.append(_text_block(str(item)))
    else:
        # Anything else – stringify.
        blocks.append(_text_block(str(raw_content)))

    # ── Extract custom fields from additional_kwargs (Chat Completions only) ──
    if message.additional_kwargs:
        log.debug(
            "Found additional_kwargs in message",
            message_type=type(message).__name__,
            kwargs_keys=list(message.additional_kwargs.keys()),
        )
        for key, value in message.additional_kwargs.items():
            if value:
                result = _custom_field_block(key, value)
                if result:
                    custom_block, _ = result
                    blocks.append(custom_block)

    return blocks


def _tool_calls_from_message(message: AIMessage) -> list[dict[str, Any]]:
    """Extract tool calls from an AIMessage (or chunk)."""
    calls = []
    for tc in message.tool_calls:
        calls.append(
            _function_call_block(
                name=tc["name"],
                args=tc["args"],
                call_id=tc["id"] or str(uuid.uuid4()),
            )
        )
    return calls


def _tool_results_from_message(message: BaseMessage) -> list[dict[str, Any]]:
    """If the message is a ToolMessage, render it as a function_call_output block."""
    if isinstance(message, ToolMessage):
        return [_function_output_block(message.tool_call_id, message.content)]
    return []


def _usage_from_message(message: AIMessage) -> Mapping[str, int]:
    """Pull token usage from response_metadata or usage_metadata."""
    if message.usage_metadata:
        um = message.usage_metadata
        # Dict-like access
        input_tokens = um.get("input_tokens", 0)
        output_tokens = um.get("output_tokens", 0)
        total_tokens = um.get("total_tokens", 0) or (input_tokens + output_tokens)

        return {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _finish_reason_from_message(message: AIMessage) -> str | None:
    if message.response_metadata:
        return message.response_metadata.get("finish_reason")
    return None


# --------------------------------------------------------------------------- #
# Main adapter class
# --------------------------------------------------------------------------- #
class LangChainOpenAIResponseAdapter:
    """
    Convert LangChain output (AIMessage / AIMessageChunk or async iterator of them)
    into the Anthropic‑compatible JSON response format used by the gateway.

    Public API mirrors the original OpenAI‑SDK based adapter:
        * adapt_response(...)
        * make_response(...)
    """

    def __init__(self, config: Config | None = None, router: ModelRouter | None = None):
        self.config = config
        self.router = router

    def _extract_model_name(self, message: BaseMessage) -> str:
        """Extract model name from message metadata with fallbacks."""
        if hasattr(message, "response_metadata"):
            meta = message.response_metadata
            # Try various common field names for model
            for key in ["model_name", "model", "engine", "deployment_name"]:
                if key in meta and meta[key]:
                    return str(meta[key])

        # Fallback to generic model name
        return "unknown"

    def _map_stop_reason(self, langchain_stop_reason: str | None) -> str:
        """Map LangChain stop reason to Anthropic format."""
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
            None: "end_turn",
        }
        return mapping.get(langchain_stop_reason, "end_turn")

    # --------------------------------------------------------------------- #
    # Public entry point
    # --------------------------------------------------------------------- #
    async def adapt_response(
        self,
        raw: BaseMessage | AsyncIterator[BaseMessageChunk],
        headers: Mapping[str, str] | None = None,
        *,
        use_responses_api: bool = False,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """
        Dispatch to the correct implementation based on whether ``raw`` is a
        finished message or a streaming iterator.
        """
        if isinstance(raw, BaseMessage):
            return self._non_stream_response(raw, headers, use_responses_api)
        else:
            return self._stream_response(raw, headers, use_responses_api)

    async def make_response(
        self,
        raw: BaseMessage | AsyncIterator[BaseMessageChunk],
        headers: Mapping[str, str] | None = None,
        *,
        use_responses_api: bool = False,
    ) -> dict[str, Any] | AsyncIterator[str]:
        """Thin wrapper kept for API symmetry with the OpenAI‑SDK adapter."""
        return await self.adapt_response(
            raw, headers, use_responses_api=use_responses_api
        )

    # --------------------------------------------------------------------- #
    # Non‑streaming (single message)
    # --------------------------------------------------------------------- #
    def _non_stream_response(
        self,
        message: BaseMessage,
        headers: Mapping[str, str] | None,
        use_responses_api: bool,
    ) -> dict[str, Any]:
        if not isinstance(message, AIMessage):
            log.error(
                "Non-AIMessage supplied to finished response builder",
                type=type(message),
            )
            raise TypeError("Finished response must be an AIMessage")

        # Build Anthropic-style response (matching ResponsesResponseAdapter)
        anthropic_response: dict[str, Any] = {
            "id": message.id or f"msg_{uuid.uuid4().hex}",
            "type": "message",
            "role": "assistant",
            "model": self._extract_model_name(message),
            "content": [],
            "stop_reason": "end_turn",
            "usage": _usage_from_message(message),
        }

        # Echo the request‑id (if supplied)
        if headers:
            low = {k.lower(): v for k, v in headers.items()}
            if "x-request-id" in low:
                anthropic_response["request_id"] = low["x-request-id"]

        # ── Content (text / image / reasoning) ----------------------------
        content_blocks = _content_blocks_from_message(message, use_responses_api)
        anthropic_response["content"].extend(content_blocks)

        # ── Tool calls (if any) --------------------------------------------
        tool_calls = _tool_calls_from_message(message)
        anthropic_response["content"].extend(tool_calls)
        if tool_calls:
            anthropic_response["stop_reason"] = "tool_use"

        # Map finish reason to Anthropic format
        finish = _finish_reason_from_message(message)
        if finish:
            anthropic_response["stop_reason"] = self._map_stop_reason(finish)

        return anthropic_response

    def _start_content_block(
        self, block_type: str, index: int, block_data: dict[str, Any]
    ) -> str:
        """Start a new content block and return the SSE event string."""
        content_start = {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": block_type, **block_data},
        }
        event_data = json.dumps(content_start)
        return f"event: content_block_start\ndata: {event_data}\n\n"

    def _stop_content_block(self, index: int) -> str:
        """Stop a content block and return the SSE event string."""
        stop_event = {
            "type": "content_block_stop",
            "index": index,
        }
        event_data = json.dumps(stop_event)
        return f"event: content_block_stop\ndata: {event_data}\n\n"

    def _send_text_delta(self, index: int, text: str) -> str:
        """Send a text delta and return the SSE event string."""
        delta_event = {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": "text_delta",
                "text": text,
            },
        }
        event_data = json.dumps(delta_event)
        return f"event: content_block_delta\ndata: {event_data}\n\n"

    def _send_thinking_delta(self, index: int, thinking: str) -> str:
        """Send a thinking delta and return the SSE event string."""
        thinking_delta = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "thinking_delta", "thinking": thinking},
        }
        event_data = json.dumps(thinking_delta)
        return f"event: content_block_delta\ndata: {event_data}\n\n"

    def _send_delta(
        self, index: int, block_type: str, field_name: str, value: str
    ) -> str:
        """Send a custom field delta and return the SSE event string."""
        custom_delta = {
            "type": "content_block_delta",
            "index": index,
            "delta": {
                "type": f"{block_type}_delta",
                field_name: value,
            },
        }
        event_data = json.dumps(custom_delta)
        return f"event: content_block_delta\ndata: {event_data}\n\n"

    async def _stream_response(
        self,
        chunk_iter: AsyncIterator[BaseMessageChunk],
        headers: Mapping[str, str] | None,
        use_responses_api: bool,
    ) -> AsyncIterator[str]:
        """
        Convert LangChain streaming chunks to Anthropic streaming format.
        Yields Server-Sent Events formatted strings.

        Key principles based on LangChain docs:
        - AIMessageChunk.content contains incremental content (not cumulative)
        - Tool calls are in chunk.tool_call_chunks during streaming
        - Accumulate all chunks into a complete message for final metadata extraction
        """
        message_started = False
        content_block_index = 0
        current_block_type: str | None = None
        current_tool_call_id: str | None = (
            None  # Track current tool call for sequential processing
        )
        request_id = None

        # Track tool call metadata by index for consistent streaming
        tool_call_map: dict[int, dict[str, str]] = {}

        # Accumulate message for final metadata extraction
        accumulated_message: AIMessageChunk | None = None

        if headers:
            low = {k.lower(): v for k, v in headers.items()}
            request_id = low.get("x-request-id")

        async for chunk in chunk_iter:
            if not isinstance(chunk, AIMessageChunk):
                log.error("Unexpected non‑AIMessageChunk in stream", type=type(chunk))
                continue

            try:
                # Accumulate chunk into complete message
                if accumulated_message is None:
                    accumulated_message = chunk
                else:
                    accumulated_message = cast(
                        AIMessageChunk, accumulated_message + chunk
                    )

                # Send message start event if not already sent
                if not message_started:
                    message_obj: dict[str, Any] = {
                        "id": accumulated_message.id or f"msg_{uuid.uuid4().hex}",
                        "type": "message",
                        "role": "assistant",
                        "model": self._extract_model_name(chunk),
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    }
                    if request_id:
                        message_obj["request_id"] = request_id

                    message_start = {
                        "type": "message_start",
                        "message": message_obj,
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                    message_started = True

                # ── Handle custom fields from additional_kwargs first ──
                if chunk.additional_kwargs:
                    # Process all configured custom fields
                    for key, value in chunk.additional_kwargs.items():
                        if value:  # Only process non-empty values
                            result = _custom_field_block(key, value)
                            if result:
                                custom_block, field_name = result
                                block_type = custom_block["type"]

                                # Start custom field content block if not already started
                                if current_block_type != block_type:
                                    # Close current block if it was open
                                    if current_block_type is not None:
                                        yield self._stop_content_block(
                                            content_block_index
                                        )
                                        content_block_index += 1

                                    # Start custom field content block
                                    start_data = {field_name: ""}
                                    yield self._start_content_block(
                                        block_type,
                                        content_block_index,
                                        start_data,
                                    )
                                    current_block_type = block_type

                                # Send custom field delta
                                yield self._send_delta(
                                    content_block_index,
                                    block_type,
                                    field_name,
                                    str(value),
                                )

                # Handle both text content and structured content (v1 reasoning)
                if chunk.content:
                    # Handle simple string content
                    if isinstance(chunk.content, str):
                        # Start text content block if not already started
                        if current_block_type != "text":
                            # Close current block if it was open
                            if current_block_type is not None:
                                yield self._stop_content_block(content_block_index)
                                content_block_index += 1

                            yield self._start_content_block(
                                "text", content_block_index, {"text": ""}
                            )
                            current_block_type = "text"

                        # Send text delta with incremental content
                        yield self._send_text_delta(content_block_index, chunk.content)

                    # Handle structured content list (v1 format with reasoning)
                    elif isinstance(chunk.content, list):
                        for item in chunk.content:
                            if isinstance(item, dict):
                                block_type = item.get("type")
                                if block_type == "web_search_call":
                                    call_id = item.get("id") or item.get("call_id")
                                    if not call_id:
                                        log.warning(
                                            "web_search_call block missing id",
                                            item=item,
                                        )
                                        continue

                                    call_id = str(call_id)

                                    if (
                                        current_block_type != "web_search_call"
                                        or current_tool_call_id != call_id
                                    ):
                                        if current_block_type is not None:
                                            yield self._stop_content_block(
                                                content_block_index
                                            )
                                            content_block_index += 1
                                            current_block_type = None

                                        current_tool_call_id = call_id

                                        current_block_type = "web_search_call"
                                        start_payload = {"thinking": ""}
                                        yield self._start_content_block(
                                            "thinking",
                                            content_block_index,
                                            start_payload,
                                        )

                                        # yield self._start_content_block(
                                        #     "server_tool_use",
                                        #     content_block_index,
                                        #     {
                                        #         "id": call_id,
                                        #         "name": "web_search",
                                        #         "input": {"query": ""},
                                        #     },
                                        # )

                                    status_messages = ["invoking web_search"]

                                    action = item.get("action")
                                    action_dict = (
                                        action if isinstance(action, dict) else {}
                                    )

                                    action_type = action_dict.get("type")
                                    # if action_type not in {
                                    #     "search",
                                    #     "open_page",
                                    #     "find",
                                    # }:
                                    #     continue

                                    action_message = ""
                                    if action_type:
                                        query: str | None = action_dict.get("query")
                                        url: str | None = action_dict.get("url")
                                        pattern: str | None = action_dict.get("pattern")
                                        action_message = f"action: {action_type}"

                                        if query:
                                            action_message += f", query: {query}"
                                        if pattern:
                                            action_message += f", pattern: {pattern}"
                                        if url:
                                            action_message += f", url: {url}"

                                    if action_message:
                                        status_messages.append(action_message)

                                    status = item.get("status")

                                    if status == "completed":
                                        status_messages.append("web_search completed")

                                    if status_messages:
                                        thinking_update = "\n".join(status_messages)
                                        yield self._send_thinking_delta(
                                            content_block_index, thinking_update
                                        )

                                    if status == "completed":
                                        yield self._stop_content_block(
                                            content_block_index
                                        )
                                        content_block_index += 1
                                        current_block_type = None

                                if block_type == "text":
                                    text_content = item.get("text", "")
                                    if text_content:
                                        # Start text block if needed
                                        if current_block_type != "text":
                                            # Close current block if it was open
                                            if current_block_type is not None:
                                                yield self._stop_content_block(
                                                    content_block_index
                                                )
                                                content_block_index += 1

                                            yield self._start_content_block(
                                                "text",
                                                content_block_index,
                                                {"text": ""},
                                            )
                                            current_block_type = "text"

                                        # Send text delta
                                        yield self._send_text_delta(
                                            content_block_index, text_content
                                        )

                                        # Ignore annotations;

                                elif block_type == "reasoning":
                                    # LangChain OpenAI reasoning format - extract from summary array
                                    summary_items = item.get("summary", [])
                                    reasoning_id = (
                                        item.get("id") if use_responses_api else None
                                    )
                                    encrypted_content = (
                                        item.get("encrypted_content")
                                        if use_responses_api
                                        else None
                                    )
                                    thinking_parts = []
                                    for summary_item in summary_items:
                                        summary_item_text = summary_item.get("text", "")
                                        if summary_item_text:
                                            thinking_parts.append(summary_item_text)

                                    thinking_text = "".join(thinking_parts)
                                    if (
                                        thinking_text
                                        or reasoning_id
                                        or encrypted_content
                                    ):
                                        # Start thinking content block if not already started
                                        if current_block_type != "thinking":
                                            # Close current block if it was open
                                            if current_block_type is not None:
                                                yield self._stop_content_block(
                                                    content_block_index
                                                )
                                                content_block_index += 1

                                            # Start thinking content block; include rs_* id and encrypted payload if present
                                            start_payload = {"thinking": ""}

                                            if reasoning_id:
                                                start_payload[
                                                    "extracted_openai_rs_id"
                                                ] = reasoning_id

                                            # we can only put custom fields in the start chunk
                                            # the reconciler will discard them from delta chunks
                                            if encrypted_content:
                                                start_payload[
                                                    "extracted_openai_rs_encrypted_content"
                                                ] = encrypted_content

                                            yield self._start_content_block(
                                                "thinking",
                                                content_block_index,
                                                start_payload,
                                            )
                                            current_block_type = "thinking"

                                        if thinking_text:
                                            yield self._send_thinking_delta(
                                                content_block_index,
                                                thinking_text,
                                            )

                            elif isinstance(item, str) and item:
                                # Start text block if needed
                                if current_block_type != "text":
                                    # Close current block if it was open
                                    if current_block_type is not None:
                                        yield self._stop_content_block(
                                            content_block_index
                                        )
                                        content_block_index += 1

                                    yield self._start_content_block(
                                        "text", content_block_index, {"text": ""}
                                    )
                                    current_block_type = "text"

                                # Send text delta
                                yield self._send_text_delta(content_block_index, item)

                # Handle tool call chunks (sequential processing like other content blocks)
                if chunk.tool_call_chunks:
                    for tool_chunk in chunk.tool_call_chunks:
                        chunk_index = tool_chunk.get("index") or 0

                        # Build tool call map: store name/id from first chunk with these values
                        tool_name = tool_chunk.get("name")
                        call_id = tool_chunk.get("call_id") or tool_chunk.get("id")
                        if tool_name and call_id:
                            tool_call_map[chunk_index] = {
                                "name": tool_name,
                                "id": call_id,
                            }

                        # Get consistent metadata from map
                        elif chunk_index in tool_call_map:
                            tool_name = tool_call_map[chunk_index]["name"]
                            call_id = tool_call_map[chunk_index]["id"]
                        else:
                            # This should not happen - first chunk should have name/id
                            log.error(
                                "Tool call chunk missing metadata",
                                chunk_index=chunk_index,
                                tool_chunk=tool_chunk,
                                tool_call_map=tool_call_map,
                            )
                            raise ValueError(
                                f"Tool call chunk at index {chunk_index} missing name/id metadata. "
                                f"First chunk should contain tool name and ID."
                            )

                        # Start new tool use block if we're not already in one or if this is a new tool call
                        if (
                            current_block_type != "tool_use"
                            or current_tool_call_id != call_id
                        ):
                            # Close current block if it was open
                            if current_block_type is not None:
                                yield self._stop_content_block(content_block_index)
                                content_block_index += 1

                            # Start tool use block
                            tool_start_block = {
                                "id": call_id,
                                "name": tool_name,
                                "input": {},
                            }

                            # Log all tool use block starts
                            log.debug(
                                "Starting tool_use content block",
                                tool_name=tool_name,
                                call_id=call_id,
                                chunk_index=chunk_index,
                                tool_chunk=tool_chunk,
                                content_block_index=content_block_index,
                                tool_start_block=tool_start_block,
                            )

                            yield self._start_content_block(
                                "tool_use",
                                content_block_index,
                                tool_start_block,
                            )
                            current_block_type = "tool_use"
                            current_tool_call_id = call_id

                        # Send tool call arguments delta if present
                        if "args" in tool_chunk and tool_chunk["args"]:
                            args_str = tool_chunk["args"]
                            if not isinstance(args_str, str):
                                try:
                                    args_str = json.dumps(args_str)
                                except Exception:
                                    args_str = str(args_str)

                            delta_event = {
                                "type": "content_block_delta",
                                "index": content_block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": args_str,
                                },
                            }

                            event_data = json.dumps(delta_event)
                            yield f"event: content_block_delta\ndata: {event_data}\n\n"

            except Exception as e:
                log.warning(
                    "Failed to process LangChain stream chunk",
                    chunk_type=type(chunk).__name__,
                    error=str(e),
                )
                continue

        # After loop ends, extract final metadata from accumulated message
        if accumulated_message:
            finish_reason = _finish_reason_from_message(accumulated_message)
            usage = _usage_from_message(accumulated_message)

            # Debug log accumulated additional_kwargs after stream is complete
            if accumulated_message.additional_kwargs:
                log.debug(
                    "Custom fields processed in accumulated message",
                    kwargs_keys=list(accumulated_message.additional_kwargs.keys()),
                )

            if accumulated_message.tool_calls:
                log.debug(
                    "tool calls sent",
                    raw_tool_calls=accumulated_message.tool_calls,
                )

            # Close any open blocks
            if current_block_type is not None:
                yield self._stop_content_block(content_block_index)

            # Send message delta with usage and stop reason from complete message
            delta_event = {
                "type": "message_delta",
                "delta": {
                    "stop_reason": self._map_stop_reason(finish_reason),
                },
                "usage": {
                    "output_tokens": usage.get("completion_tokens", 0),
                },
            }
            yield f"event: message_delta\ndata: {json.dumps(delta_event)}\n\n"

            # Send message stop event
            yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'
