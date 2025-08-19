import json
import os
from typing import Any

import httpx
import structlog

from ..config import Config
from ..router import ModelRouter

logger = structlog.get_logger(__name__)


class AnthropicOpenAIRequestAdapter:
    def __init__(self, config: Config, router: ModelRouter):
        self.config = config
        self.router = router
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                30.0,
                connect=self.config.timeouts_ms.connect / 1000,
                read=self.config.timeouts_ms.read / 1000,
            )
        )

    async def adapt_request(
        self, anthropic_request: dict[str, Any], target_model: str
    ) -> dict[str, Any]:
        """Translate Anthropic Messages API request to OpenAI Responses API request."""

        openai_request = {
            "model": target_model,
            "input": self._convert_messages(anthropic_request.get("messages", [])),
            "stream": anthropic_request.get("stream", False),
        }

        # Add system prompt if present (using 'instructions' for OpenAI Responses API)
        system = anthropic_request.get("system")
        if system:
            if isinstance(system, list):
                # Handle Anthropic system format: list of objects with 'text' field
                instruction_parts = []
                for item in system:
                    if isinstance(item, dict):
                        instruction_parts.append(item.get("text", ""))
                    else:
                        instruction_parts.append(str(item))
                openai_request["instructions"] = "\n".join(instruction_parts)
            else:
                openai_request["instructions"] = str(system)

        # Add tools if present
        tools = anthropic_request.get("tools")
        if tools:
            openai_request["tools"] = self._convert_tools(tools)

        # Map parameters
        if "temperature" in anthropic_request:
            openai_request["temperature"] = anthropic_request["temperature"]

        if "top_p" in anthropic_request:
            openai_request["top_p"] = anthropic_request["top_p"]

        if "max_tokens" in anthropic_request:
            openai_request["max_output_tokens"] = anthropic_request["max_tokens"]

        # Add reasoning effort for supported models
        if self._supports_reasoning(target_model):
            reasoning_effort = self.router.get_reasoning_effort(anthropic_request)
            logger.debug(
                "Reasoning effort calculated",
                effort=reasoning_effort,
                model=target_model,
            )
            # Always include reasoning config, even for minimal effort
            reasoning_config = {"effort": reasoning_effort}
            if reasoning_effort != "minimal":
                reasoning_config["summary"] = "auto"
            openai_request["reasoning"] = reasoning_config

        return openai_request

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic messages format to OpenAI input format."""

        def _role_to_content_type(role: str, block_type: str) -> str:
            role_l = (role or "").lower()
            if role_l in ("user", "system"):
                return f"input_{block_type}"
            if role_l in ("assistant", "bot"):
                return f"output_{block_type}"
            # fallback to input_text for unknown roles
            return "input_text"

        converted: list[dict[str, Any]] = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if not role or content is None:
                # Skip if there is no role or explicit None content
                continue

            current_msg = None

            def flush_message(curr_msg: dict[str, Any] | None) -> None:
                """Append current message if it has any content."""
                if curr_msg and curr_msg.get("content"):
                    converted.append(curr_msg)

            def append_content_to_msg(
                curr_msg: dict[str, Any] | None, content: dict[str, Any] | str
            ) -> dict[str, Any]:
                if curr_msg is None:
                    curr_msg = {"role": role, "content": [], "type": "message"}
                curr_msg["content"].append(content)
                return curr_msg

            if isinstance(content, str):
                # Simple text message (Anthropic shorthand format)
                current_msg = append_content_to_msg(current_msg, content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")

                        if block_type == "text":
                            ct = _role_to_content_type(role, block_type)
                            current_msg = append_content_to_msg(
                                current_msg, {"type": ct, "text": block.get("text", "")}
                            )

                        elif block_type == "tool_use":
                            # Start new message for tool call, flush accumulated content
                            flush_message(current_msg)
                            current_msg = None

                            # Serialize arguments
                            arguments = block.get("input", {})
                            try:
                                arguments_str = (
                                    arguments
                                    if isinstance(arguments, str)
                                    else json.dumps(arguments)
                                )
                            except (TypeError, ValueError):
                                arguments_str = str(arguments)

                            tool_msg = {
                                "type": "function_call",
                                "name": block.get("name", ""),
                                "arguments": arguments_str,
                                "call_id": block.get("id"),
                            }

                            converted.append(tool_msg)

                        elif block_type == "tool_result":
                            # Start new message for tool result, flush content
                            flush_message(current_msg)
                            current_msg = None

                            output = self._format_tool_result_content(
                                block.get("content")
                            )
                            result_msg = {
                                "type": "function_call_output",
                                "call_id": block.get("tool_use_id", "")
                                or block.get("id", ""),
                                "output": output,
                            }

                            converted.append(result_msg)

                        elif block_type == "image":
                            ct = "input_image" if role == "user" else "output_image"
                            source = block.get("source", {})
                            source_type = source.get("type", "base64")

                            if source_type == "url":
                                image_url = source.get("url", "")
                            else:  # base64 or other
                                media_type = source.get("media_type", "image/jpeg")
                                data = source.get("data", "")
                                image_url = (
                                    f"data:{media_type};base64,{data}" if data else ""
                                )

                            image_content = {
                                "type": ct,
                                "image_url": image_url,
                            }
                            current_msg = append_content_to_msg(
                                current_msg, image_content
                            )
                        else:
                            # Unknown block types: convert to text
                            text_val = (
                                block.get("text", "")
                                if isinstance(block, dict)
                                else str(block)
                            )
                            current_msg = append_content_to_msg(
                                current_msg,
                                {
                                    "type": _role_to_content_type(role, "text"),
                                    "text": text_val,
                                },
                            )

                    elif isinstance(block, str):
                        # String block within content array -> text
                        current_msg = append_content_to_msg(
                            current_msg,
                            {
                                "type": _role_to_content_type(role, "text"),
                                "text": block,
                            },
                        )
                    else:
                        # Any other type -> stringify
                        current_msg = append_content_to_msg(
                            current_msg,
                            {
                                "type": _role_to_content_type(role, "text"),
                                "text": str(block),
                            },
                        )

            else:
                # Fallback: stringify content
                current_msg = append_content_to_msg(current_msg, str(content))

            # Flush any remaining accumulated text-only message
            flush_message(current_msg)

        return converted

    def _convert_tools(
        self, anthropic_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tools format to OpenAI tools format.

        Preserve Anthropic `input_schema` verbatim as `parameters` (JSON Schema).
        No wrapping or heuristics are applied.
        """

        openai_tools = []
        for tool in anthropic_tools:
            # Preserve the original input_schema exactly as provided by Anthropic.
            # Use None when no schema is provided to make the intent explicit.
            schema = tool.get("input_schema", None)

            openai_tool = {
                "type": "function",
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                # Preserve schema verbatim (may be None).
                "parameters": schema,
            }

            openai_tools.append(openai_tool)

        return openai_tools

    def _supports_reasoning(self, model: str) -> bool:
        """Check if the model supports reasoning parameters.

        This is intentionally conservative: prefer explicit model hints from the router
        when available. Defaults to False if model is None or unlikely to match.
        """
        if not model:
            return False
        model_lower = model.lower()
        # Common explicit matches: gpt-5 family, "o4" or "o" style models used by OpenAI
        common_prefixes = ("gpt-5", "o4", "o")
        return (
            any(model_lower.startswith(p) for p in common_prefixes)
            and "-chat" not in model_lower
        )

    def _format_tool_result_content(self, content: Any) -> str:
        """Format tool result content for OpenAI."""

        if isinstance(content, str):
            return content
        elif isinstance(content, dict | list):
            return json.dumps(content)
        else:
            return str(content)

    async def make_request(
        self, openai_request: dict[str, Any], headers: dict[str, str]
    ) -> httpx.Response:
        """Make request to OpenAI API."""

        # Get API key (case-insensitive lookup handled by _get_api_key)
        api_key = self._get_api_key(headers)
        if not api_key:
            raise ValueError("OpenAI API key not found")

        request_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # If streaming, OpenAI Responses uses server-sent events
        if openai_request.get("stream"):
            request_headers["Accept"] = "text/event-stream"

        # Pass through request ID if present (case-insensitive lookup).
        lower_headers = {k.lower(): v for k, v in headers.items()}
        if "x-request-id" in lower_headers:
            request_id = lower_headers["x-request-id"]
            request_headers["x-request-id"] = request_id

        url = f"{self.config.router.openai_base_url}/v1/responses"

        logger.info(
            "Making OpenAI request",
            url=url,
            model=openai_request.get("model"),
            stream=openai_request.get("stream", False),
        )

        # Debug: Log the full request being sent but avoid leaking secrets
        debug_request = {
            k: v
            for k, v in openai_request.items()
            if k.lower() not in ("api_key", "authorization")
        }
        logger.debug("Full OpenAI request", request=debug_request)

        response = await self.client.post(
            url, headers=request_headers, json=openai_request
        )

        return response

    def _get_api_key(self, headers: dict[str, str]) -> str | None:
        """Get OpenAI API key from headers or environment (case-insensitive)."""

        # Normalize header keys to lowercase for robust lookup
        if headers:
            lower_headers = {k.lower(): v for k, v in headers.items()}

            # Support Authorization: Bearer <key> as a per-request override
            auth = lower_headers.get("authorization")
            if auth and auth.lower().startswith("bearer "):
                return auth.split(" ", 1)[1].strip()

        # Get from environment
        return os.getenv(self.config.openai.api_key_env)

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
