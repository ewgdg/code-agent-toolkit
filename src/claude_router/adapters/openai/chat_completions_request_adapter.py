import json
import os
from typing import Any

import structlog
from openai import AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from ...config import Config, ProviderConfig
from ...router import ModelRouter

logger = structlog.get_logger(__name__)


class ChatCompletionsRequestAdapter:
    """Adapter that translates Anthropic Messages API to OpenAI Chat Completions API."""

    def __init__(self, config: Config, router: ModelRouter):
        self.config = config
        self.router = router
        # We'll create per-provider clients as needed rather than a single client

    def adapt_request(
        self,
        anthropic_request: dict[str, Any],
        provider_config: ProviderConfig,
        model: str,
        model_config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Translate Anthropic Messages API request to OpenAI Chat Completions."""

        model_config = model_config or {}

        openai_request = {
            "model": model,
            "messages": self._convert_messages(anthropic_request.get("messages", [])),
            "stream": anthropic_request.get("stream", False),
        }

        # Add system message if present
        system = anthropic_request.get("system")
        if system:
            system_content = self._extract_system_content(system)
            if system_content:
                openai_request["messages"].insert(
                    0, {"role": "system", "content": system_content}
                )

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
            openai_request["max_tokens"] = anthropic_request["max_tokens"]

        # Add stop sequences if present
        if "stop_sequences" in anthropic_request:
            openai_request["stop"] = anthropic_request["stop_sequences"]

        # Add reasoning effort for supported models
        if self.config.openai.supports_reasoning(model):
            reasoning_effort = self.router.get_reasoning_effort(anthropic_request)
            logger.debug(
                "Reasoning effort calculated",
                effort=reasoning_effort,
                model=model,
            )
            # Always include reasoning config, even for minimal effort
            reasoning_config = {"effort": reasoning_effort}
            if reasoning_effort != "minimal":
                reasoning_config["summary"] = "auto"
            openai_request["reasoning"] = reasoning_config

        # Apply model configuration overrides
        if model_config:
            openai_request.update(model_config)

        return openai_request

    def _extract_system_content(self, system: Any) -> str:
        """Extract system content and concatenate into a single string."""
        if isinstance(system, str):
            return system

        if isinstance(system, list):
            text_parts = []
            for item in system:
                if isinstance(item, dict):
                    text_content = item.get("text", "")
                    if text_content:
                        text_parts.append(text_content)
                elif isinstance(item, str):
                    text_parts.append(item)
            return "\n".join(text_parts)

        return str(system)

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic messages format to OpenAI Chat Completions format."""
        converted = []

        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if not role or content is None:
                continue

            # Handle simple string content
            if isinstance(content, str):
                converted.append({"role": role, "content": content})
                continue

            # Handle complex content blocks
            if isinstance(content, list):
                openai_content = []
                tool_calls = []

                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")

                        if block_type == "text":
                            openai_content.append(
                                {"type": "text", "text": block.get("text", "")}
                            )

                        elif block_type == "image":
                            # Convert Anthropic image format to OpenAI format
                            source = block.get("source", {})
                            source_type = source.get("type", "base64")

                            if source_type == "url":
                                image_url = source.get("url", "")
                            else:  # base64
                                media_type = source.get("media_type", "image/jpeg")
                                data = source.get("data", "")
                                image_url = f"data:{media_type};base64,{data}"

                            openai_content.append(
                                {"type": "image_url", "image_url": {"url": image_url}}
                            )

                        elif block_type == "tool_use":
                            # Collect tool calls for assistant message
                            arguments = block.get("input", {})
                            try:
                                arguments_str = (
                                    arguments
                                    if isinstance(arguments, str)
                                    else json.dumps(arguments)
                                )
                            except (TypeError, ValueError):
                                arguments_str = str(arguments)

                            tool_calls.append(
                                {
                                    "id": block.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": arguments_str,
                                    },
                                }
                            )

                        elif block_type == "tool_result":
                            # Convert tool result to assistant message
                            tool_content = self._format_tool_result_content(
                                block.get("content")
                            )
                            converted.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", ""),
                                    "content": tool_content,
                                }
                            )

                    elif isinstance(block, str):
                        openai_content.append({"type": "text", "text": block})

                # Create message with content and/or tool calls
                if openai_content or tool_calls:
                    message_data = {"role": role}

                    if openai_content:
                        # If there's only one text block, use string format
                        if (
                            len(openai_content) == 1
                            and openai_content[0]["type"] == "text"
                        ):
                            message_data["content"] = openai_content[0]["text"]
                        else:
                            message_data["content"] = openai_content

                    if tool_calls:
                        message_data["tool_calls"] = tool_calls

                    converted.append(message_data)

            else:
                # Fallback for other content types
                converted.append({"role": role, "content": str(content)})

        return converted

    def _convert_tools(
        self, anthropic_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tools format to OpenAI tools format."""
        openai_tools = []

        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def _format_tool_result_content(self, content: Any) -> str:
        """Format tool result content for OpenAI."""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict | list):
            return json.dumps(content)
        else:
            return str(content)

    def _create_client(self, provider_config: ProviderConfig) -> AsyncOpenAI:
        """Create OpenAI client for the specified provider."""
        # Set up timeouts
        timeouts = provider_config.timeouts_ms or self.config.timeouts_ms
        timeout_seconds = timeouts.read / 1000

        # Get API key
        api_key = "dummy"  # Default fallback
        if provider_config.api_key_env:
            api_key = os.getenv(provider_config.api_key_env, "dummy")

        return AsyncOpenAI(
            base_url=provider_config.base_url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

    async def make_request(
        self,
        openai_request: dict[str, Any],
        provider_config: ProviderConfig,
        headers: dict[str, str],
    ) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        """Make request to OpenAI-compatible Chat Completions endpoint using SDK."""

        # Create client for this provider
        client = self._create_client(provider_config)

        # Set extra headers for request ID if present
        extra_headers = {}
        lower_headers = {k.lower(): v for k, v in headers.items()}
        if "x-request-id" in lower_headers:
            extra_headers["x-request-id"] = lower_headers["x-request-id"]

        logger.info(
            "Making OpenAI Chat Completions request",
            base_url=provider_config.base_url,
            model=openai_request.get("model"),
            stream=openai_request.get("stream", False),
        )

        try:
            # Use OpenAI SDK's chat.completions.create method
            response = await client.chat.completions.create(
                extra_headers=extra_headers, **openai_request
            )
            # The OpenAI SDK returns either ChatCompletion or AsyncStream[...]
            # based on the stream parameter, which MyPy can't infer from **kwargs
            return response  # type: ignore[no-any-return]
        finally:
            await client.close()

