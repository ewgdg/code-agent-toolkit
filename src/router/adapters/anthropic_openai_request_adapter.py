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

        # Add system prompt if present
        system = anthropic_request.get("system")
        if system:
            if isinstance(system, list):
                openai_request["system"] = "\n".join(str(item) for item in system)
            else:
                openai_request["system"] = str(system)

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

        # Add reasoning effort (check config first, fallback to model detection)
        if self.config.openai.supports_reasoning or self._supports_reasoning(
            target_model
        ):
            reasoning_effort = self.router.get_reasoning_effort(anthropic_request)
            if reasoning_effort != "minimal":
                reasoning_config = {"effort": reasoning_effort}
                openai_request["reasoning"] = reasoning_config

        return openai_request

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert Anthropic messages format to OpenAI input format."""

        converted = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")

            if not role or not content:
                continue

            openai_message = {"role": role, "content": []}

            # Handle content blocks
            if isinstance(content, str):
                # Simple text message
                openai_message["content"].append(
                    {"type": "input_text", "text": content}
                )
            elif isinstance(content, list):
                # Multiple content blocks
                for block in content:
                    if isinstance(block, dict):
                        block_type = block.get("type")

                        if block_type == "text":
                            openai_message["content"].append(
                                {"type": "input_text", "text": block.get("text", "")}
                            )
                        elif block_type == "tool_use":
                            # Convert Anthropic tool_use to OpenAI function_call
                            openai_message["content"].append(
                                {
                                    "type": "function_call",
                                    "call_id": block.get("id"),
                                    "name": block.get("name"),
                                    "arguments": json.dumps(block.get("input", {})),
                                }
                            )
                        elif block_type == "tool_result":
                            # Convert Anthropic tool_result to OpenAI output
                            tool_result_content = self._format_tool_result_content(
                                block.get("content")
                            )
                            openai_message["content"].append(
                                {
                                    "type": "function_call_output",
                                    "call_id": block.get("tool_use_id"),
                                    "output": tool_result_content,
                                }
                            )
                    elif isinstance(block, str):
                        openai_message["content"].append(
                            {"type": "input_text", "text": block}
                        )
            else:
                # Fallback to string conversion
                openai_message["content"].append(
                    {"type": "input_text", "text": str(content)}
                )

            converted.append(openai_message)

        return converted

    def _convert_tools(
        self, anthropic_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tools format to OpenAI tools format."""

        openai_tools = []
        for tool in anthropic_tools:
            # Convert Anthropic tool to OpenAI function tool
            openai_tool = {
                "type": "function",
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            }

            # Enable strict mode for better reliability
            openai_tool["strict"] = True

            # Ensure additionalProperties is false for strict mode
            parameters = openai_tool["parameters"]
            if "parameters" in openai_tool and isinstance(parameters, dict):
                if "additionalProperties" not in parameters:
                    openai_tool["parameters"]["additionalProperties"] = False

            openai_tools.append(openai_tool)

        return openai_tools

    def _supports_reasoning(self, model: str) -> bool:
        """Check if the model supports reasoning parameters."""
        reasoning_prefixes = ["o", "gpt-5"]
        model_lower = model.lower()
        return (
            any(model_lower.startswith(prefix) for prefix in reasoning_prefixes)
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

        # Get API key
        api_key = self._get_api_key(headers)
        if not api_key:
            raise ValueError("OpenAI API key not found")

        request_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # Pass through request ID if present
        if "x-request-id" in headers:
            request_headers["x-request-id"] = headers["x-request-id"]

        url = f"{self.config.router.openai_base_url}/v1/responses"

        logger.info(
            "Making OpenAI request",
            url=url,
            model=openai_request.get("model"),
            stream=openai_request.get("stream", False),
        )

        response = await self.client.post(
            url, headers=request_headers, json=openai_request
        )

        return response

    def _get_api_key(self, headers: dict[str, str]) -> str | None:
        """Get OpenAI API key from headers or environment."""

        # Check for per-request API key header (if enabled)
        per_request_key = headers.get("X-OpenAI-API-Key")
        if per_request_key:
            return per_request_key

        # Get from environment
        return os.getenv(self.config.openai.api_key_env)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()