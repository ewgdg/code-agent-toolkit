"""
Unified LangChain-based OpenAI request adapter supporting both Responses and Chat Completions APIs.

This adapter replaces the existing 4 separate adapters with 2 consolidated, feature-rich adapters,
achieving significant code reduction while maintaining complete feature parity.
"""

import json
import os
from collections.abc import AsyncIterator
from typing import Any, cast

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import ConfigurableField, RunnableSerializable
from pydantic import SecretStr

from ..config import Config, ProviderConfig
from ..config.schema import ModelConfigEntry
from ..router import ModelRouter
from .custom_chatopenai import ChatOpenAIWithCustomFields

logger = structlog.get_logger(__name__)


class LangChainOpenAIRequestAdapter:
    """
    Unified request adapter supporting both OpenAI Responses API and Chat Completions API.
    Uses LangChain for message conversion and provider abstraction.
    """

    def __init__(self, config: Config, router: ModelRouter):
        self.config = config
        self.router = router
        self._model_cache: dict[str, RunnableSerializable] = {}

    async def adapt_request(
        self,
        anthropic_request: dict[str, Any],
        provider_config: ProviderConfig,
        model: str,
        use_responses_api: bool = True,
        model_config: dict[str, Any | ModelConfigEntry] | None = None,
        support_reasoning: bool = False,
    ) -> dict[str, Any]:
        """
        Translate Anthropic Messages API request to OpenAI format.

        Args:
            anthropic_request: Anthropic Messages API request
            provider_config: Provider configuration
            model: Target model name
            use_responses_api: True for Responses API, False for Chat Completions
            model_config: Additional model configuration overrides
            support_reasoning: Whether the model supports reasoning capabilities

        Returns:
            OpenAI API request dict
        """
        try:
            # Convert Anthropic messages to LangChain format
            messages = self._convert_to_langchain_messages(anthropic_request)

            logger.debug(
                "Converted messages to LangChain format",
                message_count=len(messages),
                model=model,
                use_responses_api=use_responses_api,
            )

            # Prepare OpenAI request payload (LangChain-executable)
            adapted_request = self._prepare_openai_request(
                messages, anthropic_request, model, model_config
            )

            logger.info(
                "LangChain OpenAI request prepared",
                model=model,
                api_type="responses" if use_responses_api else "chat_completions",
                message_count=len(messages),
                has_tools=bool(anthropic_request.get("tools")),
                has_reasoning=support_reasoning
                or self.config.openai.supports_reasoning(model),
                stream=anthropic_request.get("stream", False),
            )

            return adapted_request

        except Exception as e:
            logger.exception(
                "Failed to adapt request",
                error=str(e),
                model=model,
                use_responses_api=use_responses_api,
            )
            raise ValueError(f"Request adaptation failed: {e}") from e

    def _get_langchain_model(
        self,
        provider_config: ProviderConfig,
        model: str,
    ) -> RunnableSerializable:
        """Create or retrieve cached LangChain model instance."""
        cache_key = f"{provider_config.base_url}:{model}"

        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        # Get API key
        api_key = "dummy"  # Default fallback
        if provider_config.api_key_env:
            api_key = os.getenv(provider_config.api_key_env, "dummy")

        # Set up timeouts
        timeouts = provider_config.timeouts_ms or self.config.timeouts_ms
        timeout_seconds = timeouts.read / 1000

        # Create LangChain OpenAI model with custom field extraction
        # ChatOpenAIWithCustomFields expects pydantic.SecretStr for api_key; wrap string if present
        langchain_model = ChatOpenAIWithCustomFields(
            model=model,
            api_key=SecretStr(api_key) if api_key is not None else None,
            base_url=provider_config.base_url,
            timeout=timeout_seconds,
            stream_usage=True,
            use_responses_api=True,
            output_version="responses/v1",
        ).configurable_fields(
            use_responses_api=ConfigurableField(id="use_responses_api")
        )

        self._model_cache[cache_key] = langchain_model
        return langchain_model

    def _convert_to_langchain_messages(
        self, anthropic_request: dict[str, Any]
    ) -> list[BaseMessage]:
        """
        Convert Anthropic Messages API payload into LangChain message objects with a single
        content handling pipeline per message:
        - Adds a SystemMessage if system is provided
        - For each message, aggregates text content blocks
        - Emits ToolMessage entries for any tool_result blocks
        - For assistant messages, collects tool_use blocks into AIMessage.tool_calls
        - Emits a HumanMessage (role=user) or AIMessage (role=assistant) if text and/or tool_calls exist
        """

        def _text_block(text: str) -> dict[str, str]:
            return {"type": "text", "text": text}

        messages: list[BaseMessage] = []

        # Conversation messages
        for msg in anthropic_request.get("messages", []):
            role = (msg.get("role") or "").lower()
            content = msg.get("content")

            if content is None:
                continue

            content_parts: list[str | dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            # Normalize to iterable of blocks
            blocks = content if isinstance(content, list) else [content]

            for block in blocks:
                if isinstance(block, dict):
                    btype = block.get("type")

                    if btype == "text":
                        content_parts.append(_text_block(block.get("text", "")))

                    elif btype == "thinking":
                        # thinking = block.get("thinking", "")
                        # if thinking:
                        #     thinking = "<think>" + thinking + "</think>"
                        #     content_parts.append(_text_block(thinking))
                        pass

                    elif btype == "tool_use":
                        # Only valid on assistant turns; collect into tool_calls
                        if role == "assistant":
                            tool_calls.append(
                                {
                                    "name": block.get("name", ""),
                                    "args": block.get("input", {}) or {},
                                    "id": block.get("id", ""),
                                }
                            )

                    elif btype == "tool_result":
                        # Emit ToolMessage immediately (independent of user/assistant role)
                        content_data = block.get("content")
                        tool_content, is_error = self._format_tool_result_content(
                            content_data
                        )
                        messages.append(
                            ToolMessage(
                                content=tool_content,
                                tool_call_id=block.get("tool_use_id", "")
                                or block.get("id", ""),
                                status="error" if is_error else "success",
                            )
                        )

                    elif btype == "image":
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
                            "type": "image_url",
                            "image_url": image_url,
                        }
                        content_parts.append(image_content)
                        pass

                    else:
                        # Fallback unknown block -> stringify
                        content_parts.append(
                            _text_block(
                                block.get("text", "") if "text" in block else str(block)
                            )
                        )

                elif isinstance(block, str):
                    content_parts.append(_text_block(block))

                else:
                    # Any other primitive or object -> stringify
                    content_parts.append(_text_block(str(block)))

            if role == "system":
                if content_parts:
                    messages.append(SystemMessage(content=content_parts))

            elif role == "user":
                if content_parts:
                    messages.append(HumanMessage(content=content_parts))

            elif role == "assistant":
                # Emit AIMessage even if only tool_calls exist (content can be empty string)
                messages.append(
                    AIMessage(
                        content=content_parts,
                        tool_calls=[] if tool_calls is None else tool_calls,
                    )
                )

            else:
                # Unknown role: default to HumanMessage with provided content parts
                if content_parts:
                    messages.append(HumanMessage(content=content_parts))

        return messages

    def _format_tool_result_content(self, content: Any) -> tuple[str, bool]:
        """Format tool result content and detect if it's an error.

        Returns:
            tuple: (formatted_content, is_error)
        """
        is_error = False

        if isinstance(content, str):
            return content, is_error
        elif isinstance(content, dict):
            # Check for error indicator
            if content.get("is_error", False):
                is_error = True
                error_content = content.get(
                    "content", content.get("error", "Tool execution failed")
                )
                return str(error_content), is_error
            else:
                return json.dumps(content), is_error
        elif isinstance(content, list):
            return json.dumps(content), is_error
        else:
            return str(content), is_error

    def _prepare_openai_request(
        self,
        messages: list[BaseMessage],
        anthropic_request: dict[str, Any],
        model: str,
        model_config: dict[str, Any | ModelConfigEntry] | None = None,
    ) -> dict[str, Any]:
        """Prepare a unified LangChain-executable payload for OpenAI models."""
        # Tools to be bound via LangChain
        tools = anthropic_request.get("tools") or []
        lc_tools = self._convert_tools(tools) if tools else []

        # Collect model call parameters to bind on the LC model
        params: dict[str, Any] = {}
        if "temperature" in anthropic_request:
            params["temperature"] = anthropic_request["temperature"]
        if "top_p" in anthropic_request:
            params["top_p"] = anthropic_request["top_p"]
        if "max_tokens" in anthropic_request:
            params["max_tokens"] = anthropic_request["max_tokens"]
        if "stop_sequences" in anthropic_request:
            params["stop"] = anthropic_request["stop_sequences"]

        # Apply model configuration overrides with proper priority handling
        if model_config:
            params = self.router._apply_granular_config_overrides(params, model_config)

        return {
            "model": model,
            "messages": messages,  # LangChain BaseMessage list
            "tools": lc_tools,  # OpenAI-style function tool schema for tool calling
            "params": params,  # applied via .bind(**params)
            "stream": anthropic_request.get("stream", False),
        }

    def _convert_tools(
        self, anthropic_tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """
        Convert Anthropic tools to OpenAI format.

        LangChain provides unified tool conversion across providers.
        """
        openai_tools = []
        for tool in anthropic_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            # openai_tool = convert_to_openai_tool(tool=tool)
            openai_tools.append(openai_tool)

        return openai_tools

    async def make_request(
        self,
        adapted_request: dict[str, Any],
        headers: dict[str, str],
        provider_config: ProviderConfig,
        use_responses_api: bool = True,
    ) -> AsyncIterator[BaseMessageChunk] | BaseMessage:
        """Make the actual API request using LangChain."""

        try:
            # Execute via LangChain SDK
            lc_messages: list[BaseMessage] = adapted_request["messages"]
            stream: bool = bool(adapted_request.get("stream", False))
            params: dict[str, Any] = adapted_request.get("params", {}) or {}
            tools: list[dict[str, Any]] = adapted_request.get("tools", []) or []
            target_model: str = adapted_request.get("model", "")

            # Build/lookup LC model and apply params + tools
            lc_model = self._get_langchain_model(provider_config, target_model)

            # Bind any model call parameters (temperature, max_tokens, etc)
            lc_model = lc_model.with_config(
                config={"configurable": {"use_responses_api": use_responses_api}},
            )

            if params:
                lc_model = lc_model.bind(**params)

            if tools:
                # langchain runnable should forward the bound model attributes
                bind_tools_method = getattr(lc_model, "bind_tools", None)
                if not bind_tools_method:
                    raise Exception("Unexpected Langchain behavior.")
                bind_tools_method(tools=tools)

            logger.info(
                "Invoking LangChain ChatOpenAI",
                base_url=provider_config.base_url,
                model=target_model,
                stream=stream,
                has_tools=bool(tools),
            )

            if stream:
                # Return the async event stream (LangChain yields events/messages)
                return cast(
                    AsyncIterator[BaseMessageChunk], lc_model.astream(lc_messages)
                )
            else:
                # Return the AIMessage (LangChain message object)
                return await lc_model.ainvoke(lc_messages)

        except Exception as e:
            logger.exception(
                "API request failed",
                error=str(e),
                api_type="responses" if use_responses_api else "chat_completions",
                model=adapted_request.get("model"),
            )
            raise

    async def close(self) -> None:
        """Clean up resources."""
        self._model_cache.clear()
