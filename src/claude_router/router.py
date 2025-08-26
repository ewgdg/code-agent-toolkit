import re
from typing import Any

import structlog

from .config import Config

logger = structlog.get_logger(__name__)


class RouterDecision:
    def __init__(
        self,
        target: str,
        model: str,
        reason: str = "",
        provider: str | None = None,
        adapter: str | None = None,
        model_config: dict[str, Any] | None = None,
    ):
        self.target = target  # "openai" or "anthropic" (kept for backward compat)
        self.model = model  # Model to use
        self.reason = reason
        self.provider = provider or target  # Provider name, defaults to target
        self.adapter = adapter  # Adapter type
        self.model_config = model_config or {}  # Applied model config from overrides


class ModelRouter:
    def __init__(self, config: Config):
        self.config = config

    def decide_route(
        self, headers: dict[str, str], request_data: dict[str, Any]
    ) -> RouterDecision:
        """
        Decide routing based on headers and request data.

        Returns RouterDecision with target ("openai" or "anthropic") and model.
        """

        # Extract key request info for debugging
        model = request_data.get("model", "unknown")
        messages = request_data.get("messages", [])
        available_tools = self._extract_tools(request_data)

        logger.debug(
            "Routing decision started",
            model=model,
            message_count=len(messages),
            available_tools=available_tools,
            headers=dict(headers),
        )

        # Check override rules first (highest precedence)
        override_decision = self._check_overrides(headers, request_data)
        if override_decision:
            logger.debug(
                "Override rule matched",
                target=override_decision.target,
                model=override_decision.model,
                reason=override_decision.reason,
            )
            return override_decision

        # Default: passthrough to Anthropic
        logger.debug("Using default passthrough to Anthropic")
        return RouterDecision(
            target="anthropic",
            model="passthrough",
            reason="No routing rules matched, using passthrough",
            provider="anthropic",
            adapter="anthropic-passthrough",
            model_config={},
        )

    def _check_overrides(
        self, headers: dict[str, str], request_data: dict[str, Any]
    ) -> RouterDecision | None:
        """Check override rules for routing decisions."""

        logger.debug(f"Checking {len(self.config.overrides)} override rules")

        for i, override in enumerate(self.config.overrides):
            logger.info(
                f"Evaluating override rule {i + 1}",
                condition=override.when,
                target_model=override.model,
            )

            if self._matches_override_condition(override, headers, request_data):
                # Resolve provider: explicit > parsed provider > "anthropic"
                if override.provider:
                    resolved_provider = override.provider
                    model = override.model
                else:
                    resolved_provider, model = self._parse_provider_model(
                        override.model
                    )

                # Apply model config overrides if specified
                model_config: dict[str, Any] = {}
                if override.config:
                    model_config = self._apply_granular_config_overrides(
                        model_config, override.config
                    )

                # Look up adapter from provider config
                adapter = self._resolve_adapter(resolved_provider)

                logger.debug(
                    f"Override rule {i + 1} MATCHED",
                    provider=resolved_provider,
                    model=model,
                    adapter=adapter,
                    applied_config=model_config,
                    config_overrides=override.config,
                )
                return RouterDecision(
                    target=resolved_provider,
                    model=model,
                    reason=f"Override rule {i + 1} matched: {override.when}",
                    provider=resolved_provider,
                    adapter=adapter,
                    model_config=model_config,
                )
            else:
                logger.debug(f"Override rule {i + 1} did NOT match")

        logger.debug("No override rules matched")
        return None

    def _matches_override_condition(
        self,
        override_rule: Any,  # Matches the override rule type from config
        headers: dict[str, str],
        request_data: dict[str, Any],
    ) -> bool:
        """Check if override condition matches current request."""

        condition = override_rule.when
        logger.debug("Checking condition", condition=condition)

        # Check header conditions
        if "header" in condition:
            header_conditions = condition["header"]
            logger.debug(
                "Checking header conditions", header_conditions=header_conditions
            )

            for header_name, expected_value in header_conditions.items():
                actual_value = headers.get(header_name, "")
                logger.info(
                    "Header check",
                    name=header_name,
                    expected=expected_value,
                    actual=actual_value,
                )

                if isinstance(expected_value, str):
                    if actual_value.lower() != expected_value.lower():
                        logger.info("Header condition failed")
                        return False
                elif isinstance(expected_value, list):
                    if actual_value.lower() not in [v.lower() for v in expected_value]:
                        logger.info("Header condition failed (list)")
                        return False

        # Check request data conditions
        if "request" in condition:
            request_conditions = condition["request"]
            logger.debug(
                "Checking request conditions", request_conditions=request_conditions
            )

            for field_name, expected_value in request_conditions.items():
                # Handle special pattern matching
                if field_name == "model_regex":
                    # Check if model matches the regex pattern
                    actual_value = request_data.get("model", "")
                    logger.debug(
                        "Model regex check",
                        expected_pattern=expected_value,
                        actual_model=actual_value,
                    )

                    if isinstance(expected_value, str):
                        try:
                            compiled_pattern = override_rule.get_compiled_pattern(
                                expected_value
                            )
                            if not compiled_pattern.search(actual_value):
                                logger.debug("Model regex condition failed")
                                return False
                        except re.error as e:
                            logger.error(
                                "Invalid regex pattern",
                                pattern=expected_value,
                                error=str(e),
                            )
                            return False
                    else:
                        logger.error(
                            "model_regex must be a string, got",
                            type=type(expected_value),
                        )
                        return False

                elif field_name == "has_tool":
                    # Check if request contains specific tool
                    has_tool = self._has_tool(request_data, expected_value)
                    logger.debug(
                        "Tool check", tool_name=expected_value, has_tool=has_tool
                    )

                    if not has_tool:
                        logger.debug("Tool condition failed")
                        return False

                elif field_name == "system_regex":
                    # Check if system prompt matches regex patterns
                    system_parts = self._extract_system_content(request_data)
                    logger.debug(
                        "System regex check",
                        expected_patterns=expected_value,
                        system_parts_count=len(system_parts),
                        system_parts=[part[:100] + "..." for part in system_parts],
                    )

                    compiled_pattern = override_rule.get_compiled_pattern(
                        expected_value
                    )
                    pattern_found = any(
                        compiled_pattern.search(part) for part in system_parts
                    )
                    if not pattern_found:
                        logger.debug("System regex condition failed")
                        return False

                elif field_name == "user_regex":
                    # Check if user messages match regex patterns
                    user_parts = self._extract_user_content(request_data)
                    logger.debug(
                        "User regex check",
                        expected_patterns=expected_value,
                        user_parts_count=len(user_parts),
                        user_parts=[part[:100] + "..." for part in user_parts],
                    )

                    compiled_pattern = override_rule.get_compiled_pattern(
                        expected_value
                    )
                    pattern_found = any(
                        compiled_pattern.search(part) for part in user_parts
                    )
                    if not pattern_found:
                        logger.debug("User regex condition failed")
                        return False

                else:
                    # Standard equality check
                    actual_value = request_data.get(field_name, "")
                    logger.debug(
                        "Standard field check",
                        field=field_name,
                        expected=expected_value,
                        actual=actual_value,
                    )

                    if actual_value != expected_value:
                        logger.debug("Standard field condition failed")
                        return False

        logger.debug("All conditions passed")
        return True

    def _has_tool(self, request_data: dict[str, Any], tool_name: str) -> bool:
        """
        Check if request has a specific tool available in the tools list.
        """

        tools = request_data.get("tools", [])

        for tool in tools:
            if isinstance(tool, dict) and tool.get("name") == tool_name:
                logger.info(
                    "Tool found in tools list",
                    tool_name=tool_name,
                    tools_count=len(tools),
                )
                return True

        logger.info(
            "Tool NOT found in tools list", tool_name=tool_name, tools_count=len(tools)
        )
        return False

    def _extract_tools(self, request_data: dict[str, Any]) -> list[str]:
        """Extract available tool names for debugging."""
        tool_names = []

        # Extract available tool names from tools parameter
        tools = request_data.get("tools", [])
        for tool in tools:
            if isinstance(tool, dict) and "name" in tool:
                tool_names.append(tool["name"])

        return tool_names

    def _parse_provider_model(self, provider_model_string: str) -> tuple[str, str]:
        """
        Parse provider/model format and return (provider, model).

        Examples:
        - "openai/gpt-5" -> ("openai", "gpt-5")
        - "anthropic/claude-3-sonnet" -> ("anthropic", "claude-3-sonnet")
        - "gpt-4" -> ("openai", "gpt-4")  # fallback to openai if no provider
        """
        # Parse provider/model
        if "/" in provider_model_string:
            provider, model = provider_model_string.split("/", 1)
            return provider.lower(), model
        else:
            # Fallback: assume OpenAI if no provider specified
            return "openai", provider_model_string

    def get_reasoning_effort(self, request_data: dict[str, Any]) -> str:
        """
        Extract reasoning effort from request thinking.budget_tokens.

        Maps token ranges to effort levels:
        - 0 or None: minimal (no reasoning)
        - 1-5K: low (simple tasks)
        - 5K-15K: medium (balanced tasks)
        - 15K+: high (complex reasoning)
        """

        thinking = request_data.get("thinking", {})
        budget_tokens = thinking.get("budget_tokens")

        # Handle missing or zero budget tokens
        if budget_tokens is None or budget_tokens == 0:
            return "minimal"

        # Map token ranges to effort levels using configurable thresholds
        thresholds = self.config.openai.reasoning_thresholds

        if budget_tokens <= thresholds.low_max:
            return "low"
        elif budget_tokens <= thresholds.medium_max:
            return "medium"
        else:
            return "high"

    def _extract_system_content(self, request_data: dict[str, Any]) -> list[str]:
        """
        Extract system prompt content from request data as a list of text parts.

        Handles both string and list formats:
        - String: returns [string]
        - List: extracts 'text' fields from Anthropic format objects
        """

        system = request_data.get("system")

        if not system:
            return []

        if isinstance(system, str):
            return [system]

        if isinstance(system, list):
            # Handle Anthropic system format: list of objects with 'text' field
            text_parts = []
            for item in system:
                if isinstance(item, dict):
                    text_content = item.get("text", "")
                    if text_content:
                        text_parts.append(text_content)
                elif isinstance(item, str):
                    # Handle mixed list formats
                    text_parts.append(item)
            return text_parts

        # Fallback: convert to string and return as list
        return [str(system)]

    def _extract_user_content(self, request_data: dict[str, Any]) -> list[str]:
        """
        Extract user message content from the last user message in request data.

        Returns content as a list of text parts from the most recent user message.
        Handles Anthropic Messages API format where messages is a list of message
        objects with role and content fields.
        """
        messages = request_data.get("messages", [])
        user_parts = []

        # Find the last user message
        last_user_message = None
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user_message = message
                break

        if not last_user_message:
            return []

        content = last_user_message.get("content", "")

        if isinstance(content, str):
            user_parts.append(content)
        elif isinstance(content, list):
            # Handle content as list of blocks (text, image, etc.)
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content = block.get("text", "")
                    if text_content:
                        user_parts.append(text_content)
                elif isinstance(block, str):
                    # Handle mixed content formats
                    user_parts.append(block)

        return user_parts

    def _resolve_adapter(self, provider: str) -> str:
        """
        Resolve adapter type for a given provider.

        Returns adapter type from provider config, or default based on provider name.
        """
        # Look up in provider config first
        if provider in self.config.providers:
            return self.config.providers[provider].adapter

        # Default adapters for known providers
        if provider == "anthropic":
            return "anthropic-passthrough"
        elif provider == "openai":
            return "openai-responses"
        else:
            # For unknown providers, assume they're OpenAI-compatible
            logger.warning(
                f"Unknown provider '{provider}', defaulting to "
                "openai-chat-completions adapter"
            )
            return "openai-chat-completions"

    def _apply_granular_config_overrides(
        self, target: dict[str, Any], source: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Apply configuration overrides with granular priority control.

        Args:
            target: Target configuration dict
            source: Source configuration dict (may contain ModelConfigEntry objects)

        Returns:
            Updated configuration dict
        """
        from .config.schema import ModelConfigEntry

        result = target.copy()

        for key, value in source.items():
            if isinstance(value, ModelConfigEntry):
                # Handle ModelConfigEntry with explicit priority
                if key not in result or value.priority == "always":
                    result[key] = value.value
                # If priority is "default" and key exists, keep existing value
            elif isinstance(value, dict) and "value" in value and "priority" in value:
                # Handle dict representation of ModelConfigEntry (from YAML)
                if key not in result or value["priority"] == "always":
                    result[key] = value["value"]
                # If priority is "default" and key exists, keep existing value
            elif (
                isinstance(value, dict)
                and key in result
                and isinstance(result[key], dict)
            ):
                # Recursively handle nested dictionaries
                result[key] = self._apply_granular_config_overrides(result[key], value)
            elif isinstance(value, dict):
                # New nested dictionary with default priority
                if key not in result:
                    result[key] = self._apply_granular_config_overrides({}, value)
            else:
                # Direct value with default priority
                if key not in result:
                    result[key] = value
                # If key exists, keep existing value (default priority)

        return result

    def _deep_merge_configs(
        self,
        target: dict[str, Any],
        source: dict[str, Any],
        force_override: bool = True,
    ) -> dict[str, Any]:
        """
        Deep merge configuration dictionaries.

        Args:
            target: Target configuration dict (will be modified)
            source: Source configuration dict to merge from
            force_override: If True, override. If False, only set if missing.

        Returns:
            Merged configuration dict
        """
        result = target.copy()

        for key, value in source.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge_configs(
                    result[key], value, force_override
                )
            elif key not in result or force_override:
                # Set value if key doesn't exist OR if we're forcing override
                result[key] = value
            # If key exists and force_override is False, keep existing value

        return result
