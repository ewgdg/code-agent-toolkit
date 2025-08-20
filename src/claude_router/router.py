import re
from typing import Any
from urllib.parse import parse_qs

import structlog

from .config import Config

logger = structlog.get_logger(__name__)


class RouterDecision:
    def __init__(
        self,
        target: str,
        model: str,
        reason: str = "",
        model_config: dict[str, Any] | None = None,
    ):
        self.target = target  # "openai" or "anthropic"
        self.model = model  # OpenAI model to use if target is "openai"
        self.reason = reason
        self.model_config = model_config or {}  # Model config from query params


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
                provider, model, model_config = self._parse_provider_model(
                    override.model
                )
                logger.debug(
                    f"Override rule {i + 1} MATCHED",
                    provider=provider,
                    model=model,
                    model_config=model_config,
                )
                return RouterDecision(
                    target=provider,
                    model=model,
                    reason=f"Override rule {i + 1} matched: {override.when}",
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

    def _parse_provider_model(
        self, provider_model_string: str
    ) -> tuple[str, str, dict[str, Any]]:
        """
        Parse provider/model format with optional query parameters
        and return (target, model, config).

        Examples:
        - "openai/gpt-5" -> ("openai", "gpt-5", {})
        - "anthropic/claude-3-sonnet" -> ("anthropic", "claude-3-sonnet", {})
        - "gpt-4" -> ("openai", "gpt-4", {})  # fallback to openai if no provider
        - "openai/gpt-5?reasoning.effort=low&temperature=0.5" ->
          ("openai", "gpt-5", {"reasoning": {"effort": "low"}, "temperature": 0.5})
        """
        model_config: dict[str, Any] = {}

        # Check if there are query parameters
        if "?" in provider_model_string:
            base_model, query_string = provider_model_string.split("?", 1)

            # Parse query parameters
            query_params = parse_qs(query_string)

            # Convert query parameters to nested configuration
            for key, values in query_params.items():
                # Use the first value if multiple values are provided
                value = values[0] if values else ""

                # Handle parameters (both nested and direct)
                # e.g., "reasoning.effort" -> {"reasoning": {"effort": value}}
                # e.g., "temperature" -> {"temperature": value}
                parts = key.split(".")
                current_dict = model_config

                # Navigate through nested structure, creating dicts as needed
                for part in parts[:-1]:
                    if part not in current_dict:
                        current_dict[part] = {}
                    current_dict = current_dict[part]

                # Set the final value, attempting to convert to appropriate type
                final_key = parts[-1]
                current_dict[final_key] = self._convert_param_value(value)

            provider_model_string = base_model

        # Parse provider/model as before
        if "/" in provider_model_string:
            provider, model = provider_model_string.split("/", 1)
            return provider.lower(), model, model_config
        else:
            # Fallback: assume OpenAI if no provider specified
            return "openai", provider_model_string, model_config

    def _convert_param_value(self, value: str) -> Any:
        """
        Convert query parameter value to appropriate Python type.

        Attempts to convert strings to:
        - int if it's a valid integer
        - float if it's a valid float
        - bool if it's "true"/"false" (case-insensitive)
        - original string otherwise
        """
        if not value:
            return value

        # Try boolean conversion first (case-insensitive)
        if value.lower() in ("true", "false"):
            return value.lower() == "true"

        # Try integer conversion
        try:
            if "." not in value:  # Only try int if no decimal point
                return int(value)
        except ValueError:
            pass

        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass

        # Return as string if no conversion worked
        return value

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
