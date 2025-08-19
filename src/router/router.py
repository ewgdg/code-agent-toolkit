import re
from typing import Any

import structlog

from .config import Config

logger = structlog.get_logger(__name__)


class RouterDecision:
    def __init__(self, target: str, model: str, reason: str = ""):
        self.target = target  # "openai" or "anthropic"
        self.model = model  # OpenAI model to use if target is "openai"
        self.reason = reason


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

            if self._matches_override_condition(override.when, headers, request_data):
                provider, model = self._parse_provider_model(override.model)
                logger.debug(
                    f"Override rule {i + 1} MATCHED", provider=provider, model=model
                )
                return RouterDecision(
                    target=provider,
                    model=model,
                    reason=f"Override rule {i + 1} matched: {override.when}",
                )
            else:
                logger.debug(f"Override rule {i + 1} did NOT match")

        logger.debug("No override rules matched")
        return None

    def _matches_override_condition(
        self,
        condition: dict[str, Any],
        headers: dict[str, str],
        request_data: dict[str, Any],
    ) -> bool:
        """Check if override condition matches current request."""

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

                    try:
                        if isinstance(expected_value, str):
                            if not re.search(expected_value, actual_value, re.IGNORECASE):
                                logger.debug("Model regex condition failed")
                                return False
                        elif isinstance(expected_value, list):
                            if not any(
                                re.search(pattern, actual_value, re.IGNORECASE)
                                for pattern in expected_value
                            ):
                                logger.debug("Model regex condition failed (list)")
                                return False
                    except re.error as e:
                        logger.error("Invalid regex pattern", pattern=expected_value, error=str(e))
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
                    )

                    try:
                        # Check patterns against all system content parts
                        if isinstance(expected_value, str):
                            pattern_found = any(
                                re.search(expected_value, part, re.IGNORECASE)
                                for part in system_parts
                            )
                            if not pattern_found:
                                logger.debug("System regex condition failed")
                                return False
                        elif isinstance(expected_value, list):
                            pattern_found = any(
                                any(re.search(pattern, part, re.IGNORECASE) for part in system_parts)
                                for pattern in expected_value
                            )
                            if not pattern_found:
                                logger.debug("System regex condition failed (list)")
                                return False
                    except re.error as e:
                        logger.error("Invalid regex pattern", pattern=expected_value, error=str(e))
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
        Parse provider/model format and return (target, model).

        Examples:
        - "openai/gpt-5" -> ("openai", "gpt-5")
        - "anthropic/claude-3-sonnet" -> ("anthropic", "claude-3-sonnet")
        - "gpt-4" -> ("openai", "gpt-4")  # fallback to openai if no provider
        """
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
