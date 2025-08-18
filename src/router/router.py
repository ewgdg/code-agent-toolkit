from typing import Any

import structlog

from .config import Config

logger = structlog.get_logger(__name__)


class RouterDecision:
    def __init__(self, target: str, model: str, reasoning: str = ""):
        self.target = target  # "openai" or "anthropic"
        self.model = model    # OpenAI model to use if target is "openai"
        self.reasoning = reasoning


class ModelRouter:
    def __init__(self, config: Config):
        self.config = config

    def decide_route(
        self,
        headers: dict[str, str],
        request_data: dict[str, Any]
    ) -> RouterDecision:
        """
        Decide routing based on headers and request data.
        
        Returns RouterDecision with target ("openai" or "anthropic") and model.
        """

        # Check override rules first (highest precedence)
        override_decision = self._check_overrides(headers, request_data)
        if override_decision:
            return override_decision

        # Check for ExitPlanMode tool call (indicates plan mode)
        if self.is_plan_mode(request_data):
            return RouterDecision(
                target="openai",
                model=self.config.mapping.plan_model,
                reasoning="Plan mode detected via ExitPlanMode tool call",
            )

        # Check for plan mode via header
        mode_header = self.config.mode_detection.header
        if mode_header in headers:
            mode_value = headers[mode_header].lower()
            if mode_value in self.config.mode_detection.plan_values:
                return RouterDecision(
                    target="openai",
                    model=self.config.mapping.plan_model,
                    reasoning=f"Plan mode detected via header {mode_header}={mode_value}"
                )

        # Check model name for haiku patterns
        model_name = request_data.get("model", "").lower()
        if model_name:
            for matcher in self.config.haiku_matchers:
                if matcher.lower() in model_name:
                    return RouterDecision(
                        target="openai",
                        model=self.config.mapping.background_model,
                        reasoning=f"Haiku model detected: {model_name} matches {matcher}"
                    )

        # Default: passthrough to Anthropic
        return RouterDecision(
            target="anthropic",
            model="passthrough",
            reasoning="No routing rules matched, using passthrough",
        )

    def _check_overrides(
        self,
        headers: dict[str, str],
        request_data: dict[str, Any]
    ) -> RouterDecision | None:
        """Check override rules for routing decisions."""

        for override in self.config.overrides:
            if self._matches_override_condition(override.when, headers, request_data):
                provider, model = override.model.split("/", 1)
                # Determine target based on model
                return RouterDecision(
                    target=provider,
                    model=model,
                    reasoning=f"Override rule matched: {override.when}",
                )

        return None

    def _matches_override_condition(
        self,
        condition: dict[str, Any],
        headers: dict[str, str],
        request_data: dict[str, Any]
    ) -> bool:
        """Check if override condition matches current request."""

        # Check header conditions
        if "header" in condition:
            header_conditions = condition["header"]
            for header_name, expected_value in header_conditions.items():
                actual_value = headers.get(header_name, "")
                if isinstance(expected_value, str):
                    if actual_value.lower() != expected_value.lower():
                        return False
                elif isinstance(expected_value, list):
                    if actual_value.lower() not in [v.lower() for v in expected_value]:
                        return False

        # Check request data conditions
        if "request" in condition:
            request_conditions = condition["request"]
            for field_name, expected_value in request_conditions.items():
                actual_value = request_data.get(field_name)
                if actual_value != expected_value:
                    return False

        return True

    def is_plan_mode(self, request_data: dict[str, Any]) -> bool:
        """
        Detect if request contains ExitPlanMode tool call indicating plan mode.

        Checks for ExitPlanMode in tool calls within message content.
        """
        messages = request_data.get("messages", [])

        for message in messages:
            if message.get("role") == "assistant":
                content = message.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if (
                            item.get("type") == "tool_use"
                            and item.get("name") == "ExitPlanMode"
                        ):
                            logger.info("Plan mode detected via ExitPlanMode tool call")
                            return True

        return False

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
