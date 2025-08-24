import re
from re import Pattern
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RouterConfig(BaseModel):
    listen: str = Field(default="0.0.0.0:8787", description="Host:port to listen on")
    original_base_url: str = Field(default="https://api.anthropic.com")


class ReasoningThresholds(BaseModel):
    low_max: int = Field(
        default=5000,
        description="Max tokens for low effort reasoning (1-5K: simple tasks)",
    )
    medium_max: int = Field(
        default=15000,
        description="Max tokens for medium effort reasoning (5K-15K: balanced tasks)",
    )

    @field_validator("low_max", "medium_max")
    @classmethod
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Reasoning thresholds must be positive")
        return v

    @field_validator("medium_max")
    @classmethod
    def validate_medium_greater_than_low(cls, v: int, info: Any) -> int:
        if (
            hasattr(info, "data")
            and "low_max" in info.data
            and v <= info.data["low_max"]
        ):
            raise ValueError("medium_max must be greater than low_max")
        return v


class OpenAIConfig(BaseModel):
    api_key_env: str = Field(default="OPENAI_API_KEY")
    reasoning_effort_default: str = Field(default="minimal")
    reasoning_thresholds: ReasoningThresholds = Field(
        default_factory=ReasoningThresholds
    )
    reasoning_model_prefixes: list[str] = Field(
        default=["gpt-5", "o4", "o"]
    )

    @field_validator("reasoning_effort_default")
    @classmethod
    def validate_reasoning_effort(cls, v: str) -> str:
        if v not in ["minimal", "low", "medium", "high"]:
            raise ValueError(
                "reasoning_effort_default must be 'minimal', 'low', 'medium', or 'high'"
            )
        return v


class TimeoutsConfig(BaseModel):
    connect: int = Field(default=5000, description="Connect timeout in milliseconds")
    read: int = Field(default=600000, description="Read timeout in milliseconds")

    @field_validator("connect", "read")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class LoggingConfig(BaseModel):
    level: str = Field(default="info")

    @field_validator("level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        if v.lower() not in ["debug", "info", "warning", "error", "critical"]:
            raise ValueError("Invalid log level")
        return v.lower()


class ProviderConfig(BaseModel):
    base_url: str = Field(description="Base URL for the provider API")
    adapter: str = Field(description="Adapter type: anthropic-passthrough, openai-responses, openai-chat-completions")
    api_key_env: str | None = Field(default=None, description="Environment variable name for API key")
    timeouts_ms: TimeoutsConfig | None = Field(default=None, description="Provider-specific timeouts")

    @field_validator("adapter")
    @classmethod
    def validate_adapter(cls, v: str) -> str:
        valid_adapters = ["anthropic-passthrough", "openai-responses", "openai-chat-completions"]
        if v not in valid_adapters:
            raise ValueError(f"adapter must be one of: {', '.join(valid_adapters)}")
        return v


class OverrideRule(BaseModel):
    when: dict[str, Any] = Field(description="Conditions for this rule")
    model: str = Field(description="Model to use when conditions match")
    provider: str | None = Field(default=None, description="Provider to use when conditions match")
    compiled_patterns: dict[str, Pattern[str]] = Field(
        default_factory=dict, exclude=True
    )

    def get_compiled_pattern(self, pattern: str) -> Pattern[str]:
        """Get or compile regex pattern."""
        if pattern not in self.compiled_patterns:
            self.compiled_patterns[pattern] = re.compile(pattern, re.IGNORECASE)
        return self.compiled_patterns[pattern]


class Config(BaseModel):
    router: RouterConfig = Field(default_factory=RouterConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict, description="Provider configurations")
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    timeouts_ms: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    overrides: list[OverrideRule] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")  # Prevent unknown fields
