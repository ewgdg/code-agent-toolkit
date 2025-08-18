from typing import Any

from pydantic import BaseModel, Field, validator


class RouterConfig(BaseModel):
    listen: str = Field(default="0.0.0.0:8787", description="Host:port to listen on")
    original_base_url: str = Field(default="https://api.anthropic.com")
    openai_base_url: str = Field(default="https://api.openai.com")


class ModeDetectionConfig(BaseModel):
    header: str = Field(default="X-Claude-Code-Mode")
    plan_values: list[str] = Field(default=["plan", "planning"])


class MappingConfig(BaseModel):
    plan_model: str = Field(default="gpt-4o")
    background_model: str = Field(default="gpt-4o-mini")


class ReasoningThresholds(BaseModel):
    low_max: int = Field(
        default=5000,
        description="Max tokens for low effort reasoning (1-5K: simple tasks)",
    )
    medium_max: int = Field(
        default=15000,
        description="Max tokens for medium effort reasoning (5K-15K: balanced tasks)",
    )

    @validator("low_max", "medium_max")
    def validate_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Reasoning thresholds must be positive")
        return v

    @validator("medium_max")
    def validate_medium_greater_than_low(cls, v: int, values: dict[str, Any]) -> int:
        if "low_max" in values and v <= values["low_max"]:
            raise ValueError("medium_max must be greater than low_max")
        return v


class OpenAIConfig(BaseModel):
    api_key_env: str = Field(default="OPENAI_API_KEY")
    reasoning_effort_default: str = Field(default="minimal")
    supports_reasoning: bool = Field(default=False)
    reasoning_thresholds: ReasoningThresholds = Field(
        default_factory=ReasoningThresholds
    )

    @validator("reasoning_effort_default")
    def validate_reasoning_effort(cls, v: str) -> str:
        if v not in ["minimal", "low", "medium", "high"]:
            raise ValueError(
                "reasoning_effort_default must be 'minimal', 'low', 'medium', or 'high'"
            )
        return v


class TimeoutsConfig(BaseModel):
    connect: int = Field(default=5000, description="Connect timeout in milliseconds")
    read: int = Field(default=600000, description="Read timeout in milliseconds")

    @validator("connect", "read")
    def validate_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Timeout must be positive")
        return v


class LoggingConfig(BaseModel):
    level: str = Field(default="info")

    @validator("level")
    def validate_log_level(cls, v: str) -> str:
        if v.lower() not in ["debug", "info", "warning", "error", "critical"]:
            raise ValueError("Invalid log level")
        return v.lower()


class OverrideRule(BaseModel):
    when: dict[str, Any] = Field(description="Conditions for this rule")
    model: str = Field(description="Model to use when conditions match")


class Config(BaseModel):
    router: RouterConfig = Field(default_factory=RouterConfig)
    mode_detection: ModeDetectionConfig = Field(default_factory=ModeDetectionConfig)
    haiku_matchers: list[str] = Field(default=["haiku", "claude-3-haiku"])
    mapping: MappingConfig = Field(default_factory=MappingConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    timeouts_ms: TimeoutsConfig = Field(default_factory=TimeoutsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    overrides: list[OverrideRule] = Field(default_factory=list)

    class Config:
        extra = "forbid"  # Prevent unknown fields
