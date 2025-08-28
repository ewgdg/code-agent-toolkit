import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.claude_router.config import Config, ConfigLoader
from src.claude_router.config.schema import (
    ModelConfigEntry,
    OverrideRule,
    WhenCondition,
)


class TestConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.router.listen == "0.0.0.0:8787"
        assert config.router.original_base_url == "https://api.anthropic.com"

        assert config.openai.reasoning_effort_default == "minimal"
        assert config.openai.reasoning_thresholds.low_max == 5000
        assert config.openai.reasoning_thresholds.medium_max == 15000

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid reasoning effort
        with pytest.raises(ValidationError):
            Config(openai={"reasoning_effort_default": "invalid"})

        # Test invalid timeout
        with pytest.raises(ValidationError):
            Config(timeouts_ms={"connect": -1})

        # Test invalid threshold order
        with pytest.raises(ValidationError):
            Config(
                openai={"reasoning_thresholds": {"low_max": 10000, "medium_max": 5000}}
            )

    def test_config_loader_defaults(self):
        """Test config loader with non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "nonexistent.yaml"
            loader = ConfigLoader(config_path)
            config = loader.get_config()

            # Should return default config
            assert config.router.listen == "0.0.0.0:8787"

    def test_config_loader_from_file(self):
        """Test config loader from YAML file."""
        config_data = {
            "router": {
                "listen": "127.0.0.1:9999",
                "original_base_url": "https://custom.anthropic.com",
            },
            "openai": {"reasoning_effort_default": "high"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            config = loader.get_config()

            assert config.router.listen == "127.0.0.1:9999"
            assert config.router.original_base_url == "https://custom.anthropic.com"
            assert config.openai.reasoning_effort_default == "high"

        finally:
            config_path.unlink()

    def test_invalid_yaml_fallback(self):
        """Test fallback to defaults with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            config = loader.get_config()

            # Should fallback to defaults
            assert config.router.listen == "0.0.0.0:8787"

        finally:
            config_path.unlink()

    def test_model_config_entry_validation(self):
        """Test validation of ModelConfigEntry schema."""
        # Test ModelConfigEntry validation
        entry = ModelConfigEntry(value="low")
        assert entry.value == "low"
        assert entry.when is None  # default value

        entry = ModelConfigEntry(value=0.3, when=WhenCondition(current_in=[None, 0.1]))
        assert entry.value == 0.3
        assert entry.when.current_in == [None, 0.1]

    def test_override_rule_with_model_config(self):
        """Test OverrideRule with model_config field."""
        override_rule = OverrideRule(
            when={"request": {"model_regex": "test"}},
            model="openai/gpt-5",
            config={
                "reasoning": {
                    "effort": ModelConfigEntry(
                        value="low", when=WhenCondition(current_not_in=["low", "high"])
                    )
                }
            },
        )

        assert override_rule.when == {"request": {"model_regex": "test"}}
        assert override_rule.model == "openai/gpt-5"
        assert override_rule.config["reasoning"]["effort"].value == "low"
        assert override_rule.config["reasoning"]["effort"].when.current_not_in == [
            "low",
            "high",
        ]

    def test_config_with_model_config_overrides(self):
        """Test full Config with model_config overrides in YAML."""
        config_data = {
            "overrides": [
                {
                    "when": {"request": {"model_regex": "gpt-5"}},
                    "model": "openai/gpt-5",
                    "config": {
                        "reasoning": {"effort": "low"},  # no conditions = always apply
                    },
                },
                {
                    "when": {"request": {"model_regex": "haiku"}},
                    "model": "openai/gpt-4o-mini",
                    "config": {
                        "temperature": {
                            "value": 0.3,
                            "when": {"current_not_equals": 0.3},
                        },
                        "max_tokens": 1000,  # no conditions = always apply
                    },
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loader = ConfigLoader(config_path)
            config = loader.get_config()

            assert len(config.overrides) == 2

            # First override
            first_rule = config.overrides[0]
            assert first_rule.model == "openai/gpt-5"
            assert first_rule.config == {"reasoning": {"effort": "low"}}

            # Second override
            second_rule = config.overrides[1]
            assert second_rule.model == "openai/gpt-4o-mini"
            # The YAML parser creates a dict, not a ModelConfigEntry directly
            # This is expected behavior - the granular logic will handle both formats
            assert second_rule.config["temperature"] == {
                "value": 0.3,
                "when": {"current_not_equals": 0.3},
            }
            assert second_rule.config["max_tokens"] == 1000

        finally:
            config_path.unlink()
