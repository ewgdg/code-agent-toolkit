import tempfile
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.router.config import Config, ConfigLoader


class TestConfig:
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.router.listen == "0.0.0.0:8787"
        assert config.router.original_base_url == "https://api.anthropic.com"
        assert config.router.openai_base_url == "https://api.openai.com"

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
