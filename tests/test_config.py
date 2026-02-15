"""
Tests for the configuration loader.
"""

from pathlib import Path

import pytest

from backend.core.config import (
    DataConfig,
    DataProvider,
    ExecutionConfig,
    ExecutionProvider,
    LLMConfig,
    LLMProvider,
    Settings,
    get_settings,
)


class TestLLMConfig:
    """Tests for LLMConfig model."""

    def test_default_values(self) -> None:
        """Test default LLM configuration values."""
        config = LLMConfig()
        assert config.provider == LLMProvider.OPENROUTER
        assert config.model == "anthropic/claude-3.5-sonnet"
        assert config.temperature == 0.2
        assert config.max_tokens is None
        assert config.max_context_tokens is None

    def test_custom_values(self) -> None:
        """Test custom LLM configuration values."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-opus",
            temperature=0.5,
            max_tokens=4000,
        )
        assert config.provider == LLMProvider.ANTHROPIC
        assert config.model == "claude-3-opus"
        assert config.temperature == 0.5
        assert config.max_tokens == 4000

    def test_temperature_validation(self) -> None:
        """Test temperature must be between 0 and 2."""
        with pytest.raises(ValueError):
            LLMConfig(temperature=-0.1)
        with pytest.raises(ValueError):
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation(self) -> None:
        """Test max_tokens must be positive."""
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=0)
        with pytest.raises(ValueError):
            LLMConfig(max_tokens=-100)


class TestDataConfig:
    """Tests for DataConfig model."""

    def test_default_values(self) -> None:
        """Test default data configuration values."""
        config = DataConfig()
        assert config.provider == DataProvider.YFINANCE
        assert config.fallback_providers == []
        assert config.cache_ttl_seconds == 300


class TestExecutionConfig:
    """Tests for ExecutionConfig model."""

    def test_default_values(self) -> None:
        """Test default execution configuration values."""
        config = ExecutionConfig()
        assert config.provider == ExecutionProvider.DOCKER
        assert config.timeout == 300
        assert config.memory_limit == "2g"
        assert "pandas" in config.allowed_modules
        assert "numpy" in config.allowed_modules

    def test_timeout_validation(self) -> None:
        """Test timeout must be between 1 and 600."""
        with pytest.raises(ValueError):
            ExecutionConfig(timeout=0)
        with pytest.raises(ValueError):
            ExecutionConfig(timeout=601)


class TestSettings:
    """Tests for Settings model."""

    def test_default_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default settings values."""
        # Clear any existing APP_DEBUG env var to test true defaults
        monkeypatch.delenv("APP_DEBUG", raising=False)

        settings = Settings()
        assert settings.app_name == "Natural Language Backtesting Service"
        assert settings.app_version == "0.1.0"
        assert settings.debug is False
        assert isinstance(settings.llm, LLMConfig)
        assert isinstance(settings.data, DataConfig)
        assert isinstance(settings.execution, ExecutionConfig)

    def test_load_from_yaml(self, temp_config_file: Path) -> None:
        """Test loading settings from YAML file."""
        settings = Settings.from_yaml(temp_config_file)

        # Check LLM config
        assert settings.llm.provider == LLMProvider.OPENROUTER
        assert settings.llm.model == "anthropic/claude-3.5-sonnet"

        # Check data config
        assert settings.data.provider == DataProvider.MOCK

        # Check execution config
        assert settings.execution.provider == ExecutionProvider.LOCAL
        assert settings.execution.timeout == 60

    def test_load_from_nonexistent_yaml(self, tmp_path: Path) -> None:
        """Test loading settings when YAML file doesn't exist uses defaults."""
        nonexistent_path = tmp_path / "nonexistent.yaml"
        settings = Settings.from_yaml(nonexistent_path)

        # Should use defaults
        assert settings.llm.provider == LLMProvider.OPENROUTER
        assert settings.data.provider == DataProvider.YFINANCE

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test environment variables override config values."""
        monkeypatch.setenv("APP_DEBUG", "true")
        monkeypatch.setenv("OPENROUTER_API_KEY", "test_key_from_env")

        settings = Settings()
        assert settings.debug is True
        assert settings.openrouter_api_key == "test_key_from_env"

    def test_default_cors_origins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test default CORS origins include common local frontend ports."""
        monkeypatch.delenv("APP_CORS_ORIGINS", raising=False)

        settings = Settings()
        assert "http://localhost:3000" in settings.cors_origins
        assert "http://localhost:5173" in settings.cors_origins

    def test_cors_origins_csv_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CORS origins can be provided as comma-separated env value."""
        monkeypatch.setenv(
            "APP_CORS_ORIGINS",
            "https://app.example.com, https://admin.example.com",
        )

        settings = Settings()
        assert settings.cors_origins == [
            "https://app.example.com",
            "https://admin.example.com",
        ]

    def test_cors_origins_json_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test CORS origins can be provided as JSON array env value."""
        monkeypatch.setenv(
            "APP_CORS_ORIGINS",
            '["https://app.example.com", "https://admin.example.com"]',
        )

        settings = Settings()
        assert settings.cors_origins == [
            "https://app.example.com",
            "https://admin.example.com",
        ]


class TestGetSettings:
    """Tests for get_settings function."""

    def test_caching(self) -> None:
        """Test that get_settings returns cached instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2

    def test_cache_clear(self) -> None:
        """Test that cache can be cleared."""
        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()
        # After clearing cache, should be different instance
        assert settings1 is not settings2
