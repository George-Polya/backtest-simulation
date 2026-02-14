"""
Configuration loader for the backtesting service.

Loads configuration from config.yaml and environment variables using pydantic-settings.
Supports llm/data/execution sections as described in PRD.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENROUTER = "openrouter"
    LANGCHAIN = "langchain"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class DataProvider(str, Enum):
    """Supported data providers."""

    YFINANCE = "yfinance"
    MOCK = "mock"
    SUPABASE = "supabase"
    LOCAL = "local"


class ExecutionProvider(str, Enum):
    """Supported code execution providers."""

    DOCKER = "docker"
    LOCAL = "local"


class LLMConfig(BaseModel):
    """
    LLM provider configuration.

    Note: API keys should NOT be stored here.
    Use environment variables (OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY).
    """

    provider: LLMProvider = Field(
        default=LLMProvider.OPENROUTER,
        description="LLM provider to use",
    )
    model: str = Field(
        default="anthropic/claude-3.5-sonnet",
        description="Model identifier",
    )
    site_url: Optional[str] = Field(
        default=None,
        description="Site URL for OpenRouter HTTP-Referer header",
    )
    site_name: Optional[str] = Field(
        default=None,
        description="Site name for OpenRouter X-Title header",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Temperature for generation (0.0 = most deterministic)",
    )
    seed: int = Field(
        default=42,
        ge=0,
        description="Random seed for reproducible LLM outputs (deterministic by default)",
    )
    max_tokens: int = Field(
        default=8000,
        gt=0,
        description="Maximum tokens to generate",
    )
    max_context_tokens: int = Field(
        default=128000,
        gt=0,
        description="Maximum context tokens the model supports",
    )
    # Reasoning/Thinking model support
    reasoning_enabled: bool = Field(
        default=False,
        description="Enable reasoning mode for thinking models (o1, deepseek-r1, kimi-k2-thinking, etc.)",
    )
    reasoning_max_tokens: Optional[int] = Field(
        default=None,
        description="Max tokens for reasoning. If None, uses max_tokens value.",
    )
    # Web search support (OpenRouter only)
    # See: https://openrouter.ai/announcements/introducing-web-search-via-the-api
    web_search_enabled: bool = Field(
        default=False,
        description="Enable web search for real-time information retrieval (OpenRouter only)",
    )
    web_search_max_results: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of web search results to fetch (1-10)",
    )
    web_search_prompt: Optional[str] = Field(
        default=None,
        description="Custom prompt for integrating web search results into the response",
    )


class DataConfig(BaseModel):
    """
    Data provider configuration.

    Supports primary provider selection with fallback chain for resilience.
    When the primary provider fails, the system automatically tries fallback
    providers in order.

    Example config.yaml:
        data:
          provider: yfinance
          fallback_providers:
            - mock
          cache_ttl_seconds: 300
          enable_caching: true
          retry_attempts: 3
          retry_delay_seconds: 1.0
    """

    provider: DataProvider = Field(
        default=DataProvider.YFINANCE,
        description="Primary data provider to use",
    )
    fallback_providers: list[DataProvider] = Field(
        default_factory=list,
        description="Fallback providers in order of preference",
    )
    cache_ttl_seconds: int = Field(
        default=300,
        ge=0,
        description="Cache TTL in seconds for data provider responses",
    )
    enable_caching: bool = Field(
        default=True,
        description="Enable caching of data provider responses",
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of retry attempts before falling back",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.0,
        le=30.0,
        description="Delay between retry attempts in seconds",
    )
    local_storage_path: str = Field(
        default="./data/prices",
        description="Local CSV storage path (used when provider='local')",
    )

    @field_validator("fallback_providers")
    @classmethod
    def validate_fallback_providers(
        cls, v: list[DataProvider], info
    ) -> list[DataProvider]:
        """
        Validate fallback providers don't include the primary provider.
        """
        # Note: We can't access 'provider' field here directly in Pydantic v2
        # The validation will be done at Settings level
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for p in v:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique

    def get_all_providers(self) -> list[DataProvider]:
        """
        Get list of all providers (primary + fallbacks) in order.

        Returns:
            List of DataProvider enums starting with primary.
        """
        providers = [self.provider]
        for fb in self.fallback_providers:
            if fb not in providers:
                providers.append(fb)
        return providers


class ExecutionConfig(BaseModel):
    """Code execution configuration."""

    provider: ExecutionProvider = Field(
        default=ExecutionProvider.DOCKER,
        description="Code execution provider to use",
    )
    fallback_to_local: bool = Field(
        default=True,
        description="Fallback to local execution if Docker fails",
    )
    docker_image: str = Field(
        default="backtest-runner:latest",
        description="Docker image for backtest execution (must have pandas, backtesting, etc.)",
    )
    docker_socket_url: Optional[str] = Field(
        default=None,
        description="Docker socket URL (e.g., 'unix:///var/run/docker.sock'). "
                    "If None, uses aiodocker default. "
                    "For macOS Docker Desktop, try 'unix://$HOME/.docker/run/docker.sock'",
    )
    timeout: int = Field(
        default=300,
        gt=0,
        le=600,
        description="Execution timeout in seconds",
    )
    memory_limit: str = Field(
        default="2g",
        description="Memory limit for execution (Docker format)",
    )
    allowed_modules: list[str] = Field(
        default_factory=lambda: ["pandas", "numpy"],
        description="Allowed Python modules for execution",
    )


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and config.yaml.

    Priority (highest to lowest):
    1. Environment variables (from .env file or system)
    2. config.yaml file
    3. Default values

    Secrets (loaded from .env only - NEVER commit to git):
        LLM API Keys:
        - OPENROUTER_API_KEY: OpenRouter API key
        - ANTHROPIC_API_KEY: Anthropic Claude API key
        - OPENAI_API_KEY: OpenAI API key
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # Application settings
    app_name: str = Field(
        default="Natural Language Backtesting Service",
        description="Application name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version",
    )
    debug: bool = Field(
        default=False,
        description="Debug mode",
        alias="APP_DEBUG",
    )

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug_bool(cls, v):
        """Handle empty string as False for boolean debug field."""
        if v == "" or v is None:
            return False
        if isinstance(v, str):
            return v.lower() in ("true", "1", "yes", "on")
        return v

    # LLM API Keys (from .env)
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key",
    )

    # Supabase credentials (from .env)
    supabase_url: str = Field(
        default="",
        description="Supabase project URL",
    )
    supabase_anon_key: str = Field(
        default="",
        description="Supabase anonymous key",
    )

    # Configuration sections (from config.yaml)
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM provider configuration",
    )
    data: DataConfig = Field(
        default_factory=DataConfig,
        description="Data provider configuration",
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Code execution configuration",
    )

    def get_llm_api_key(self) -> str:
        """
        Get the appropriate API key based on the configured LLM provider.

        Returns:
            API key for the current LLM provider.

        Raises:
            ValueError: If no API key is configured for the provider.
        """
        provider = self.llm.provider
        key_map = {
            LLMProvider.OPENROUTER: self.openrouter_api_key,
            LLMProvider.LANGCHAIN: self.openrouter_api_key,  # LangChain uses OpenRouter
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
        }
        api_key = key_map.get(provider, "")
        if not api_key:
            raise ValueError(
                f"No API key configured for provider '{provider.value}'. "
                f"Set the {provider.value.upper()}_API_KEY environment variable."
            )
        return api_key

    @classmethod
    def from_yaml(cls, config_path: Path | str | None = None) -> "Settings":
        """
        Load settings from a YAML configuration file.

        Args:
            config_path: Path to config.yaml file. If None, looks for config.yaml
                        in the current directory and project root.

        Returns:
            Settings instance with values from YAML merged with env vars.
        """
        config_data: dict = {}

        if config_path is None:
            # Look for config.yaml in common locations
            search_paths = [
                Path.cwd() / "config.yaml",
                Path(__file__).parent.parent.parent / "config.yaml",
            ]
            for path in search_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path is not None:
            config_path = Path(config_path)
            if config_path.exists():
                with open(config_path, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f) or {}

        return cls(**config_data)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are only loaded once.
    Call get_settings.cache_clear() to reload settings.

    Returns:
        Cached Settings instance.
    """
    return Settings.from_yaml()
