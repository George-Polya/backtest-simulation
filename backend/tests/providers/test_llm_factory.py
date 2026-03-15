"""
Tests for LLM provider factory.

Tests:
- Factory creates correct adapter based on settings
- Factory raises error for unsupported providers
- Factory registration/unregistration
"""

from unittest.mock import MagicMock

import pytest

from backend.core.config import LLMConfig, LLMProvider as LLMProviderEnum, Settings
from backend.providers.llm.base import LLMProvider, LLMProviderError
from backend.providers.llm.factory import LLMProviderFactory
from backend.providers.llm.langchain_adapter import LangChainAdapter


@pytest.fixture
def langchain_settings() -> Settings:
    """Create settings configured for LangChain."""
    settings = MagicMock(spec=Settings)
    settings.llm = LLMConfig(
        provider=LLMProviderEnum.LANGCHAIN,
        model="anthropic/claude-3.5-sonnet",
        temperature=0.2,
        max_tokens=8000,
    )
    settings.get_llm_api_key.return_value = "sk-or-v1-test-key"
    return settings


class TestLLMProviderFactory:
    """Tests for LLMProviderFactory."""

    def test_create_langchain_adapter(
        self,
        langchain_settings: Settings,
    ) -> None:
        """Test factory creates LangChainAdapter for LangChain provider."""
        provider = LLMProviderFactory.create(langchain_settings)

        assert isinstance(provider, LangChainAdapter)
        assert provider.provider_name == "langchain"

    def test_create_raises_llm_provider_error_on_failure(self) -> None:
        """Test factory wraps creation errors in LLMProviderError."""
        settings = MagicMock(spec=Settings)
        settings.llm = LLMConfig(provider=LLMProviderEnum.LANGCHAIN)
        # Empty API key will raise ValueError in LangChainAdapter
        settings.get_llm_api_key.return_value = ""

        with pytest.raises(LLMProviderError, match="Failed to create LLM provider"):
            LLMProviderFactory.create(settings)

    def test_get_supported_providers(self) -> None:
        """Test getting list of supported providers."""
        providers = LLMProviderFactory.get_supported_providers()

        assert "langchain" in providers
        assert len(providers) == 1
        assert isinstance(providers, list)

    def test_is_provider_supported(self) -> None:
        """Test checking if provider is supported."""
        assert LLMProviderFactory.is_provider_supported(LLMProviderEnum.LANGCHAIN)

    def test_register_custom_provider_override(self) -> None:
        """Test registering a custom provider factory can override existing."""
        from backend.providers.llm.factory import _PROVIDER_REGISTRY

        # Save original factory
        original_factory = _PROVIDER_REGISTRY.get(LLMProviderEnum.LANGCHAIN)

        class MockProvider(LLMProvider):
            async def generate(self, prompt, config=None, system_prompt=None):
                pass

            def get_model_info(self):
                pass

            @property
            def provider_name(self):
                return "mock"

        def mock_factory(settings):
            return MockProvider()

        try:
            # Override with mock
            LLMProviderFactory.register(LLMProviderEnum.LANGCHAIN, mock_factory)

            settings = MagicMock(spec=Settings)
            settings.llm = LLMConfig(provider=LLMProviderEnum.LANGCHAIN)
            settings.get_llm_api_key.return_value = "test"

            provider = LLMProviderFactory.create(settings)
            assert isinstance(provider, MockProvider)
        finally:
            # Restore original
            if original_factory:
                LLMProviderFactory.register(LLMProviderEnum.LANGCHAIN, original_factory)

    def test_unregister_and_reregister_provider(self) -> None:
        """Test unregistering and re-registering a provider."""
        from backend.providers.llm.factory import _PROVIDER_REGISTRY

        # Save original factory
        original_factory = _PROVIDER_REGISTRY.get(LLMProviderEnum.LANGCHAIN)

        try:
            # Unregister
            result = LLMProviderFactory.unregister(LLMProviderEnum.LANGCHAIN)
            assert result is True
            assert not LLMProviderFactory.is_provider_supported(LLMProviderEnum.LANGCHAIN)

            # Re-unregister should return False
            result = LLMProviderFactory.unregister(LLMProviderEnum.LANGCHAIN)
            assert result is False
        finally:
            # Restore original
            if original_factory:
                LLMProviderFactory.register(LLMProviderEnum.LANGCHAIN, original_factory)
