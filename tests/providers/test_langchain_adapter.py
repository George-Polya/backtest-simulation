"""
Tests for LangChain LLM adapter.

Tests:
- Adapter initialization and configuration
- Request building
- Response parsing
- Error handling
"""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import APIConnectionError, APIStatusError, RateLimitError as OpenAIRateLimitError

from backend.core.config import LLMConfig, LLMProvider as LLMProviderEnum
from backend.providers.llm.base import (
    AuthenticationError,
    GenerationConfig,
    LLMProviderError,
    ModelNotFoundError,
    RateLimitError,
)
from backend.providers.llm.langchain_adapter import LangChainAdapter


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLM config."""
    return LLMConfig(
        provider=LLMProviderEnum.LANGCHAIN,
        model="anthropic/claude-3.5-sonnet",
        site_url="https://test.com",
        site_name="Test App",
        temperature=0.2,
        max_tokens=8000,
    )


@pytest.fixture
def adapter(llm_config: LLMConfig) -> LangChainAdapter:
    """Create a LangChain adapter for testing."""
    return LangChainAdapter(
        api_key="sk-or-v1-test-key",
        llm_config=llm_config,
    )


class TestLangChainAdapterInit:
    """Tests for LangChainAdapter initialization."""

    def test_init_success(self, llm_config: LLMConfig) -> None:
        """Test successful adapter initialization."""
        adapter = LangChainAdapter(
            api_key="sk-or-v1-test-key",
            llm_config=llm_config,
        )
        assert adapter.provider_name == "langchain"

    def test_init_empty_api_key_raises(self, llm_config: LLMConfig) -> None:
        """Test that empty API key raises ValueError."""
        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            LangChainAdapter(
                api_key="",
                llm_config=llm_config,
            )

    def test_init_creates_chat_openai_client(self, llm_config: LLMConfig) -> None:
        """Test that adapter creates ChatOpenAI client."""
        adapter = LangChainAdapter(
            api_key="sk-or-v1-test-key",
            llm_config=llm_config,
        )
        # Client should be created
        assert adapter._client is not None


class TestModelInfo:
    """Tests for model info retrieval."""

    def test_get_model_info(self, adapter: LangChainAdapter) -> None:
        """Test getting model info."""
        info = adapter.get_model_info()

        assert info.model_id == "anthropic/claude-3.5-sonnet"
        assert info.provider == "langchain"
        assert info.display_name == "claude-3.5-sonnet"
        assert info.max_output_tokens == 8000
        assert info.cost_per_1k_input == Decimal("0.003")
        assert info.cost_per_1k_output == Decimal("0.015")

    def test_get_model_info_uses_model_defaults_when_limits_omitted(self) -> None:
        """Test model-specific limits are used when config omits token limits."""
        config = LLMConfig(
            provider=LLMProviderEnum.LANGCHAIN,
            model="minimax/minimax-m2.5",
            temperature=0.2,
        )
        adapter = LangChainAdapter(
            api_key="sk-or-v1-test-key",
            llm_config=config,
        )
        info = adapter.get_model_info()

        assert info.max_context_tokens == 204800
        assert info.max_output_tokens == 131072


class TestMessageBuilding:
    """Tests for message building."""

    def test_build_messages_basic(self, adapter: LangChainAdapter) -> None:
        """Test basic message building."""
        messages = adapter._build_messages("Hello")

        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Hello"

    def test_build_messages_with_system_prompt(
        self, adapter: LangChainAdapter
    ) -> None:
        """Test message building with system prompt."""
        messages = adapter._build_messages("Hello", system_prompt="Be helpful")

        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "Be helpful"
        assert isinstance(messages[1], HumanMessage)
        assert messages[1].content == "Hello"


class TestGenerate:
    """Tests for generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(self, llm_config: LLMConfig) -> None:
        """Test successful generation."""
        # Mock the LangChain client's ainvoke method
        mock_response = AIMessage(
            content="Generated text",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
                "finish_reason": "stop",
            },
        )
        mock_response.id = "chatcmpl-123"

        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            result = await adapter.generate("Test prompt")

        assert result.content == "Generated text"
        assert result.finish_reason == "stop"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 5

    @pytest.mark.asyncio
    async def test_generate_with_custom_config(self, llm_config: LLMConfig) -> None:
        """Test generation with custom config."""
        mock_response = AIMessage(
            content="Response",
            response_metadata={"finish_reason": "stop"},
        )

        # Mock both the original client and the bound client
        mock_bound = MagicMock()
        mock_bound.ainvoke = AsyncMock(return_value=mock_response)

        mock_client = MagicMock()
        mock_client.bind = MagicMock(return_value=mock_bound)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            custom_config = GenerationConfig(temperature=0.9, max_tokens=100)
            await adapter.generate("Test", config=custom_config)

        # Verify bind was called with custom parameters
        mock_client.bind.assert_called_once_with(temperature=0.9, max_tokens=100)
        mock_bound.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_passes_openai_params(self, llm_config: LLMConfig) -> None:
        """Test generation forwards OpenAI-compatible sampling params."""
        mock_response = AIMessage(
            content="Response",
            response_metadata={"finish_reason": "stop"},
        )
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            cfg = GenerationConfig(
                top_p=0.9,
                frequency_penalty=0.2,
                presence_penalty=0.1,
                seed=123,
            )
            await adapter.generate("Test", config=cfg)

        kwargs = mock_client.ainvoke.call_args.kwargs
        assert kwargs["top_p"] == 0.9
        assert kwargs["frequency_penalty"] == 0.2
        assert kwargs["presence_penalty"] == 0.1
        assert kwargs["seed"] == 123

    @pytest.mark.asyncio
    async def test_generate_passes_reasoning_and_web_search_extra_body(
        self,
    ) -> None:
        """Test OpenRouter-specific reasoning/web search parameters are sent."""
        llm_config = LLMConfig(
            provider=LLMProviderEnum.LANGCHAIN,
            model="moonshotai/kimi-k2-thinking",
            temperature=0.2,
            reasoning_enabled=True,
            reasoning_max_tokens=2048,
            web_search_enabled=True,
            web_search_max_results=3,
            web_search_prompt="Use reliable sources only",
        )
        mock_response = AIMessage(
            content="Response",
            response_metadata={"finish_reason": "stop"},
        )
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            await adapter.generate("Test")

        kwargs = mock_client.ainvoke.call_args.kwargs
        assert "extra_body" in kwargs
        extra_body = kwargs["extra_body"]
        assert extra_body["reasoning"] == {"enabled": True, "max_tokens": 2048}
        assert extra_body["plugins"][0]["id"] == "web"
        assert extra_body["plugins"][0]["max_results"] == 3
        assert extra_body["plugins"][0]["search_prompt"] == "Use reliable sources only"

    @pytest.mark.asyncio
    async def test_generate_content_fallback_from_reasoning(
        self,
        llm_config: LLMConfig,
    ) -> None:
        """Test empty content is recovered from reasoning metadata."""
        mock_response = AIMessage(
            content="",
            additional_kwargs={
                "reasoning": {
                    "summary": [
                        {"type": "summary_text", "text": "Reasoning output"},
                    ]
                }
            },
            response_metadata={"finish_reason": "stop"},
        )
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            result = await adapter.generate("Test")

        assert result.content == "Reasoning output"

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, llm_config: LLMConfig) -> None:
        """Test generation with system prompt."""
        mock_response = AIMessage(
            content="Response",
            response_metadata={"finish_reason": "stop"},
        )

        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            await adapter.generate("Test", system_prompt="Be helpful")

        # Verify messages include system prompt
        call_args = mock_client.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert isinstance(messages[0], SystemMessage)
        assert messages[0].content == "Be helpful"

    @pytest.mark.asyncio
    async def test_generate_rate_limit_error(self, llm_config: LLMConfig) -> None:
        """Test rate limit error handling."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 429
        error = OpenAIRateLimitError(
            message="Rate limited",
            response=mock_response,
            body={"error": {"message": "Rate limited"}},
        )
        mock_client.ainvoke = AsyncMock(side_effect=error)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            with pytest.raises(RateLimitError, match="Rate limit exceeded"):
                await adapter.generate("Test")

    @pytest.mark.asyncio
    async def test_generate_authentication_error(self, llm_config: LLMConfig) -> None:
        """Test authentication error handling."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 401
        error = APIStatusError(
            message="Invalid API key",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )
        mock_client.ainvoke = AsyncMock(side_effect=error)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            with pytest.raises(AuthenticationError, match="Authentication failed"):
                await adapter.generate("Test")

    @pytest.mark.asyncio
    async def test_generate_model_not_found_error(self, llm_config: LLMConfig) -> None:
        """Test model not found error handling."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 404
        error = APIStatusError(
            message="Model not found",
            response=mock_response,
            body={"error": {"message": "Model not found"}},
        )
        mock_client.ainvoke = AsyncMock(side_effect=error)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            with pytest.raises(ModelNotFoundError):
                await adapter.generate("Test")

    @pytest.mark.asyncio
    async def test_generate_connection_error(self, llm_config: LLMConfig) -> None:
        """Test connection error handling."""
        mock_client = MagicMock()
        error = APIConnectionError(request=MagicMock())
        mock_client.ainvoke = AsyncMock(side_effect=error)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            with pytest.raises(LLMProviderError, match="Connection failed"):
                await adapter.generate("Test")

    @pytest.mark.asyncio
    async def test_generate_generic_error(self, llm_config: LLMConfig) -> None:
        """Test generic error handling."""
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            with pytest.raises(LLMProviderError, match="LangChain generation failed"):
                await adapter.generate("Test")


class TestHealthCheck:
    """Tests for health check functionality."""

    @pytest.mark.asyncio
    async def test_health_check_success(self, llm_config: LLMConfig) -> None:
        """Test successful health check."""
        mock_response = AIMessage(content="ok")
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(return_value=mock_response)

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            result = await adapter.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, llm_config: LLMConfig) -> None:
        """Test failed health check returns False."""
        mock_client = MagicMock()
        mock_client.ainvoke = AsyncMock(side_effect=Exception("Connection failed"))

        with patch(
            "backend.providers.llm.langchain_adapter.ChatOpenAI",
            return_value=mock_client,
        ):
            adapter = LangChainAdapter(
                api_key="test-key",
                llm_config=llm_config,
            )
            result = await adapter.health_check()

        assert result is False


class TestClose:
    """Tests for close functionality."""

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self, llm_config: LLMConfig) -> None:
        """Test that close completes without error."""
        adapter = LangChainAdapter(
            api_key="test-key",
            llm_config=llm_config,
        )

        # Should not raise any exceptions
        await adapter.close()
