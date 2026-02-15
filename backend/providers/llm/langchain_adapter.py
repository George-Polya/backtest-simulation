"""
LangChain LLM Provider Adapter.

Implements the LLMProvider interface using LangChain's ChatOpenAI with OpenRouter.
Provides an alternative integration path for those already using LangChain ecosystem.

API Documentation: https://openrouter.ai/docs
LangChain Documentation: https://python.langchain.com/docs/integrations/chat/openai
"""

from decimal import Decimal
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from openai import APIConnectionError, APIStatusError, RateLimitError as OpenAIRateLimitError

from backend.core.config import LLMConfig, Settings
from backend.providers.llm.base import (
    AuthenticationError,
    GenerationConfig,
    GenerationResult,
    LLMProvider,
    LLMProviderError,
    ModelInfo,
    ModelNotFoundError,
    RateLimitError,
)
from backend.providers.llm.model_limits import resolve_openrouter_limits

# OpenRouter API base URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model costs (per 1k tokens, in USD)
DEFAULT_MODEL_COSTS: dict[str, tuple[Decimal, Decimal]] = {
    "anthropic/claude-3.5-sonnet": (Decimal("0.003"), Decimal("0.015")),
    "anthropic/claude-sonnet-4": (Decimal("0.003"), Decimal("0.015")),
    "anthropic/claude-3-opus": (Decimal("0.015"), Decimal("0.075")),
    "anthropic/claude-3-haiku": (Decimal("0.00025"), Decimal("0.00125")),
    "openai/gpt-4o": (Decimal("0.005"), Decimal("0.015")),
    "openai/gpt-4o-mini": (Decimal("0.00015"), Decimal("0.0006")),
    "moonshotai/kimi-k2-thinking": (Decimal("0.0006"), Decimal("0.002")),
}


class LangChainAdapter(LLMProvider):
    """
    LangChain-based LLM Provider implementation using OpenRouter.

    Uses LangChain's ChatOpenAI with base_url set to OpenRouter for unified
    access to models from multiple providers (Anthropic, OpenAI, etc.).

    This adapter is useful when you want to leverage LangChain's ecosystem
    (chains, agents, memory, etc.) while using OpenRouter as the backend.

    Attributes:
        _client: LangChain ChatOpenAI client configured for OpenRouter
        _llm_config: LLM configuration from settings
        _model_info: Cached model metadata

    Example:
        adapter = LangChainAdapter(
            api_key="sk-or-v1-...",
            llm_config=settings.llm,
        )
        result = await adapter.generate("Write a haiku about coding")
    """

    def __init__(
        self,
        api_key: str,
        llm_config: LLMConfig,
    ) -> None:
        """
        Initialize the LangChain adapter.

        Args:
            api_key: OpenRouter API key
            llm_config: LLM configuration containing model and generation params

        Raises:
            ValueError: If api_key is empty
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        self._api_key = api_key
        self._llm_config = llm_config
        (
            self._resolved_max_context_tokens,
            self._resolved_max_output_tokens,
        ) = resolve_openrouter_limits(
            model_id=llm_config.model,
            configured_max_tokens=llm_config.max_tokens,
            configured_max_context_tokens=llm_config.max_context_tokens,
        )
        self._model_info = self._build_model_info()

        # Build default headers for OpenRouter
        default_headers: dict[str, str] = {}
        if llm_config.site_url:
            default_headers["HTTP-Referer"] = llm_config.site_url
        if llm_config.site_name:
            default_headers["X-Title"] = llm_config.site_name

        # Initialize LangChain ChatOpenAI with OpenRouter base URL
        self._client = ChatOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=self._resolved_max_output_tokens,
            default_headers=default_headers if default_headers else None,
        )

    def _build_model_info(self) -> ModelInfo:
        """Build ModelInfo from configuration."""
        model_id = self._llm_config.model
        costs = DEFAULT_MODEL_COSTS.get(
            model_id, (Decimal("0"), Decimal("0"))
        )

        return ModelInfo(
            model_id=model_id,
            provider="langchain",
            display_name=model_id.split("/")[-1] if "/" in model_id else model_id,
            max_context_tokens=self._resolved_max_context_tokens,
            max_output_tokens=self._resolved_max_output_tokens,
            cost_per_1k_input=costs[0],
            cost_per_1k_output=costs[1],
            supports_system_prompt=True,
            supports_streaming=True,
        )

    def _build_messages(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> list[SystemMessage | HumanMessage]:
        """
        Build LangChain messages list for the API request.

        Args:
            prompt: User prompt/message
            system_prompt: Optional system prompt

        Returns:
            List of LangChain message objects
        """
        messages: list[SystemMessage | HumanMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        messages.append(HumanMessage(content=prompt))

        return messages

    def _build_extra_body(self, config: GenerationConfig) -> dict[str, Any] | None:
        """
        Build extra_body for OpenRouter-specific parameters.

        Args:
            config: Generation configuration

        Returns:
            Extra body dict or None if no extra parameters
        """
        extra: dict[str, Any] = {}

        # Add custom provider parameters from config.extra (except local override key)
        if config.extra:
            web_search_override = config.extra.pop("web_search_enabled", None)
            extra.update(config.extra)
            # Restore original config.extra so caller config is not mutated
            if web_search_override is not None:
                config.extra["web_search_enabled"] = web_search_override

        # Thinking models: OpenRouter reasoning token controls
        if self._llm_config.reasoning_enabled:
            reasoning_tokens = (
                self._llm_config.reasoning_max_tokens
                or (config.max_tokens if config else self._resolved_max_output_tokens)
            )
            extra["reasoning"] = {
                "enabled": True,
                "max_tokens": reasoning_tokens,
            }

        # OpenRouter web search plugin (supports per-request override)
        web_search_enabled = (
            config.extra.get("web_search_enabled")
            if config.extra and "web_search_enabled" in config.extra
            else self._llm_config.web_search_enabled
        )
        if web_search_enabled:
            web_plugin: dict[str, Any] = {
                "id": "web",
                "max_results": self._llm_config.web_search_max_results,
            }
            if self._llm_config.web_search_prompt:
                web_plugin["search_prompt"] = self._llm_config.web_search_prompt

            existing_plugins = extra.get("plugins", [])
            existing_plugins.append(web_plugin)
            extra["plugins"] = existing_plugins

        return extra if extra else None

    async def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """
        Generate text using LangChain ChatOpenAI with OpenRouter.

        Args:
            prompt: The user prompt/message
            config: Generation configuration. Uses settings defaults if None.
            system_prompt: Optional system prompt

        Returns:
            GenerationResult containing the generated text and metadata

        Raises:
            LLMProviderError: If generation fails
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If API key is invalid
        """
        if config is None:
            config = GenerationConfig(
                temperature=self._llm_config.temperature,
                max_tokens=self._resolved_max_output_tokens,
                seed=self._llm_config.seed,
            )

        messages = self._build_messages(prompt, system_prompt)
        extra_body = self._build_extra_body(config)

        try:
            # Create a bound client with custom config if different from defaults
            client = self._client
            if (
                config.temperature != self._llm_config.temperature
                or config.max_tokens != self._resolved_max_output_tokens
            ):
                # Create new client with updated parameters
                client = self._client.bind(
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )

            # Apply additional parameters if set
            invoke_kwargs: dict[str, Any] = {}
            if config.stop_sequences:
                invoke_kwargs["stop"] = config.stop_sequences

            if config.top_p != 1.0:
                invoke_kwargs["top_p"] = config.top_p

            if config.frequency_penalty != 0.0:
                invoke_kwargs["frequency_penalty"] = config.frequency_penalty

            if config.presence_penalty != 0.0:
                invoke_kwargs["presence_penalty"] = config.presence_penalty

            if config.seed is not None:
                invoke_kwargs["seed"] = config.seed

            if extra_body:
                invoke_kwargs["extra_body"] = extra_body

            # Make API call using LangChain
            response = await client.ainvoke(messages, **invoke_kwargs)

            # Extract content
            content = str(response.content) if response.content else ""

            # Thinking models may return reasoning metadata while content is empty
            if not content and hasattr(response, "additional_kwargs"):
                reasoning = response.additional_kwargs.get("reasoning", {})
                if isinstance(reasoning, dict):
                    if isinstance(reasoning.get("summary"), list):
                        summary_parts: list[str] = []
                        for part in reasoning["summary"]:
                            if isinstance(part, dict):
                                text = part.get("text")
                                if text:
                                    summary_parts.append(str(text))
                        if summary_parts:
                            content = "\n".join(summary_parts)
                    if not content:
                        summary_text = reasoning.get("text") or reasoning.get("summary")
                        if isinstance(summary_text, str):
                            content = summary_text

            # Extract usage statistics from response metadata
            usage_dict: dict[str, int] = {}
            if hasattr(response, "response_metadata"):
                token_usage = response.response_metadata.get("token_usage", {})
                if token_usage:
                    usage_dict = {
                        "prompt_tokens": token_usage.get("prompt_tokens", 0),
                        "completion_tokens": token_usage.get("completion_tokens", 0),
                        "total_tokens": token_usage.get("total_tokens", 0),
                    }

            # Extract finish reason
            finish_reason = "stop"
            if hasattr(response, "response_metadata"):
                finish_reason = response.response_metadata.get("finish_reason", "stop")

            # Build raw response for debugging
            raw_response: dict[str, Any] = {}
            if hasattr(response, "response_metadata"):
                raw_response = dict(response.response_metadata)
            if hasattr(response, "id"):
                raw_response["id"] = response.id

            return GenerationResult(
                content=content,
                model_info=self._model_info,
                usage=usage_dict,
                finish_reason=finish_reason,
                raw_response=raw_response,
            )

        except OpenAIRateLimitError as e:
            raise RateLimitError(
                f"Rate limit exceeded: {e}",
                provider="langchain",
                retry_after=None,
            ) from e
        except APIStatusError as e:
            if e.status_code == 401 or e.status_code == 403:
                raise AuthenticationError(
                    f"Authentication failed: {e.message}",
                    provider="langchain",
                ) from e
            if e.status_code == 404:
                raise ModelNotFoundError(
                    f"Model not found: {e.message}",
                    provider="langchain",
                ) from e
            raise LLMProviderError(
                f"API error ({e.status_code}): {e.message}",
                provider="langchain",
            ) from e
        except APIConnectionError as e:
            raise LLMProviderError(
                f"Connection failed: {e}",
                provider="langchain",
            ) from e
        except Exception as e:
            error_msg = str(e).lower()

            # Fallback handling for non-openai wrapped exceptions
            if "rate limit" in error_msg or "429" in str(e):
                raise RateLimitError(
                    f"Rate limit exceeded: {e}",
                    provider="langchain",
                    retry_after=None,
                ) from e

            if "401" in str(e) or "403" in str(e) or "unauthorized" in error_msg:
                raise AuthenticationError(
                    f"Authentication failed: {e}",
                    provider="langchain",
                ) from e

            if "404" in str(e) or "model not found" in error_msg:
                raise ModelNotFoundError(
                    f"Model not found: {e}",
                    provider="langchain",
                ) from e

            raise LLMProviderError(
                f"LangChain generation failed: {e}",
                provider="langchain",
            ) from e

    def get_model_info(self) -> ModelInfo:
        """Get metadata about the configured model."""
        return self._model_info

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "langchain"

    async def health_check(self) -> bool:
        """
        Check if the LangChain/OpenRouter connection is working.

        Makes a lightweight request to verify connectivity and authentication.

        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Simple test message
            messages = [HumanMessage(content="test")]
            await self._client.ainvoke(messages, max_tokens=5)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """
        Close the LangChain client.

        LangChain's ChatOpenAI doesn't require explicit cleanup,
        but this method is provided for interface consistency.
        """
        # LangChain ChatOpenAI doesn't have an explicit close method
        pass

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
    ) -> "LangChainAdapter":
        """
        Create adapter from application settings.

        Convenience factory method for creating the adapter with proper
        configuration from the global settings object.

        Args:
            settings: Application settings

        Returns:
            Configured LangChainAdapter instance

        Raises:
            ValueError: If OpenRouter API key is not configured
        """
        api_key = settings.get_llm_api_key()
        return cls(
            api_key=api_key,
            llm_config=settings.llm,
        )
