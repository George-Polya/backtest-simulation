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
from backend.providers.llm.langchain_client import (
    build_chat_openai_kwargs,
    build_model_info,
    build_openrouter_extra_body,
    OPENROUTER_BASE_URL,
    resolve_generation_config,
)
from backend.providers.llm.model_limits import resolve_openrouter_limits


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
            model_id=llm_config.model_name,
            configured_max_tokens=llm_config.max_tokens,
            configured_max_context_tokens=llm_config.max_context_tokens,
        )
        self._model_info = build_model_info(
            llm_config,
            resolved_max_context_tokens=self._resolved_max_context_tokens,
            resolved_max_output_tokens=self._resolved_max_output_tokens,
        )

        # Initialize LangChain ChatOpenAI with OpenRouter base URL
        self._client = ChatOpenAI(
            **build_chat_openai_kwargs(
                api_key=api_key,
                llm_config=llm_config,
                config=resolve_generation_config(
                    llm_config,
                    None,
                    resolved_max_output_tokens=self._resolved_max_output_tokens,
                ),
                resolved_max_output_tokens=self._resolved_max_output_tokens,
                include_extra_body=False,
            )
        )

    def _build_model_info(self) -> ModelInfo:
        """Build ModelInfo from configuration."""
        return build_model_info(
            self._llm_config,
            resolved_max_context_tokens=self._resolved_max_context_tokens,
            resolved_max_output_tokens=self._resolved_max_output_tokens,
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
        return build_openrouter_extra_body(
            self._llm_config,
            config,
            resolved_max_output_tokens=self._resolved_max_output_tokens,
        )

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
            config = resolve_generation_config(
                self._llm_config,
                None,
                resolved_max_output_tokens=self._resolved_max_output_tokens,
            )

        # Resolve None temperature to the provider's configured default
        effective_temperature = (
            config.temperature
            if config.temperature is not None
            else self._llm_config.temperature
        )

        messages = self._build_messages(prompt, system_prompt)
        extra_body = self._build_extra_body(config)

        try:
            # Create a bound client with custom config if different from defaults
            client = self._client
            if (
                effective_temperature != self._llm_config.temperature
                or config.max_tokens != self._resolved_max_output_tokens
            ):
                # Create new client with updated parameters
                client = ChatOpenAI(
                    **build_chat_openai_kwargs(
                        api_key=self._api_key,
                        llm_config=self._llm_config,
                        config=config,
                        resolved_max_output_tokens=self._resolved_max_output_tokens,
                        include_extra_body=False,
                    )
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
