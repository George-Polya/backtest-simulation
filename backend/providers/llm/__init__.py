"""
LLM Provider abstraction layer.

Provides provider-agnostic interfaces for LLM operations following SOLID DIP.

Exports:
    - ModelInfo: Metadata about the LLM model
    - GenerationConfig: Configuration for text generation
    - GenerationResult: Result of an LLM generation request
    - LLMProvider: Abstract base class for LLM providers
    - LLMProviderFactory: Factory for creating LLM provider instances
    - OpenRouterAdapter: OpenRouter API adapter (OpenAI SDK)
    - LangChainAdapter: LangChain-based adapter
    - AnthropicAdapter: Native Anthropic (Claude) adapter
    - OpenAIAdapter: Native OpenAI adapter
    - Exceptions: LLMProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
"""

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
from backend.providers.llm.anthropic_adapter import AnthropicAdapter
from backend.providers.llm.factory import LLMProviderFactory
from backend.providers.llm.langchain_adapter import LangChainAdapter
from backend.providers.llm.openai_adapter import OpenAIAdapter
from backend.providers.llm.openrouter import OpenRouterAdapter

__all__ = [
    # Core types
    "ModelInfo",
    "GenerationConfig",
    "GenerationResult",
    "LLMProvider",
    # Factory
    "LLMProviderFactory",
    # Adapters
    "OpenRouterAdapter",
    "LangChainAdapter",
    "AnthropicAdapter",
    "OpenAIAdapter",
    # Exceptions
    "LLMProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
]
