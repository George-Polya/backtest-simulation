"""
LLM Provider abstraction layer.

Provides provider-agnostic interfaces for LLM operations following SOLID DIP.

Exports:
    - ModelInfo: Metadata about the LLM model
    - GenerationConfig: Configuration for text generation
    - GenerationResult: Result of an LLM generation request
    - LLMProvider: Abstract base class for LLM providers
    - LLMProviderFactory: Factory for creating LLM provider instances
    - LangChainAdapter: LangChain-based adapter (OpenRouter API)
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
from backend.providers.llm.factory import LLMProviderFactory
from backend.providers.llm.langchain_adapter import LangChainAdapter

__all__ = [
    # Core types
    "ModelInfo",
    "GenerationConfig",
    "GenerationResult",
    "LLMProvider",
    # Factory
    "LLMProviderFactory",
    # Adapters
    "LangChainAdapter",
    # Exceptions
    "LLMProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
]
