"""
Shared abstractions and shared exceptions for code generation backends.

BacktestCodeGenerator keeps business orchestration responsibilities such as
ticker extraction, data availability checks, prompt assembly, response DTO
conversion, and domain error translation. A code generation backend is
responsible only for producing code/summary/ticker candidates from the prompt,
including any internal self-correction loop.
"""

from dataclasses import dataclass, field
from typing import Protocol

from backend.providers.llm.base import GenerationConfig, GenerationResult, ModelInfo


class CodeGenerationError(Exception):
    """Base exception for code generation errors."""

    def __init__(self, message: str, details: dict | None = None):
        self.details = details or {}
        super().__init__(message)


class ValidationError(CodeGenerationError):
    """Raised when generated code fails validation."""

    def __init__(self, message: str, errors: list[str] | None = None):
        self.errors = errors or []
        super().__init__(message, {"validation_errors": self.errors})


class CodeValidationResult(Protocol):
    """Minimal validation result contract used by generation backends."""

    @property
    def is_valid(self) -> bool:
        """Whether the generated code is valid."""

    @property
    def errors(self) -> list[str]:
        """Validation errors used for corrective retries."""

    @property
    def warnings(self) -> list[str]:
        """Non-fatal validation warnings."""


class CodeValidationCallable(Protocol):
    """Validator callable passed into generation backends."""

    def __call__(self, code: str) -> CodeValidationResult:
        """Validate generated code and return structured issues."""


@dataclass(frozen=True)
class CodeGenerationBackendRequest:
    """Input owned by the business orchestrator and consumed by a backend."""

    prompt: str
    config: GenerationConfig | None = None
    validate_code: CodeValidationCallable | None = None
    max_validation_retries: int = 0


@dataclass(frozen=True)
class CodeGenerationBackendResult:
    """Structured output returned by a code generation backend."""

    generation_result: GenerationResult
    code: str
    summary: str
    required_tickers: list[str] = field(default_factory=list)


class CodeGenerationBackend(Protocol):
    """Protocol implemented by direct and agent-based generation backends."""

    async def generate(
        self,
        request: CodeGenerationBackendRequest,
    ) -> CodeGenerationBackendResult:
        """Generate a code candidate and any structured metadata."""

    def get_model_info(self) -> ModelInfo:
        """Return the model metadata used by this backend."""
