"""
Code generation backends and shared abstractions.
"""

from backend.services.code_generation.base import (
    CodeGenerationBackend,
    CodeGenerationBackendRequest,
    CodeGenerationBackendResult,
)

__all__ = [
    "CodeGenerationBackend",
    "CodeGenerationBackendRequest",
    "CodeGenerationBackendResult",
]
