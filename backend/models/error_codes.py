"""
Standardized error codes shared between backend and frontend.

Each ErrorCode maps to a specific failure scenario. The frontend uses
these codes (not HTTP status codes or message strings) to determine
error-specific behavior such as automatic retries.
"""

from enum import StrEnum


class ErrorCode(StrEnum):
    """Application-level error codes returned in API error responses."""

    # --- Execution / Docker ---
    DOCKER_IMAGE_NOT_AVAILABLE = "DOCKER_IMAGE_NOT_AVAILABLE"
    DOCKER_CLIENT_FAILED = "DOCKER_CLIENT_FAILED"
    EXECUTION_FAILED = "EXECUTION_FAILED"

    # --- Code generation ---
    CODE_GENERATION_FAILED = "CODE_GENERATION_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    DATA_NOT_AVAILABLE = "DATA_NOT_AVAILABLE"

    # --- Job management ---
    JOB_NOT_FOUND = "JOB_NOT_FOUND"
    JOB_NOT_COMPLETED = "JOB_NOT_COMPLETED"

    # --- Request ---
    INVALID_REQUEST = "INVALID_REQUEST"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"

    # --- Formatting ---
    RESULT_FORMAT_FAILED = "RESULT_FORMAT_FAILED"

    # --- Catch-all ---
    INTERNAL_ERROR = "INTERNAL_ERROR"
