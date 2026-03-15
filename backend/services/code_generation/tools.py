"""
Internal tools used by the code-generation agent.

Logging is handled by the callback handler — tools contain only business logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from backend.services.code_generation.base import CodeValidationCallable
from backend.utils.ticker_extraction import extract_tickers_from_code


class ValidateGeneratedCodeInput(BaseModel):
    """Arguments for generated code validation."""

    code: str = Field(..., description="Complete Python backtest code candidate.")


class ExtractRequiredTickersInput(BaseModel):
    """Arguments for ticker extraction from candidate code."""

    code: str = Field(..., description="Complete Python backtest code candidate.")


def build_generation_tools(
    *,
    validate_code: CodeValidationCallable | None,
    on_validation_errors: Callable[[list[str]], None] | None = None,
) -> list[Any]:
    """Create tools for the generation agent.

    Logging is handled by AgentLoggingCallbackHandler,
    so tools only contain business logic.

    Args:
        validate_code: Validator callable, or None to skip the validate tool.
        on_validation_errors: Optional callback invoked with the latest
            validation error list whenever ``validate_generated_code`` runs.
            Used by the backend to capture errors for the ``GraphRecursionError``
            → ``ValidationError`` translation.
    """
    tools: list[Any] = []

    if validate_code is not None:

        @tool(
            "validate_generated_code",
            args_schema=ValidateGeneratedCodeInput,
            description=(
                "Validate a complete Python backtest candidate. "
                "Call this before finalizing. If invalid, revise the full code "
                "and validate again."
            ),
        )
        def validate_generated_code(code: str) -> dict[str, Any]:
            result = validate_code(code)
            errors = list(result.errors)
            if on_validation_errors is not None:
                on_validation_errors(errors)
            return {
                "is_valid": result.is_valid,
                "errors": errors,
                "warnings": list(result.warnings),
            }

        tools.append(validate_generated_code)

    @tool(
        "extract_required_tickers_from_code",
        args_schema=ExtractRequiredTickersInput,
        description=(
            "Extract ticker symbols referenced by the current Python code. "
            "Use this to keep required_tickers aligned with the final code."
        ),
    )
    def extract_required_tickers_from_code(code: str) -> dict[str, Any]:
        tickers = extract_tickers_from_code(code)
        return {"tickers": tickers, "count": len(tickers)}

    tools.append(extract_required_tickers_from_code)
    return tools
