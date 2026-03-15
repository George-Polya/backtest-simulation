"""
Tests for the LangGraph-based agent code-generation backend.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError
from openai.types.chat import ChatCompletion

from backend.core.config import (
    LLMConfig,
    LLMModelConfig,
    LLMProvider as LLMProviderEnum,
)
from backend.services.code_generation.base import CodeGenerationBackendRequest
from backend.services.code_generation.base import CodeGenerationError, ValidationError
from backend.services.code_generation.langchain_agent_backend import (
    LangChainAgentCodeGenerationBackend,
)
from backend.services.code_validator import ValidationResult


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create LLM config with agent settings enabled."""
    return LLMConfig(
        provider=LLMProviderEnum.LANGCHAIN,
        model=LLMModelConfig(name="anthropic/claude-3.5-sonnet"),
        temperature=0.0,
        max_tokens=8000,
        agent_max_iterations=4,
        agent_timeout_seconds=30,
    )


@pytest.fixture
def system_prompt_path(tmp_path: Path) -> Path:
    """Create a temporary agent system prompt."""
    path = tmp_path / "backtest_agent_system.txt"
    path.write_text("Use tools before finalizing.")
    return path


def _valid_code() -> str:
    return "class ExampleStrategy(Strategy):\n    def next(self):\n        pass\n"


def _successful_graph_result(
    *,
    required_tickers: list[str] | None = None,
    summary: str = "summary",
    code: str | None = None,
) -> dict[str, object]:
    """Build a minimal successful graph execution result."""
    return {
        "messages": [AIMessage(content="")],
        "structured_response": {
            "required_tickers": required_tickers or ["AAPL"],
            "summary": summary,
            "code": code or _valid_code(),
        },
        "is_complete": True,
        "error": None,
    }


def _patch_graph(side_effect=None, return_value=None):
    """Create a patch for build_code_generation_graph that returns a mock compiled graph."""
    mock_graph = SimpleNamespace()
    if side_effect is not None:
        mock_graph.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        mock_graph.ainvoke = AsyncMock(return_value=return_value)

    def fake_build(*args, **kwargs):
        recursion_limit = kwargs.get("recursion_limit", 25)
        return mock_graph, recursion_limit

    return patch(
        "backend.services.code_generation.langchain_agent_backend.build_code_generation_graph",
        side_effect=fake_build,
    ), mock_graph


@pytest.mark.asyncio
async def test_agent_backend_returns_structured_result(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Backend should convert structured graph output into the backend result contract."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_result = {
        "messages": [
            AIMessage(
                content="",
                response_metadata={
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                    "finish_reason": "stop",
                },
                id="msg-1",
            )
        ],
        "structured_response": {
            "required_tickers": ["AAPL", "SPY"],
            "summary": "summary",
            "code": _valid_code(),
        },
        "is_complete": True,
        "error": None,
    }

    graph_patch, _ = _patch_graph(return_value=graph_result)

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        result = await backend.generate(
            CodeGenerationBackendRequest(
                prompt="Generate code for AAPL.",
                validate_code=lambda code: ValidationResult(is_valid=True),
                max_validation_retries=2,
            )
        )

    assert result.code == _valid_code().strip()
    assert result.summary == "summary"
    assert result.required_tickers == ["AAPL", "SPY"]
    assert result.generation_result.usage["total_tokens"] == 30
    assert result.generation_result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_agent_backend_raises_validation_error_on_invalid_final_code(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Backend should reject structured output that still fails final validation."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_patch, _ = _patch_graph(return_value=_successful_graph_result())

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        with pytest.raises(ValidationError, match="failed final validation"):
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(
                        is_valid=False,
                        errors=["bad code"],
                    ),
                    max_validation_retries=2,
                )
            )


@pytest.mark.asyncio
async def test_agent_backend_maps_agent_failure_to_code_generation_error(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Backend should translate agent runtime failures into CodeGenerationError."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_patch, _ = _patch_graph(side_effect=RuntimeError("boom"))

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        with pytest.raises(CodeGenerationError, match="Agent generation failed"):
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(is_valid=True),
                    max_validation_retries=2,
                )
            )


@pytest.mark.asyncio
async def test_agent_backend_maps_missing_structured_response_to_error(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Missing structured response from graph should surface as CodeGenerationError."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_result = {
        "messages": [AIMessage(content="")],
        "structured_response": None,
        "is_complete": True,
        "error": None,
    }
    graph_patch, _ = _patch_graph(return_value=graph_result)

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        with pytest.raises(CodeGenerationError, match="no structured response"):
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(is_valid=True),
                    max_validation_retries=0,
                )
            )


@pytest.mark.asyncio
async def test_agent_backend_maps_graph_error_to_code_generation_error(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Graph-level error field should surface as CodeGenerationError."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_result = {
        "messages": [AIMessage(content="")],
        "structured_response": {"required_tickers": ["AAPL"], "summary": "s", "code": "c"},
        "is_complete": True,
        "error": "Structured output parsing failed after 3 attempts",
    }
    graph_patch, _ = _patch_graph(return_value=graph_result)

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        with pytest.raises(CodeGenerationError, match="parsing failed"):
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(is_valid=True),
                    max_validation_retries=0,
                )
            )


@pytest.mark.asyncio
async def test_agent_backend_maps_malformed_structured_output_to_code_generation_error(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Missing required structured fields should surface as CodeGenerationError."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_result = {
        "messages": [AIMessage(content="")],
        "structured_response": {"required_tickers": ["AAPL"]},  # missing summary, code
        "is_complete": True,
        "error": None,
    }
    graph_patch, _ = _patch_graph(return_value=graph_result)

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        with pytest.raises(CodeGenerationError, match="malformed structured output"):
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(is_valid=True),
                    max_validation_retries=0,
                )
            )


@pytest.mark.asyncio
async def test_agent_backend_maps_length_limit_error_to_domain_error(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Length-limited structured outputs should surface as an output-limit failure."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_patch, _ = _patch_graph(
        side_effect=openai.LengthFinishReasonError(
            completion=cast(
                ChatCompletion,
                cast(
                    Any,
                    SimpleNamespace(
                        usage=SimpleNamespace(
                            completion_tokens=8000,
                            prompt_tokens=5118,
                            total_tokens=13118,
                            completion_tokens_details=SimpleNamespace(reasoning_tokens=8000),
                        )
                    ),
                ),
            )
        )
    )

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        with pytest.raises(CodeGenerationError, match="output limit") as error_info:
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(is_valid=True),
                    max_validation_retries=0,
                )
            )

    assert error_info.value.details["error_type"] == "LengthFinishReasonError"
    assert error_info.value.details["usage"].completion_tokens == 8000


@pytest.mark.asyncio
async def test_agent_backend_maps_timeout_to_code_generation_error(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Async timeouts should map to CodeGenerationError with timeout details."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_patch, _ = _patch_graph(return_value=_successful_graph_result())

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch, patch(
        "backend.services.code_generation.langchain_agent_backend.asyncio.wait_for",
        new=AsyncMock(side_effect=asyncio.TimeoutError()),
    ):
        with pytest.raises(CodeGenerationError, match="timed out after 30 seconds") as error_info:
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(is_valid=True),
                    max_validation_retries=0,
                )
            )

    assert error_info.value.details["error_type"] == "TimeoutError"


@pytest.mark.asyncio
async def test_agent_backend_maps_recursion_limit_to_code_generation_error(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """GraphRecursionError should map to CodeGenerationError."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_patch, _ = _patch_graph(
        side_effect=GraphRecursionError("Recursion limit reached")
    )

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        with pytest.raises(CodeGenerationError, match="iteration limit") as error_info:
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(is_valid=True),
                    max_validation_retries=0,
                )
            )

    assert error_info.value.details["error_type"] == "GraphRecursionError"


@pytest.mark.asyncio
async def test_agent_backend_supports_generation_without_validator(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Backends should still work when no validator tool is provided."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_patch, _ = _patch_graph(
        return_value=_successful_graph_result(required_tickers=["AAPL", "SPY"])
    )

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        result = await backend.generate(
            CodeGenerationBackendRequest(
                prompt="Generate code for AAPL.",
                validate_code=None,
                max_validation_retries=0,
            )
        )

    assert result.required_tickers == ["AAPL", "SPY"]


@pytest.mark.asyncio
async def test_agent_backend_wraps_system_prompt_loading_failures(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Missing system prompt files should raise a domain error with file details."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    system_prompt_path.unlink()

    graph_patch, _ = _patch_graph(return_value=_successful_graph_result())

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        with pytest.raises(CodeGenerationError, match="Failed to load agent system prompt") as error_info:
            await backend.generate(
                CodeGenerationBackendRequest(
                    prompt="Generate code for AAPL.",
                    validate_code=lambda code: ValidationResult(is_valid=True),
                    max_validation_retries=0,
                )
            )

    assert error_info.value.details["error_type"] == "FileNotFoundError"
    assert error_info.value.details["path"] == str(system_prompt_path)


@pytest.mark.asyncio
async def test_agent_backend_ticker_normalization(
    llm_config: LLMConfig,
    system_prompt_path: Path,
) -> None:
    """Tickers should be normalized to uppercase and deduplicated."""
    backend = LangChainAgentCodeGenerationBackend(
        api_key="test-key",
        llm_config=llm_config,
        system_prompt_path=system_prompt_path,
    )
    graph_patch, _ = _patch_graph(
        return_value=_successful_graph_result(required_tickers=["aapl", "spy", "AAPL"])
    )

    with patch(
        "backend.services.code_generation.langchain_agent_backend.ChatOpenAI",
        return_value=MagicMock(spec=[]),
    ), graph_patch:
        result = await backend.generate(
            CodeGenerationBackendRequest(
                prompt="Generate code for AAPL.",
                validate_code=lambda code: ValidationResult(is_valid=True),
                max_validation_retries=0,
            )
        )

    assert result.required_tickers == ["AAPL", "SPY"]
