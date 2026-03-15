"""
LangGraph-based agent backend for backtest code generation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import openai
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from pydantic import BaseModel, ValidationError as PydanticValidationError

from backend.core.config import LLMConfig
from backend.providers.llm.base import GenerationResult, LLMProviderError, ModelInfo
from backend.providers.llm.langchain_client import (
    build_chat_openai_kwargs,
    build_model_info,
    resolve_generation_config,
)
from backend.providers.llm.model_limits import resolve_openrouter_limits
from backend.services.code_generation.base import (
    CodeGenerationBackend,
    CodeGenerationError,
    CodeGenerationBackendRequest,
    CodeGenerationBackendResult,
    ValidationError,
)
from backend.services.code_generation.graph.builder import build_code_generation_graph
from backend.services.code_generation.graph.callbacks import (
    AgentLoggingCallbackHandler,
)
from backend.services.code_generation.graph.nodes import AgentStructuredResponse
from backend.services.code_generation.logging import (
    AgentLoggingContext,
    log_agent_run_end,
    log_agent_run_start,
    new_agent_run_id,
)
from backend.services.code_generation.reference_tools import (
    get_api_reference,
    get_code_patterns,
)
from backend.services.code_generation.tools import build_generation_tools
from backend.utils.ticker_extraction import extract_tickers_from_code

logger = logging.getLogger(__name__)


class LangChainAgentCodeGenerationBackend(CodeGenerationBackend):
    """LangGraph StateGraph-based code generation backend."""

    DEFAULT_SYSTEM_PROMPT_PATH = (
        Path(__file__).resolve().parents[2] / "prompts" / "backtest_agent_system.txt"
    )

    def __init__(
        self,
        *,
        api_key: str,
        llm_config: LLMConfig,
        system_prompt_path: Path | str | None = None,
    ) -> None:
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
        self._system_prompt_path = (
            Path(system_prompt_path)
            if system_prompt_path is not None
            else self.DEFAULT_SYSTEM_PROMPT_PATH
        )
        self._system_prompt: str | None = None

    def get_model_info(self) -> ModelInfo:
        """Return model metadata for this backend."""
        return self._model_info

    def _load_system_prompt(self) -> str:
        """Load the agent system prompt from disk once."""
        if self._system_prompt is None:
            try:
                self._system_prompt = self._system_prompt_path.read_text(
                    encoding="utf-8"
                )
            except OSError as error:
                raise CodeGenerationError(
                    f"Failed to load agent system prompt: {self._system_prompt_path}",
                    {
                        "error_type": type(error).__name__,
                        "path": str(self._system_prompt_path),
                    },
                ) from error
        return self._system_prompt

    def _build_model(self, request: CodeGenerationBackendRequest) -> ChatOpenAI:
        """Create a request-scoped ChatOpenAI client for agent execution."""
        effective_config = resolve_generation_config(
            self._llm_config,
            request.config,
            resolved_max_output_tokens=self._resolved_max_output_tokens,
        )
        kwargs = build_chat_openai_kwargs(
            self._api_key,
            self._llm_config,
            effective_config,
            resolved_max_output_tokens=self._resolved_max_output_tokens,
            include_extra_body=True,
        )
        return ChatOpenAI(**kwargs)

    def _calculate_recursion_limit(
        self, request: CodeGenerationBackendRequest
    ) -> int:
        """Convert agent_max_iterations to a LangGraph recursion_limit.

        Each agent -> tools -> agent cycle consumes 3 steps, so iterations * 3 + buffer.
        """
        max_iterations = max(
            request.max_validation_retries + 1,
            self._llm_config.agent_max_iterations,
        )
        return (max_iterations * 3) + 1

    @staticmethod
    def _extract_generation_metadata(
        result: dict[str, Any],
        model_info: ModelInfo,
        structured_response: AgentStructuredResponse,
    ) -> GenerationResult:
        """Convert agent output into the existing GenerationResult shape."""
        messages = result.get("messages", [])
        last_ai_message = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )

        usage: dict[str, int] = {}
        finish_reason = "stop"
        raw_response: dict[str, Any] | None = None
        if last_ai_message is not None:
            response_metadata = dict(last_ai_message.response_metadata or {})
            token_usage = response_metadata.get("token_usage", {})
            if token_usage:
                usage = {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }
            finish_reason = response_metadata.get("finish_reason", "stop")
            raw_response = response_metadata or None
            if last_ai_message.id:
                raw_response = raw_response or {}
                raw_response["id"] = last_ai_message.id

        return GenerationResult(
            content=json.dumps(structured_response.model_dump(), ensure_ascii=False),
            model_info=model_info,
            usage=usage,
            finish_reason=finish_reason,
            raw_response=raw_response,
        )

    @staticmethod
    def _normalize_required_tickers(tickers: list[str]) -> list[str]:
        """Normalize agent-declared tickers to uppercase unique values."""
        normalized: list[str] = []
        seen: set[str] = set()
        for ticker in tickers:
            cleaned = str(ticker).upper().strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                normalized.append(cleaned)
        return normalized

    async def generate(
        self,
        request: CodeGenerationBackendRequest,
    ) -> CodeGenerationBackendResult:
        """Generate code using a LangGraph StateGraph with internal tools."""
        agent_run_id = new_agent_run_id()
        logging_context = AgentLoggingContext(
            logger=logger,
            agent_run_id=agent_run_id,
            provider=self._model_info.provider,
            model_id=self._model_info.model_id,
            debug_logging=self._llm_config.agent_debug_logging,
        )
        timeout_seconds = self._llm_config.agent_timeout_seconds
        recursion_limit = self._calculate_recursion_limit(request)
        run_started_at = log_agent_run_start(
            logging_context,
            prompt=request.prompt,
            max_iterations=self._llm_config.agent_max_iterations,
            timeout_seconds=timeout_seconds,
        )

        # Track validation errors across the graph execution for
        # GraphRecursionError → ValidationError translation.
        last_validation_errors: list[str] = []

        def _capture_validation_errors(errors: list[str]) -> None:
            last_validation_errors[:] = errors

        try:
            # Build model and tools
            model = self._build_model(request)
            validation_tools = build_generation_tools(
                validate_code=request.validate_code,
                on_validation_errors=_capture_validation_errors,
            )
            all_tools = [get_api_reference, get_code_patterns, *validation_tools]

            # Build LangGraph graph
            enable_checkpointer = getattr(
                self._llm_config, "agent_enable_checkpointer", False
            )
            graph, recursion_limit = build_code_generation_graph(
                model=model,
                tools=all_tools,
                enable_checkpointer=enable_checkpointer,
                recursion_limit=recursion_limit,
            )

            # Create callback handler
            callback_handler = AgentLoggingCallbackHandler(logging_context)

            # Compose initial state
            initial_state: dict[str, Any] = {
                "messages": [
                    SystemMessage(content=self._load_system_prompt()),
                    HumanMessage(content=request.prompt),
                ],
                "agent_run_id": agent_run_id,
                "system_prompt": self._load_system_prompt(),
                "validation_attempt": 0,
                "max_validation_attempts": request.max_validation_retries + 1,
                "last_validation_errors": [],
                "structured_response": None,
                "is_complete": False,
                "error": None,
            }

            # Execute graph
            config: dict[str, Any] = {
                "callbacks": [callback_handler],
                "recursion_limit": recursion_limit,
            }
            if enable_checkpointer:
                config["configurable"] = {"thread_id": agent_run_id}

            raw_result = await asyncio.wait_for(
                graph.ainvoke(initial_state, config=config),
                timeout=timeout_seconds,
            )

        except CodeGenerationError as error:
            log_agent_run_end(
                logging_context,
                started_at=run_started_at,
                success=False,
                error=error,
            )
            raise
        except asyncio.TimeoutError as error:
            wrapped = CodeGenerationError(
                f"Agent generation timed out after {timeout_seconds} seconds",
                {"error_type": type(error).__name__},
            )
            log_agent_run_end(
                logging_context,
                started_at=run_started_at,
                success=False,
                error=wrapped,
            )
            raise wrapped from error
        except GraphRecursionError as error:
            wrapped: Exception
            if last_validation_errors:
                wrapped = ValidationError(
                    "Generated code failed validation before the agent hit its iteration limit",
                    list(last_validation_errors),
                )
            else:
                wrapped = CodeGenerationError(
                    "Agent exceeded the configured iteration limit before producing valid output",
                    {"error_type": type(error).__name__},
                )
            log_agent_run_end(
                logging_context,
                started_at=run_started_at,
                success=False,
                error=wrapped,
            )
            raise wrapped from error
        except openai.LengthFinishReasonError as error:
            details: dict[str, Any] = {"error_type": type(error).__name__}
            completion = getattr(error, "completion", None)
            usage = getattr(completion, "usage", None)
            if usage is not None:
                details["usage"] = (
                    usage.model_dump() if hasattr(usage, "model_dump") else usage
                )
            wrapped = CodeGenerationError(
                "Agent generation hit the model output limit",
                details,
            )
            log_agent_run_end(
                logging_context,
                started_at=run_started_at,
                success=False,
                error=wrapped,
            )
            raise wrapped from error
        except LLMProviderError as error:
            wrapped = CodeGenerationError(
                f"Agent generation failed: {error}",
                {"error_type": type(error).__name__},
            )
            log_agent_run_end(
                logging_context,
                started_at=run_started_at,
                success=False,
                error=wrapped,
            )
            raise wrapped from error
        except Exception as error:
            wrapped = CodeGenerationError(
                f"Agent generation failed: {error}",
                {"error_type": type(error).__name__},
            )
            log_agent_run_end(
                logging_context,
                started_at=run_started_at,
                success=False,
                error=wrapped,
            )
            raise wrapped from error

        # Extract result
        structured = raw_result.get("structured_response")
        graph_error = raw_result.get("error")
        if structured is None or graph_error:
            wrapped = CodeGenerationError(
                graph_error or "Agent returned no structured response",
                {"error_type": "MissingStructuredResponse"},
            )
            log_agent_run_end(
                logging_context,
                started_at=run_started_at,
                success=False,
                error=wrapped,
            )
            raise wrapped

        try:
            parsed_response = AgentStructuredResponse.model_validate(structured)
        except PydanticValidationError as error:
            wrapped = CodeGenerationError(
                "Agent returned malformed structured output",
                {"error_type": type(error).__name__},
            )
            log_agent_run_end(
                logging_context,
                started_at=run_started_at,
                success=False,
                error=wrapped,
            )
            raise wrapped from error

        # Post-processing
        code = parsed_response.code.strip()
        summary = parsed_response.summary.strip()
        required_tickers = (
            self._normalize_required_tickers(list(parsed_response.required_tickers))
            if parsed_response.required_tickers
            else self._normalize_required_tickers(extract_tickers_from_code(code))
        )

        # Final validation
        if request.validate_code is not None:
            validation_result = request.validate_code(code)
            if not validation_result.is_valid:
                wrapped = ValidationError(
                    "Agent returned code that still failed final validation",
                    list(validation_result.errors),
                )
                log_agent_run_end(
                    logging_context,
                    started_at=run_started_at,
                    success=False,
                    code=code,
                    summary=summary,
                    required_tickers=required_tickers,
                    error=wrapped,
                )
                raise wrapped

        generation_result = self._extract_generation_metadata(
            raw_result,
            self._model_info,
            parsed_response,
        )
        log_agent_run_end(
            logging_context,
            started_at=run_started_at,
            success=True,
            code=code,
            summary=summary,
            required_tickers=required_tickers,
        )
        return CodeGenerationBackendResult(
            generation_result=generation_result,
            code=code,
            summary=summary,
            required_tickers=required_tickers,
        )
