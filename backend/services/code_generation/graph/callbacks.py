"""LangGraph agent callback handler for agent and tool call logging."""

from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from backend.services.code_generation.logging import (
    AgentLoggingContext,
    summarize_tool_payload,
)


class AgentLoggingCallbackHandler(BaseCallbackHandler):
    """Callback handler that converts LangChain/LangGraph events into structured logs.

    Records LLM calls, tool calls, and error events occurring during agent execution
    in a consistent format via AgentLoggingContext.
    """

    def __init__(self, logging_context: AgentLoggingContext) -> None:
        super().__init__()
        self._ctx = logging_context
        self._llm_start_times: dict[UUID, float] = {}
        self._tool_start_times: dict[UUID, float] = {}
        self._iteration = 0

    # -- LLM events -----------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._iteration += 1
        self._llm_start_times[run_id] = perf_counter()
        self._ctx.emit(
            "llm_call_start",
            iteration=self._iteration,
            model=serialized.get("kwargs", {}).get("model_name", "unknown"),
            prompt_count=len(prompts),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        started_at = self._llm_start_times.pop(run_id, perf_counter())
        duration_ms = round((perf_counter() - started_at) * 1000, 2)

        token_usage: dict[str, int] = {}
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

        self._ctx.emit(
            "llm_call_end",
            iteration=self._iteration,
            duration_ms=duration_ms,
            success=True,
            token_usage=token_usage or None,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        started_at = self._llm_start_times.pop(run_id, perf_counter())
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        self._ctx.emit(
            "llm_call_end",
            iteration=self._iteration,
            duration_ms=duration_ms,
            success=False,
            error_type=type(error).__name__,
            error_message=str(error)[:200],
        )

    # -- Tool events ----------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_start_times[run_id] = perf_counter()
        tool_name = serialized.get("name", "unknown")
        self._ctx.emit(
            "tool_call_start",
            iteration=self._iteration,
            tool_name=tool_name,
            tool_input=summarize_tool_payload(input_str),
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        started_at = self._tool_start_times.pop(run_id, perf_counter())
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        self._ctx.emit(
            "tool_call_end",
            iteration=self._iteration,
            duration_ms=duration_ms,
            success=True,
            tool_output=summarize_tool_payload(output),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        started_at = self._tool_start_times.pop(run_id, perf_counter())
        duration_ms = round((perf_counter() - started_at) * 1000, 2)
        self._ctx.emit(
            "tool_call_end",
            iteration=self._iteration,
            duration_ms=duration_ms,
            success=False,
            error_type=type(error).__name__,
            error_message=str(error)[:200],
        )

    @property
    def current_iteration(self) -> int:
        return self._iteration
