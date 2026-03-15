"""Tests for the AgentLoggingCallbackHandler callback handler."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

from langchain_core.outputs import LLMResult

from backend.services.code_generation.graph.callbacks import (
    AgentLoggingCallbackHandler,
)
from backend.services.code_generation.logging import AgentLoggingContext


def _make_mock_context() -> MagicMock:
    """Create a MagicMock that behaves like AgentLoggingContext."""
    ctx = MagicMock(spec=AgentLoggingContext)
    ctx.emit = MagicMock()
    return ctx


class TestLLMEvents:
    """Verify LLM call event capture."""

    def test_on_llm_start_increments_iteration(self) -> None:
        ctx = _make_mock_context()
        handler = AgentLoggingCallbackHandler(ctx)
        assert handler.current_iteration == 0

        handler.on_llm_start(
            serialized={"kwargs": {"model_name": "gpt-4"}},
            prompts=["hello"],
            run_id=uuid4(),
        )
        assert handler.current_iteration == 1

    def test_on_llm_end_emits_success(self) -> None:
        ctx = _make_mock_context()
        handler = AgentLoggingCallbackHandler(ctx)
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"kwargs": {}}, prompts=["p"], run_id=run_id
        )
        response = LLMResult(
            generations=[],
            llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}},
        )
        handler.on_llm_end(response=response, run_id=run_id)

        # llm_call_start + llm_call_end = 2 calls
        assert ctx.emit.call_count == 2
        end_call = ctx.emit.call_args_list[1]
        assert end_call[0][0] == "llm_call_end"
        assert end_call[1]["success"] is True
        assert end_call[1]["token_usage"]["total_tokens"] == 30

    def test_on_llm_error_emits_failure(self) -> None:
        ctx = _make_mock_context()
        handler = AgentLoggingCallbackHandler(ctx)
        run_id = uuid4()

        handler.on_llm_start(
            serialized={"kwargs": {}}, prompts=["p"], run_id=run_id
        )
        handler.on_llm_error(error=ValueError("boom"), run_id=run_id)

        end_call = ctx.emit.call_args_list[1]
        assert end_call[1]["success"] is False
        assert end_call[1]["error_type"] == "ValueError"


class TestToolEvents:
    """Verify tool call event capture."""

    def test_on_tool_start_end(self) -> None:
        ctx = _make_mock_context()
        handler = AgentLoggingCallbackHandler(ctx)
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "validate_generated_code"},
            input_str='{"code": "print(1)"}',
            run_id=run_id,
        )
        handler.on_tool_end(output='{"is_valid": true}', run_id=run_id)

        assert ctx.emit.call_count == 2
        start_call = ctx.emit.call_args_list[0]
        assert start_call[0][0] == "tool_call_start"
        assert start_call[1]["tool_name"] == "validate_generated_code"

        end_call = ctx.emit.call_args_list[1]
        assert end_call[0][0] == "tool_call_end"
        assert end_call[1]["success"] is True

    def test_on_tool_error(self) -> None:
        ctx = _make_mock_context()
        handler = AgentLoggingCallbackHandler(ctx)
        run_id = uuid4()

        handler.on_tool_start(
            serialized={"name": "test_tool"},
            input_str="input",
            run_id=run_id,
        )
        handler.on_tool_error(error=RuntimeError("fail"), run_id=run_id)

        end_call = ctx.emit.call_args_list[1]
        assert end_call[1]["success"] is False
        assert end_call[1]["error_type"] == "RuntimeError"

    def test_tool_input_is_summarized(self) -> None:
        """Verify that tool input redaction is handled at the callback level."""
        ctx = _make_mock_context()
        handler = AgentLoggingCallbackHandler(ctx)
        run_id = uuid4()

        long_input = "x" * 1000
        handler.on_tool_start(
            serialized={"name": "test"},
            input_str=long_input,
            run_id=run_id,
        )

        start_call = ctx.emit.call_args_list[0]
        tool_input = start_call[1]["tool_input"]
        # Should be shorter than the original since summarize_tool_payload was applied
        assert isinstance(tool_input, dict)
        assert tool_input.get("chars") == 1000
