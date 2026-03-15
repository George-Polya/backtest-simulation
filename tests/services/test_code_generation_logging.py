"""
Tests for agent/tool logging helpers.
"""

import logging

from backend.services.code_generation.logging import (
    AgentLoggingContext,
    format_agent_event,
    summarize_code,
    summarize_text,
)


def test_summarize_text_truncates_preview() -> None:
    """Long text should be summarized without preserving the full body."""
    text = "x" * 300
    summary = summarize_text(text, preview_chars=50)
    assert summary["chars"] == 300
    assert len(summary["preview"]) == 50
    assert summary["truncated"] is True


def test_summarize_code_uses_hash_and_line_count() -> None:
    """Code summaries should expose compact metadata instead of raw code."""
    code = "line1\nline2\nline3"
    summary = summarize_code(code, preview_chars=5)
    assert summary["lines"] == 3
    assert summary["chars"] == len(code)
    assert len(summary["sha256"]) == 64
    assert summary["preview"] == "line1"


def test_agent_logging_context_emits_structured_log(caplog) -> None:
    """Logging context should emit a structured JSON payload."""
    logger = logging.getLogger("test.agent.logging")
    context = AgentLoggingContext(
        logger=logger,
        agent_run_id="run-1",
        provider="langchain",
        model_id="test-model",
    )
    with caplog.at_level(logging.INFO, logger="test.agent.logging"):
        context.emit("agent_run_start", success=True)

    assert "agent_run_start" in caplog.text
    assert "run-1" in caplog.text


def test_agent_logging_context_emits_pretty_log_when_debug_enabled(caplog) -> None:
    """Debug logging should use a concise human-readable format."""
    logger = logging.getLogger("test.agent.logging.pretty")
    context = AgentLoggingContext(
        logger=logger,
        agent_run_id="run-12345678",
        provider="langchain",
        model_id="openai/gpt-5.4",
        debug_logging=True,
    )
    with caplog.at_level(logging.INFO, logger="test.agent.logging.pretty"):
        context.emit(
            "agent_iteration_end",
            iteration=2,
            duration_ms=1450.25,
            tool_call_count=1,
            structured_response_present=True,
            success=True,
        )

    assert "agent[run-1234]" in caplog.text
    assert "iter=2 ok 1450.25ms tools=1 structured=yes" in caplog.text
    assert "agent_event {" not in caplog.text


def test_format_agent_event_compacts_run_end_message() -> None:
    """Human-readable event formatting should stay compact for run summaries."""
    message = format_agent_event(
        {
            "event": "agent_run_end",
            "agent_run_id": "abc123456789",
            "provider": "langchain",
            "model_id": "openai/gpt-5.4",
            "success": True,
            "duration_ms": 4200.5,
            "required_tickers": {
                "count": 2,
                "items": ["AAPL", "SPY"],
                "truncated": False,
            },
            "summary": {
                "chars": 84,
                "preview": "Momentum strategy with a moving-average filter",
                "truncated": False,
            },
            "code": {
                "chars": 3210,
                "lines": 101,
                "sha256": "12345678abcdef00",
                "preview": "class Strategy:",
                "truncated": False,
            },
        }
    )

    assert message.startswith("agent[abc12345] done ok 4200.50ms")
    assert "tickers=2[AAPL, SPY]" in message
    assert 'summary=84c "Momentum strategy with a moving-average filter"' in message
    assert "code=3210c/101l sha=12345678" in message
