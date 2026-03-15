"""
Logging helpers for agent-based code generation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from uuid import uuid4


def new_agent_run_id() -> str:
    """Create a request-scoped identifier for agent runs."""
    return uuid4().hex


def _truncate_text(value: str, limit: int = 160) -> tuple[str, bool]:
    """Truncate long text previews to a fixed limit."""
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _compact_text(value: str | None, limit: int = 80) -> str:
    """Collapse multiline text into a single short preview."""
    if not value:
        return ""
    collapsed = " ".join(str(value).split())
    preview, truncated = _truncate_text(collapsed, limit)
    if truncated:
        return f"{preview}..."
    return preview


def summarize_text(value: str | None, *, preview_chars: int = 160) -> dict[str, Any]:
    """Summarize free-form text without logging the full payload."""
    if not value:
        return {"chars": 0, "preview": ""}
    preview, truncated = _truncate_text(value, preview_chars)
    return {
        "chars": len(value),
        "preview": preview,
        "truncated": truncated,
    }


def summarize_code(value: str | None, *, preview_chars: int = 120) -> dict[str, Any]:
    """Summarize generated code without logging the full body."""
    if not value:
        return {"chars": 0, "lines": 0, "sha256": "", "preview": ""}
    preview, truncated = _truncate_text(value, preview_chars)
    return {
        "chars": len(value),
        "lines": value.count("\n") + 1,
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "preview": preview,
        "truncated": truncated,
    }


def summarize_errors(errors: list[str] | None, *, limit: int = 5) -> dict[str, Any]:
    """Summarize validation errors to a bounded list."""
    errors = errors or []
    preview = errors[:limit]
    return {
        "count": len(errors),
        "items": preview,
        "truncated": len(errors) > limit,
    }


def summarize_tool_payload(payload: Any) -> dict[str, Any]:
    """Create a bounded summary for tool input/output payloads."""
    if isinstance(payload, str):
        return summarize_text(payload)
    if isinstance(payload, list):
        return {
            "type": "list",
            "count": len(payload),
            "preview": payload[:5],
            "truncated": len(payload) > 5,
        }
    if isinstance(payload, dict):
        summary: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, str):
                summary[key] = summarize_text(value)
            elif isinstance(value, list):
                summary[key] = {
                    "count": len(value),
                    "preview": value[:5],
                    "truncated": len(value) > 5,
                }
            else:
                summary[key] = value
        return summary
    return {"value": str(payload)}


def _format_text_summary(summary: dict[str, Any]) -> str:
    """Render a compact single-line text summary."""
    chars = summary.get("chars", 0)
    preview = _compact_text(summary.get("preview", ""))
    if preview:
        return f"{chars}c \"{preview}\""
    return f"{chars}c"


def _format_code_summary(summary: dict[str, Any]) -> str:
    """Render a compact single-line code summary."""
    chars = summary.get("chars", 0)
    lines = summary.get("lines", 0)
    sha = str(summary.get("sha256", ""))[:8]
    return f"{chars}c/{lines}l sha={sha}" if sha else f"{chars}c/{lines}l"


def _format_collection_summary(summary: dict[str, Any]) -> str:
    """Render a compact collection summary."""
    count = summary.get("count", 0)
    items = summary.get("items")
    if items is None:
        items = summary.get("preview", [])
    if not isinstance(items, list):
        return str(count)
    joined = ", ".join(str(item) for item in items[:3])
    if not joined:
        return str(count)
    if count > len(items) or summary.get("truncated"):
        return f"{count}[{joined}, ...]"
    return f"{count}[{joined}]"


def _format_value(value: Any) -> str:
    """Render summarized values as compact log fragments."""
    if isinstance(value, dict):
        if "sha256" in value:
            return _format_code_summary(value)
        if "chars" in value:
            return _format_text_summary(value)
        if "count" in value and ("items" in value or "preview" in value):
            return _format_collection_summary(value)
        parts: list[str] = []
        for key, item in list(value.items())[:3]:
            parts.append(f"{key}={_format_value(item)}")
        return ", ".join(parts)
    if isinstance(value, list):
        preview = ", ".join(str(item) for item in value[:3])
        if len(value) > 3:
            preview = f"{preview}, ..."
        return f"[{preview}]"
    if isinstance(value, float):
        return f"{value:.2f}"
    return _compact_text(str(value), limit=60)


def _short_agent_run_id(agent_run_id: str) -> str:
    """Reduce noisy UUIDs in terminal logs."""
    return agent_run_id[:8]


def format_agent_event(data: dict[str, Any]) -> str:
    """Format an agent event as a concise human-readable log line."""
    run_id = _short_agent_run_id(str(data.get("agent_run_id", "")))
    event = data.get("event", "agent_event")
    prefix = f"agent[{run_id}]"

    if event == "agent_run_start":
        return (
            f"{prefix} start model={data.get('model_id')} "
            f"max_iter={data.get('max_iterations')} timeout={data.get('timeout_seconds')}s "
            f"prompt={_format_value(data.get('prompt', {}))}"
        )
    if event == "agent_run_end":
        status = "ok" if data.get("success") else "error"
        parts = [
            f"{prefix} done {status}",
            f"{data.get('duration_ms', 0):.2f}ms"
            if isinstance(data.get("duration_ms"), (int, float))
            else f"{data.get('duration_ms')}ms",
        ]
        if data.get("required_tickers") is not None:
            parts.append(f"tickers={_format_value(data['required_tickers'])}")
        if data.get("summary") is not None:
            parts.append(f"summary={_format_value(data['summary'])}")
        if data.get("code") is not None:
            parts.append(f"code={_format_value(data['code'])}")
        if data.get("error_type") is not None:
            parts.append(f"error={data['error_type']}")
        if data.get("error_message") is not None:
            parts.append(f"msg=\"{_compact_text(str(data['error_message']), 100)}\"")
        return " ".join(parts)
    if event == "agent_iteration_start":
        return (
            f"{prefix} iter={data.get('iteration')} start "
            f"messages={data.get('message_count')}"
        )
    if event == "agent_iteration_end":
        status = "ok" if data.get("success") else "error"
        structured = "yes" if data.get("structured_response_present") else "no"
        line = (
            f"{prefix} iter={data.get('iteration')} {status} "
            f"{_format_value(data.get('duration_ms'))}ms "
            f"tools={data.get('tool_call_count')} structured={structured}"
        )
        if data.get("tool_names"):
            line += f" [{', '.join(data['tool_names'])}]"
        if data.get("token_usage"):
            usage = data["token_usage"]
            prompt = usage.get("prompt_tokens", 0)
            completion = usage.get("completion_tokens", 0)
            total = usage.get("total_tokens", 0)
            line += f" tokens={total}(in={prompt},out={completion})"
        if data.get("ai_content_preview"):
            line += f" response={_format_value(data['ai_content_preview'])}"
        if data.get("error_type") is not None:
            line += f" error={data['error_type']}"
        return line
    if event == "tool_call_start":
        return (
            f"{prefix} iter={data.get('iteration')} tool={data.get('tool_name')} start "
            f"input={_format_value(data.get('tool_input'))}"
        )
    if event == "tool_call_end":
        status = "ok" if data.get("success") else "error"
        line = (
            f"{prefix} iter={data.get('iteration')} tool={data.get('tool_name')} {status} "
            f"{_format_value(data.get('duration_ms'))}ms"
        )
        if data.get("tool_output") is not None:
            line += f" output={_format_value(data['tool_output'])}"
        if data.get("error_type") is not None:
            line += f" error={data['error_type']}"
        return line
    if event == "structured_output_retry":
        line = (
            f"{prefix} retry attempt={data.get('attempt')} "
            f"error={data.get('error_type')}"
        )
        if data.get("error_message") is not None:
            line += f" msg=\"{_compact_text(str(data['error_message']), 100)}\""
        return line
    if event == "llm_call_start":
        return (
            f"{prefix} iter={data.get('iteration')} llm_call start "
            f"model={data.get('model')} prompts={data.get('prompt_count')}"
        )
    if event == "llm_call_end":
        status = "ok" if data.get("success") else "error"
        line = (
            f"{prefix} iter={data.get('iteration')} llm_call {status} "
            f"{_format_value(data.get('duration_ms'))}ms"
        )
        if data.get("token_usage"):
            usage = data["token_usage"]
            total = usage.get("total_tokens", 0)
            line += f" tokens={total}"
        if data.get("error_type"):
            line += f" error={data['error_type']}"
        return line
    if event == "graph_node_start":
        return f"{prefix} node={data.get('node_name')} start"
    if event == "graph_node_end":
        status = "ok" if data.get("success") else "error"
        return (
            f"{prefix} node={data.get('node_name')} {status} "
            f"{_format_value(data.get('duration_ms'))}ms"
        )

    details = ", ".join(
        f"{key}={_format_value(value)}"
        for key, value in data.items()
        if key not in {"event", "agent_run_id", "provider", "model_id"}
    )
    return f"{prefix} {event} {details}".rstrip()


@dataclass(frozen=True)
class AgentLoggingContext:
    """Request-scoped context shared across agent logging helpers."""

    logger: logging.Logger
    agent_run_id: str
    provider: str
    model_id: str
    debug_logging: bool = False

    def emit(self, event: str, **payload: Any) -> None:
        """Emit a structured log line."""
        data = {
            "event": event,
            "agent_run_id": self.agent_run_id,
            "provider": self.provider,
            "model_id": self.model_id,
            **payload,
        }
        if self.debug_logging:
            self.logger.info(format_agent_event(data))
            return
        self.logger.info("agent_event %s", json.dumps(data, sort_keys=True, default=str))


def log_agent_run_start(
    context: AgentLoggingContext,
    *,
    prompt: str,
    max_iterations: int,
    timeout_seconds: int,
) -> float:
    """Log the start of an agent run and return its start timestamp."""
    started_at = perf_counter()
    context.emit(
        "agent_run_start",
        prompt=summarize_text(prompt),
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
    )
    return started_at


def log_agent_run_end(
    context: AgentLoggingContext,
    *,
    started_at: float,
    success: bool,
    code: str | None = None,
    summary: str | None = None,
    required_tickers: list[str] | None = None,
    error: Exception | None = None,
) -> None:
    """Log completion or failure of an agent run."""
    payload: dict[str, Any] = {
        "success": success,
        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
    }
    if code is not None:
        payload["code"] = summarize_code(code)
    if summary is not None:
        payload["summary"] = summarize_text(summary)
    if required_tickers is not None:
        payload["required_tickers"] = {
            "count": len(required_tickers),
            "items": required_tickers[:10],
            "truncated": len(required_tickers) > 10,
        }
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    context.emit("agent_run_end", **payload)


def log_agent_iteration_start(
    context: AgentLoggingContext,
    *,
    iteration: int,
    message_count: int,
) -> float:
    """Log the start of a single agent iteration."""
    started_at = perf_counter()
    context.emit(
        "agent_iteration_start",
        iteration=iteration,
        message_count=message_count,
    )
    return started_at


def log_agent_iteration_end(
    context: AgentLoggingContext,
    *,
    iteration: int,
    started_at: float,
    tool_call_count: int,
    structured_response_present: bool,
    error: Exception | None = None,
    tool_names: list[str] | None = None,
    ai_content_preview: str | None = None,
    token_usage: dict[str, int] | None = None,
) -> None:
    """Log the end of a single agent iteration."""
    payload: dict[str, Any] = {
        "iteration": iteration,
        "duration_ms": round((perf_counter() - started_at) * 1000, 2),
        "tool_call_count": tool_call_count,
        "structured_response_present": structured_response_present,
        "success": error is None,
    }
    if tool_names:
        payload["tool_names"] = tool_names
    if ai_content_preview is not None:
        payload["ai_content_preview"] = summarize_text(ai_content_preview)
    if token_usage:
        payload["token_usage"] = token_usage
    if error is not None:
        payload["error_type"] = type(error).__name__
        payload["error_message"] = str(error)
    context.emit("agent_iteration_end", **payload)


def log_tool_start(
    context: AgentLoggingContext,
    *,
    iteration: int,
    tool_name: str,
    tool_input: Any,
) -> float:
    """Log the start of a tool call."""
    started_at = perf_counter()
    context.emit(
        "tool_call_start",
        iteration=iteration,
        tool_name=tool_name,
        tool_input=summarize_tool_payload(tool_input),
    )
    return started_at


def log_tool_end(
    context: AgentLoggingContext,
    *,
    iteration: int,
    tool_name: str,
    started_at: float,
    tool_output: Any,
) -> None:
    """Log successful completion of a tool call."""
    context.emit(
        "tool_call_end",
        iteration=iteration,
        tool_name=tool_name,
        duration_ms=round((perf_counter() - started_at) * 1000, 2),
        success=True,
        tool_output=summarize_tool_payload(tool_output),
    )


def log_tool_error(
    context: AgentLoggingContext,
    *,
    iteration: int,
    tool_name: str,
    started_at: float,
    error: Exception,
) -> None:
    """Log failed completion of a tool call."""
    context.emit(
        "tool_call_end",
        iteration=iteration,
        tool_name=tool_name,
        duration_ms=round((perf_counter() - started_at) * 1000, 2),
        success=False,
        error_type=type(error).__name__,
        error_message=str(error),
    )
