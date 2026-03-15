"""LangGraph agent graph state definitions."""

from __future__ import annotations

from typing import Any

from langgraph.graph import MessagesState


class CodeGenerationState(MessagesState):
    """Graph state for the code generation agent.

    Extends MessagesState to inherit the messages field and add_messages reducer,
    and defines additional state required for code generation.
    """

    # Agent execution context
    agent_run_id: str
    system_prompt: str

    # Validation loop state
    validation_attempt: int
    max_validation_attempts: int
    last_validation_errors: list[str]

    # Final result
    structured_response: dict[str, Any] | None
    is_complete: bool
    error: str | None
