"""Tests for CodeGenerationState state definitions."""

from __future__ import annotations

from backend.services.code_generation.graph.state import CodeGenerationState


class TestCodeGenerationState:
    """Verify the CodeGenerationState TypedDict structure."""

    def test_has_messages_field(self) -> None:
        """Should inherit the messages field from MessagesState."""
        annotations = CodeGenerationState.__annotations__
        assert "agent_run_id" in annotations
        assert "system_prompt" in annotations

    def test_has_validation_fields(self) -> None:
        """Fields required for the validation loop should be defined."""
        annotations = CodeGenerationState.__annotations__
        assert "validation_attempt" in annotations
        assert "max_validation_attempts" in annotations
        assert "last_validation_errors" in annotations

    def test_has_result_fields(self) -> None:
        """Final result fields should be defined."""
        annotations = CodeGenerationState.__annotations__
        assert "structured_response" in annotations
        assert "is_complete" in annotations
        assert "error" in annotations
