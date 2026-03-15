"""Tests for graph node functions."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from backend.services.code_generation.graph.nodes import (
    AgentStructuredResponse,
    create_validation_node,
    should_continue,
    should_retry_or_end,
)


class TestAgentStructuredResponse:
    """Verify the AgentStructuredResponse Pydantic model."""

    def test_valid_response(self) -> None:
        resp = AgentStructuredResponse(
            required_tickers=["AAPL", "SPY"],
            summary="Test strategy",
            code="print('hello')",
        )
        assert resp.required_tickers == ["AAPL", "SPY"]
        assert resp.summary == "Test strategy"

    def test_missing_tickers_defaults_to_empty(self) -> None:
        resp = AgentStructuredResponse(summary="test", code="code")
        assert resp.required_tickers == []

    def test_missing_code_raises(self) -> None:
        with pytest.raises(Exception):
            AgentStructuredResponse(summary="test")


class TestShouldContinue:
    """Tests for the should_continue router."""

    def test_routes_to_tools_when_tool_calls(self) -> None:
        ai_msg = AIMessage(content="", tool_calls=[{"name": "test", "args": {}, "id": "1"}])
        state = {"messages": [ai_msg], "is_complete": False}
        assert should_continue(state) == "tools"

    def test_routes_to_validate_when_no_tool_calls(self) -> None:
        ai_msg = AIMessage(content="some response")
        state = {"messages": [ai_msg], "is_complete": False}
        assert should_continue(state) == "validate"

    def test_routes_to_end_when_complete(self) -> None:
        state = {"messages": [], "is_complete": True}
        assert should_continue(state) == "__end__"

    def test_routes_to_validate_on_empty_messages(self) -> None:
        state = {"messages": [], "is_complete": False}
        assert should_continue(state) == "validate"


class TestShouldRetryOrEnd:
    """Tests for the should_retry_or_end router."""

    def test_routes_to_end_when_complete(self) -> None:
        state = {"is_complete": True}
        assert should_retry_or_end(state) == "__end__"

    def test_routes_to_agent_when_not_complete(self) -> None:
        state = {"is_complete": False}
        assert should_retry_or_end(state) == "agent"


class TestValidationNode:
    """Tests for create_validation_node."""

    @pytest.mark.asyncio
    async def test_parses_valid_json(self) -> None:
        import json

        node = create_validation_node(AgentStructuredResponse)
        content = json.dumps({
            "required_tickers": ["AAPL"],
            "summary": "test",
            "code": "print(1)",
        })
        ai_msg = AIMessage(content=content)
        state = {"messages": [ai_msg], "validation_attempt": 0, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is True
        assert result["structured_response"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_retries_on_invalid_json(self) -> None:
        node = create_validation_node(AgentStructuredResponse)
        ai_msg = AIMessage(content="not valid json")
        state = {"messages": [ai_msg], "validation_attempt": 0, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is False
        assert result["validation_attempt"] == 1
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_gives_up_after_max_attempts(self) -> None:
        node = create_validation_node(AgentStructuredResponse)
        ai_msg = AIMessage(content="bad json")
        state = {"messages": [ai_msg], "validation_attempt": 2, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is True
        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_skips_when_tool_calls_present(self) -> None:
        node = create_validation_node(AgentStructuredResponse)
        ai_msg = AIMessage(
            content="",
            tool_calls=[{"name": "validate", "args": {}, "id": "1"}],
        )
        state = {"messages": [ai_msg], "validation_attempt": 0, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is False

    @pytest.mark.asyncio
    async def test_parses_from_additional_kwargs_parsed(self) -> None:
        """Provider-native structured output stored in additional_kwargs['parsed']."""
        node = create_validation_node(AgentStructuredResponse)
        ai_msg = AIMessage(
            content="",
            additional_kwargs={
                "parsed": {
                    "required_tickers": ["SPY"],
                    "summary": "parsed summary",
                    "code": "print('parsed')",
                },
            },
        )
        state = {"messages": [ai_msg], "validation_attempt": 0, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is True
        assert result["structured_response"]["summary"] == "parsed summary"
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_parsed_kwargs_takes_precedence_over_content(self) -> None:
        """When both parsed kwargs and content exist, parsed kwargs wins."""
        import json

        node = create_validation_node(AgentStructuredResponse)
        # Content has different data than parsed — parsed should take precedence
        ai_msg = AIMessage(
            content=json.dumps({
                "required_tickers": ["WRONG"],
                "summary": "wrong",
                "code": "wrong",
            }),
            additional_kwargs={
                "parsed": {
                    "required_tickers": ["AAPL"],
                    "summary": "correct",
                    "code": "correct",
                },
            },
        )
        state = {"messages": [ai_msg], "validation_attempt": 0, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is True
        assert result["structured_response"]["summary"] == "correct"

    @pytest.mark.asyncio
    async def test_retries_on_empty_content_without_parsed(self) -> None:
        """Empty content with no parsed kwargs should trigger retry, not crash."""
        node = create_validation_node(AgentStructuredResponse)
        ai_msg = AIMessage(content="")
        state = {"messages": [ai_msg], "validation_attempt": 0, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is False
        assert result["validation_attempt"] == 1
        # Should include retry feedback message
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_empty_content_gives_up_after_max_attempts(self) -> None:
        """Empty content should stop retrying when max_validation_attempts is reached."""
        node = create_validation_node(AgentStructuredResponse)
        ai_msg = AIMessage(content="")
        state = {"messages": [ai_msg], "validation_attempt": 2, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is True
        assert result.get("error") is not None
        assert "3 attempts" in result["error"]

    @pytest.mark.asyncio
    async def test_whitespace_content_gives_up_after_max_attempts(self) -> None:
        """Whitespace-only content should also respect max_validation_attempts."""
        node = create_validation_node(AgentStructuredResponse)
        ai_msg = AIMessage(content="   ")
        state = {"messages": [ai_msg], "validation_attempt": 2, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is True
        assert result.get("error") is not None

    @pytest.mark.asyncio
    async def test_handles_non_ai_message(self) -> None:
        node = create_validation_node(AgentStructuredResponse)
        human_msg = HumanMessage(content="hello")
        state = {"messages": [human_msg], "validation_attempt": 0, "max_validation_attempts": 3}
        result = await node(state)
        assert result["is_complete"] is False
        assert "error" in result
