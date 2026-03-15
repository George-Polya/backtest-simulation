"""Tests for the graph builder."""

from __future__ import annotations

from unittest.mock import MagicMock

from langchain.tools import tool

from backend.services.code_generation.graph.builder import build_code_generation_graph


def _make_dummy_tool():
    """Create a real @tool for ToolNode compatibility."""

    @tool("dummy_tool")
    def dummy_tool(x: str) -> str:
        """A dummy tool for testing."""
        return x

    return dummy_tool


class TestBuildCodeGenerationGraph:
    """Verify the build_code_generation_graph factory."""

    def _make_mock_model(self) -> MagicMock:
        model = MagicMock()
        model.bind_tools = MagicMock(return_value=model)
        return model

    def test_returns_compiled_graph_and_limit(self) -> None:
        model = self._make_mock_model()
        compiled, limit = build_code_generation_graph(
            model=model, tools=[_make_dummy_tool()], recursion_limit=10
        )
        assert compiled is not None
        assert limit == 10

    def test_graph_has_expected_nodes(self) -> None:
        model = self._make_mock_model()
        compiled, _ = build_code_generation_graph(
            model=model, tools=[_make_dummy_tool()]
        )
        node_names = set(compiled.get_graph().nodes.keys())
        assert "agent" in node_names
        assert "tools" in node_names
        assert "validate" in node_names

    def test_checkpointer_disabled_by_default(self) -> None:
        model = self._make_mock_model()
        compiled, _ = build_code_generation_graph(
            model=model, tools=[_make_dummy_tool()]
        )
        # checkpointer should be None when MemorySaver is disabled
        assert compiled.checkpointer is None

    def test_checkpointer_enabled(self) -> None:
        model = self._make_mock_model()
        compiled, _ = build_code_generation_graph(
            model=model, tools=[_make_dummy_tool()], enable_checkpointer=True
        )
        assert compiled.checkpointer is not None

    def test_default_recursion_limit(self) -> None:
        model = self._make_mock_model()
        _, limit = build_code_generation_graph(
            model=model, tools=[_make_dummy_tool()]
        )
        assert limit == 25
