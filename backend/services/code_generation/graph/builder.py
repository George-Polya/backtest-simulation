"""LangGraph StateGraph builder."""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from backend.services.code_generation.graph.nodes import (
    AgentStructuredResponse,
    create_agent_node,
    create_validation_node,
    should_continue,
    should_retry_or_end,
)
from backend.services.code_generation.graph.state import CodeGenerationState


def build_code_generation_graph(
    model: ChatOpenAI,
    tools: list[Any],
    *,
    enable_checkpointer: bool = False,
    recursion_limit: int = 25,
) -> tuple[Any, int]:
    """Build the code generation agent StateGraph.

    Graph structure:
        START -> agent -> (tools -> agent) | (validate -> agent | END)

    Args:
        model: ChatOpenAI instance.
        tools: List of tools available to the agent.
        enable_checkpointer: Enable MemorySaver for debugging.
        recursion_limit: Maximum number of graph execution steps.

    Returns:
        Tuple of (compiled_graph, recursion_limit).
    """
    # Define graph
    graph = StateGraph(CodeGenerationState)

    # Add nodes
    agent_node = create_agent_node(model, tools, response_schema=AgentStructuredResponse)
    tool_node = ToolNode(tools)
    validation_node = create_validation_node(AgentStructuredResponse)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("validate", validation_node)

    # Define edges
    graph.set_entry_point("agent")

    # agent -> tools or validate
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "validate": "validate",
            "__end__": END,
        },
    )

    # tools -> agent (receive tool results and return to agent)
    graph.add_edge("tools", "agent")

    # validate -> agent (retry) or END (complete)
    graph.add_conditional_edges(
        "validate",
        should_retry_or_end,
        {
            "agent": "agent",
            "__end__": END,
        },
    )

    # Compile
    checkpointer = MemorySaver() if enable_checkpointer else None

    compiled = graph.compile(
        checkpointer=checkpointer,
    )

    return compiled, recursion_limit
