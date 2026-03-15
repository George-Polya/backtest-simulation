"""LangGraph agent graph node definitions."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from backend.services.code_generation.graph.state import CodeGenerationState


class AgentStructuredResponse(BaseModel):
    """Final structured response that the agent must return."""

    required_tickers: list[str] = Field(
        default_factory=list,
        description="All tickers used by the generated code, including benchmarks.",
    )
    summary: str = Field(
        ...,
        description="Short summary of how the strategy was interpreted.",
    )
    code: str = Field(
        ...,
        description="Complete executable Python backtest code without markdown fences.",
    )


def create_agent_node(
    model: ChatOpenAI,
    tools: list[Any],
    response_schema: type[BaseModel] | None = None,
):
    """Create the LLM call node.

    Binds tools to the model and invokes the LLM with the messages from state.
    When *response_schema* is provided the model is additionally configured with
    ``response_format`` so the provider enforces the JSON schema when the model
    chooses not to call a tool.  This replaces the old ``response_format``
    parameter that ``create_agent`` accepted.
    """
    model_with_tools = model.bind_tools(tools, strict=True)

    if response_schema is not None:
        schema = response_schema.model_json_schema()
        # Ensure all properties are in 'required' for strict mode compatibility,
        # even when the Pydantic model uses default values.
        if "properties" in schema:
            schema["required"] = list(schema["properties"].keys())
        model_with_tools = model_with_tools.bind(
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "strict": True,
                    "schema": schema,
                },
            },
        )

    async def agent_node(state: CodeGenerationState) -> dict[str, Any]:
        """Invoke the LLM to determine the next action (tool call or response)."""
        messages = state["messages"]
        response = await model_with_tools.ainvoke(messages)
        return {"messages": [response]}

    return agent_node


def create_validation_node(
    response_schema: type[BaseModel],
):
    """Create the structured output validation node.

    Parses the structured response from the last AI message and
    appends retry feedback to messages on failure.
    """

    async def validation_node(state: CodeGenerationState) -> dict[str, Any]:
        """Parse and validate the structured response from the last AI message."""
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if not isinstance(last_message, AIMessage):
            return {
                "error": "Last message is not an AI response",
                "is_complete": False,
            }

        # If tool_calls are present, this is not the final response yet
        if last_message.tool_calls:
            return {"is_complete": False}

        # Attempt to parse structured output.
        # Provider-native structured output (response_format=json_schema) stores the
        # parsed result in additional_kwargs["parsed"], with content often empty.
        # Fall back to parsing content when the parsed key is absent.
        try:
            parsed_kwargs = (last_message.additional_kwargs or {}).get("parsed")

            if parsed_kwargs is not None:
                # Provider already parsed the response — may be a dict or
                # a Pydantic model instance.
                if isinstance(parsed_kwargs, BaseModel):
                    parsed_kwargs = parsed_kwargs.model_dump()
                response = response_schema.model_validate(parsed_kwargs)
            else:
                content = last_message.content
                if isinstance(content, str) and content.strip():
                    parsed = json.loads(content)
                    response = response_schema.model_validate(parsed)
                elif isinstance(content, dict):
                    response = response_schema.model_validate(content)
                else:
                    attempt = state.get("validation_attempt", 0) + 1
                    max_attempts = state.get("max_validation_attempts", 3)
                    error_msg = f"No structured output found (content type: {type(content).__name__})"

                    if attempt >= max_attempts:
                        return {
                            "error": f"{error_msg} after {attempt} attempts",
                            "is_complete": True,
                            "validation_attempt": attempt,
                        }

                    feedback = (
                        "Backend correction feedback for your previous attempt:\n"
                        f"- Problem: {error_msg}\n"
                        "- Regenerate the entire structured response with non-empty "
                        "`summary` and `code` (include `required_tickers`, may be empty).\n"
                        "- Do not return prose, markdown fences, diffs, or partial patches."
                    )
                    return {
                        "messages": [HumanMessage(content=feedback)],
                        "validation_attempt": attempt,
                        "is_complete": False,
                        "error": None,
                    }

            return {
                "structured_response": response.model_dump(),
                "is_complete": True,
                "error": None,
            }

        except (json.JSONDecodeError, PydanticValidationError) as e:
            attempt = state.get("validation_attempt", 0) + 1
            max_attempts = state.get("max_validation_attempts", 3)

            if attempt >= max_attempts:
                return {
                    "error": f"Structured output parsing failed after {attempt} attempts: {e}",
                    "is_complete": True,
                    "validation_attempt": attempt,
                }

            # Append retry feedback message
            feedback = (
                "Backend correction feedback for your previous attempt:\n"
                f"- Failure type: {type(e).__name__}\n"
                f"- Problem: {e}\n"
                "- Regenerate the entire structured response with non-empty "
                "`summary`, `code`, and `required_tickers`.\n"
                "- Do not return prose, markdown fences, diffs, or partial patches."
            )
            return {
                "messages": [HumanMessage(content=feedback)],
                "validation_attempt": attempt,
                "is_complete": False,
                "error": None,
            }

    return validation_node


def should_continue(state: CodeGenerationState) -> Literal["tools", "validate", "__end__"]:
    """Router that determines the next step after the agent node.

    - If tool_calls are present -> route to tools node
    - If no tool_calls -> route to validation node
    - If complete -> end
    """
    if state.get("is_complete"):
        return "__end__"

    messages = state["messages"]
    last_message = messages[-1] if messages else None

    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"

    return "validate"


def should_retry_or_end(state: CodeGenerationState) -> Literal["agent", "__end__"]:
    """Router that determines whether to retry after the validation node.

    - If complete -> end
    - Otherwise -> retry by routing back to agent
    """
    if state.get("is_complete"):
        return "__end__"
    return "agent"
