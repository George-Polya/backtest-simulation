"""
Code Viewer Component.

Provides a component for displaying and inspecting generated backtest code
with syntax highlighting and user code input functionality.
"""

import dash_ace
import dash_bootstrap_components as dbc
from dash import html


def create_code_viewer_content() -> html.Div:
    """
    Create the Code Viewer content.

    Displays generated Python code with syntax highlighting,
    model information, and allows user to edit code.

    Returns:
        Div containing code viewer elements.
    """
    return html.Div(
        [
            # Model info badge (hidden by default)
            html.Div(
                id="div-model-info",
                className="mb-2",
                style={"display": "none"},
            ),
            # Strategy summary
            html.Div(
                id="div-strategy-summary",
                className="mb-3",
                style={"display": "none"},
            ),
            # Code Editor
            html.Div(
                dash_ace.DashAceEditor(
                    id="ace-generated-code",
                    value="# No code generated yet.\n# Enter a strategy and click 'Generate Code'.",
                    mode="python",
                    theme="github",
                    fontSize=14,
                    showPrintMargin=False,
                    showGutter=True,
                    highlightActiveLine=True,
                    wrapEnabled=True,
                    style={
                        "height": "400px",
                        "width": "100%",
                        "borderRadius": "0.375rem",
                        "border": "1px solid #dee2e6",
                    },
                ),
                className="mt-2",
            ),
            # Generation time info
            html.Div(
                id="div-generation-info",
                className="mt-2 small text-muted",
                style={"display": "none"},
            ),
        ]
    )


def create_model_info_badge(provider: str, model_id: str) -> dbc.Badge:
    """
    Create a badge displaying LLM model information.

    Args:
        provider: LLM provider name.
        model_id: Model identifier.

    Returns:
        Dash Bootstrap Badge component.
    """
    # Provider color mapping
    provider_colors = {
        "openrouter": "primary",
        "anthropic": "warning",
        "openai": "success",
    }
    color = provider_colors.get(provider.lower(), "secondary")

    return dbc.Badge(
        [
            html.I(className="fas fa-robot me-1"),
            f"{provider}: {model_id}",
        ],
        color=color,
        className="me-2",
    )


def create_strategy_summary_alert(summary: str) -> dbc.Alert:
    """
    Create an alert displaying the AI-generated strategy summary.

    Args:
        summary: Strategy interpretation summary from the LLM.

    Returns:
        Dash Bootstrap Alert component.
    """
    return dbc.Alert(
        [
            html.Strong(
                [
                    html.I(className="fas fa-brain me-2"),
                    "Strategy Interpretation: ",
                ]
            ),
            html.Span(summary),
        ],
        color="info",
        className="mb-0",
    )


def format_code_for_display(code: str) -> str:
    """
    Format Python code for markdown display with syntax highlighting.

    Args:
        code: Raw Python code string.

    Returns:
        Markdown-formatted code block.
    """
    return f"```python\n{code}\n```"
