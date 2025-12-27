"""
CSV Download Button Components for Backtest Results.

Provides a button group for downloading backtest results as CSV files:
- Equity Curve
- Drawdown
- Monthly Returns
- Metrics
- Trades

Follows consistent Bootstrap styling with icons and labels.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html


# Download button configurations
DOWNLOAD_BUTTONS = [
    {
        "id": "btn-download-equity",
        "label": "Equity Curve",
        "icon": "fas fa-chart-line",
        "download_id": "download-equity",
    },
    {
        "id": "btn-download-drawdown",
        "label": "Drawdown",
        "icon": "fas fa-chart-area",
        "download_id": "download-drawdown",
    },
    {
        "id": "btn-download-monthly",
        "label": "Monthly Returns",
        "icon": "fas fa-calendar-alt",
        "download_id": "download-monthly",
    },
    {
        "id": "btn-download-metrics",
        "label": "Metrics",
        "icon": "fas fa-chart-bar",
        "download_id": "download-metrics",
    },
    {
        "id": "btn-download-trades",
        "label": "Trades",
        "icon": "fas fa-exchange-alt",
        "download_id": "download-trades",
    },
]


def create_download_button(
    button_id: str,
    label: str,
    icon: str,
    disabled: bool = True,
) -> dbc.Button:
    """
    Create a single download button with icon and label.

    Args:
        button_id: Unique identifier for the button.
        label: Button label text.
        icon: Font Awesome icon class.
        disabled: Whether the button is disabled (default: True).

    Returns:
        Dash Bootstrap Button component.
    """
    return dbc.Button(
        [
            html.I(className=f"{icon} me-1"),
            html.Span(label, className="d-none d-md-inline"),
        ],
        id=button_id,
        color="secondary",
        outline=True,
        size="sm",
        disabled=disabled,
        className="me-1",
    )


def create_download_components() -> html.Div:
    """
    Create the download button group with dcc.Download components.

    Returns:
        html.Div containing button group and Download components.
    """
    # Create buttons
    buttons = [
        create_download_button(
            button_id=config["id"],
            label=config["label"],
            icon=config["icon"],
            disabled=True,
        )
        for config in DOWNLOAD_BUTTONS
    ]

    # Create dcc.Download components
    download_components = [
        dcc.Download(id=config["download_id"])
        for config in DOWNLOAD_BUTTONS
    ]

    return html.Div(
        [
            # Button group with label
            dbc.Row(
                [
                    dbc.Col(
                        html.Span(
                            [
                                html.I(className="fas fa-download me-2"),
                                "Export CSV:",
                            ],
                            className="text-muted small",
                        ),
                        width="auto",
                        className="d-flex align-items-center",
                    ),
                    dbc.Col(
                        dbc.ButtonGroup(
                            buttons,
                            size="sm",
                        ),
                        width="auto",
                    ),
                ],
                className="g-2 align-items-center",
            ),
            # Hidden Download components
            html.Div(download_components, style={"display": "none"}),
        ],
        id="div-download-buttons",
        className="mb-3",
    )


def get_download_button_ids() -> list[str]:
    """
    Get list of all download button IDs.

    Returns:
        List of button ID strings.
    """
    return [config["id"] for config in DOWNLOAD_BUTTONS]


def get_download_component_ids() -> list[str]:
    """
    Get list of all dcc.Download component IDs.

    Returns:
        List of download component ID strings.
    """
    return [config["download_id"] for config in DOWNLOAD_BUTTONS]
