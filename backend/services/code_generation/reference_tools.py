"""Reference tools that the agent calls on demand."""

from __future__ import annotations

from pathlib import Path

from langchain.tools import tool

_PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@tool("get_api_reference")
def get_api_reference() -> str:
    """Return the backtesting.py API reference.

    Detailed documentation on the Strategy class, Backtest class, load_data function,
    params structure, and available attributes/methods.
    Always call this tool before writing code.
    """
    return (_PROMPTS_DIR / "ref_api.txt").read_text(encoding="utf-8")


@tool("get_code_patterns")
def get_code_patterns() -> str:
    """Return verified code patterns and examples.

    Reference for DCA (dollar-cost averaging), benchmark simulation,
    run_backtest result format, and data integrity validation patterns.
    Call this tool when you need complex strategies, DCA, or benchmark comparisons.
    """
    return (_PROMPTS_DIR / "ref_patterns.txt").read_text(encoding="utf-8")
