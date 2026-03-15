"""
Ticker Symbol Extraction Utilities.

Provides robust ticker extraction from various sources:
- Python code (load_data calls, variable assignments, list literals)
- Natural language text
- Parameter dictionaries

Follows SOLID principles with clear separation of concerns.
"""

import logging
import re
from typing import List, Set, Dict, Any

logger = logging.getLogger(__name__)


# Ticker blacklist - common words that look like tickers but aren't
TICKER_BLACKLIST: Set[str] = {
    "I", "A", "AN", "THE", "AND", "OR", "IF", "FOR", "IN", "ON", "TO",
    "OF", "AT", "BY", "UP", "IT", "IS", "AS", "BE", "DO", "GO", "SO",
    "NO", "AM", "PM", "US", "UK", "EU", "USA", "GDP", "CEO", "CFO", "CTO",
    "API", "ETF", "IPO", "ROI", "EPS", "PE", "PB", "YTD", "QTD", "MTD",
    "BUY", "SELL", "HOLD", "LONG", "SHORT", "PUT", "CALL", "ATH", "ATL",
}


class TickerExtractionResult:
    """Result of ticker extraction with metadata."""

    def __init__(
        self,
        tickers: List[str],
        source: str,
        confidence: str = "high",
        metadata: Dict[str, Any] | None = None,
    ):
        """
        Initialize extraction result.

        Args:
            tickers: List of extracted ticker symbols
            source: Source of extraction (e.g., "code", "llm", "params")
            confidence: Confidence level ("high", "medium", "low")
            metadata: Optional additional metadata
        """
        self.tickers = tickers
        self.source = source
        self.confidence = confidence
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"TickerExtractionResult(tickers={self.tickers}, source={self.source}, confidence={self.confidence})"


def extract_tickers_from_code(code: str) -> List[str]:
    """
    Extract ticker symbols from Python code.

    Detects patterns:
    1. load_data(tickers=["QLD", "SPY"])
    2. load_data(["QLD", "SPY"])
    3. strategy_symbol = "QLD"
    4. tickers = ["AAPL", "MSFT"]

    Args:
        code: Python source code string

    Returns:
        List of unique ticker symbols (sorted, deduplicated)

    Examples:
        >>> code = 'data = load_data(["AAPL", "SPY"], start, end)'
        >>> extract_tickers_from_code(code)
        ['AAPL', 'SPY']
    """
    tickers: Set[str] = set()

    # Pattern 1 & 2: load_data function calls
    load_data_patterns = [
        # load_data(tickers=["QLD", "SPY"], ...)
        r'load_data\s*\(\s*tickers\s*=\s*\[([^\]]+)\]',
        # load_data(["QLD"], ...)
        r'load_data\s*\(\s*\[([^\]]+)\]',
    ]

    for pattern in load_data_patterns:
        for match in re.finditer(pattern, code):
            ticker_list_str = match.group(1)
            # Extract string literals: "QLD", 'SPY'
            found = re.findall(r'["\']([A-Z]{2,5})["\']', ticker_list_str)
            tickers.update(t for t in found if t not in TICKER_BLACKLIST)

    # Pattern 3: Variable assignments with ticker-like names
    # strategy_symbol = "QLD"
    # ticker = "AAPL"
    assignment_pattern = r'(\w*(?:ticker|symbol|stock|asset)\w*)\s*=\s*["\']([A-Za-z]{2,5})["\']'
    for match in re.finditer(assignment_pattern, code, re.IGNORECASE):
        _, ticker = match.groups()
        ticker = ticker.upper()  # Normalize to uppercase
        if ticker not in TICKER_BLACKLIST:
            tickers.add(ticker)

    # Pattern 4: List literals with multiple tickers
    # tickers = ["AAPL", "MSFT", "GOOGL"]
    list_pattern = r'\[([^\]]*["\'][A-Z]{2,5}["\'][^\]]*)\]'
    for match in re.finditer(list_pattern, code):
        found = re.findall(r'["\']([A-Z]{2,5})["\']', match.group(1))
        # Filter: length >= 2 and not in blacklist
        tickers.update(t for t in found if len(t) >= 2 and t not in TICKER_BLACKLIST)

    result = sorted(list(tickers))
    logger.debug(f"Extracted {len(result)} tickers from code: {result}")
    return result


def extract_tickers_from_text(text: str, blacklist: Set[str] | None = None) -> List[str]:
    """
    Extract ticker symbols from natural language text.

    Uses regex patterns to find uppercase ticker-like words.

    Args:
        text: Natural language text
        blacklist: Optional custom blacklist (uses default if None)

    Returns:
        List of potential ticker symbols

    Examples:
        >>> extract_tickers_from_text("Buy AAPL and TSLA when price drops")
        ['AAPL', 'TSLA']
    """
    if blacklist is None:
        blacklist = TICKER_BLACKLIST

    # Pattern: 2-5 uppercase letters, not preceded/followed by letters
    ticker_pattern = re.compile(r'(?<![A-Za-z])([A-Z]{2,5})(?![A-Za-z])', re.UNICODE)

    tickers = set()
    for match in ticker_pattern.finditer(text):
        ticker = match.group(1)
        if ticker not in blacklist:
            tickers.add(ticker)

    return sorted(list(tickers))


def merge_ticker_sources(
    code_tickers: List[str],
    llm_tickers: List[str],
    params_tickers: List[str],
    benchmarks: List[str],
) -> Dict[str, Any]:
    """
    Merge tickers from multiple sources with priority and validation.

    Priority order:
    1. Code tickers (highest - actually used in code)
    2. LLM tickers (declared by LLM)
    3. Params tickers (user specified)
    4. Benchmarks (always included)

    Args:
        code_tickers: Tickers extracted from generated code
        llm_tickers: Tickers declared by LLM in structured output
        params_tickers: Tickers specified in params
        benchmarks: Benchmark ticker symbols

    Returns:
        Dict with:
        - 'final': Final merged ticker list
        - 'by_source': Breakdown by source
        - 'warnings': Any inconsistencies detected
    """
    # Convert to sets for operations
    code_set = set(code_tickers)
    llm_set = set(llm_tickers)
    params_set = set(params_tickers)
    benchmark_set = set(benchmarks)

    # Merge all sources
    all_tickers = code_set | llm_set | params_set | benchmark_set
    final_list = sorted(list(all_tickers))

    # Detect inconsistencies
    warnings = []

    # Check if code uses tickers not declared by LLM
    code_only = code_set - llm_set
    if code_only:
        warnings.append(
            f"Code uses tickers not declared by LLM: {sorted(list(code_only))}"
        )

    # Check if LLM declared tickers not used in code
    llm_only = llm_set - code_set
    if llm_only:
        warnings.append(
            f"LLM declared tickers not used in code: {sorted(list(llm_only))}"
        )

    # Log results
    logger.info("📊 Ticker merge result:")
    logger.info(f"  - Code tickers: {code_tickers}")
    logger.info(f"  - LLM tickers: {llm_tickers}")
    logger.info(f"  - Params tickers: {params_tickers}")
    logger.info(f"  - Benchmarks: {benchmarks}")
    logger.info(f"  - Final merged: {final_list}")

    if warnings:
        for warning in warnings:
            logger.warning(f"⚠️ {warning}")

    return {
        "final": final_list,
        "by_source": {
            "code": code_tickers,
            "llm": llm_tickers,
            "params": params_tickers,
            "benchmarks": benchmarks,
        },
        "warnings": warnings,
    }


def validate_required_tickers(
    required_tickers: List[str],
    available_data: Dict[str, Any],
) -> List[str]:
    """
    Validate that all required ticker data is available.

    Args:
        required_tickers: List of ticker symbols required by code
        available_data: Dictionary mapping ticker to data (ticker -> DataFrame)

    Returns:
        List of missing tickers (empty if all available)

    Raises:
        ValueError: If any required ticker is missing

    Examples:
        >>> validate_required_tickers(["AAPL"], {"AAPL": df})
        []
        >>> validate_required_tickers(["AAPL", "SPY"], {"AAPL": df})
        Traceback (most recent call last):
        ...
        ValueError: Required ticker data not available: ['SPY']
    """
    missing = set(required_tickers) - set(available_data.keys())

    if missing:
        missing_list = sorted(list(missing))
        logger.error(f"❌ Required tickers missing: {missing_list}")
        raise ValueError(
            f"Required ticker data not available: {missing_list}. "
            f"Please verify ticker symbols are correct and data provider is accessible."
        )

    logger.info(f"✅ All {len(required_tickers)} required tickers available")
    return []


def format_ticker_report(merge_result: Dict[str, Any]) -> str:
    """
    Format a human-readable ticker merge report.

    Args:
        merge_result: Result from merge_ticker_sources()

    Returns:
        Formatted report string
    """
    lines = [
        "=== Ticker Merge Report ===",
        f"Final ticker list ({len(merge_result['final'])}): {', '.join(merge_result['final'])}",
        "",
        "Breakdown by source:",
    ]

    for source, tickers in merge_result['by_source'].items():
        lines.append(f"  {source:12s}: {', '.join(tickers) if tickers else '(none)'}")

    if merge_result['warnings']:
        lines.append("")
        lines.append("⚠️ Warnings:")
        for warning in merge_result['warnings']:
            lines.append(f"  - {warning}")

    return "\n".join(lines)
