"""
Backtest Code Generator Service.

Converts natural language investment strategies into executable Python backtest code
using LLM providers and validates the generated code for safety and correctness.
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Protocol

from backend.models.backtest import (
    BacktestParams,
    BacktestRequest,
    GeneratedCode,
    ModelInfo as BacktestModelInfo,
)
from backend.providers.data.base import DataProvider, DateRange
from backend.providers.llm.base import (
    GenerationConfig,
    LLMProvider,
    ModelInfo,
)
from backend.services.code_generation.base import (
    CodeGenerationBackend,
    CodeGenerationError,
    CodeGenerationBackendRequest,
    CodeValidationResult,
)
from backend.services.code_generation.base import ValidationError as ValidationError  # noqa: F401 - re-exported via services/__init__.py
from backend.utils.ticker_extraction import (
    extract_tickers_from_code,
    merge_ticker_sources,
    format_ticker_report,
)

logger = logging.getLogger(__name__)


# =============================================================================
class DataAvailabilityError(CodeGenerationError):
    """Raised when required data is not available."""

    def __init__(self, message: str, tickers: list[str] | None = None):
        self.tickers = tickers or []
        super().__init__(message, {"unavailable_tickers": self.tickers})


class PromptBuildError(CodeGenerationError):
    """Raised when prompt building fails."""

    pass


# =============================================================================
# Validator Protocol (Interface for Task 8)
# =============================================================================


class CodeValidator(Protocol):
    """
    Protocol for code validators.

    This interface will be implemented in Task 8.
    Allows dependency injection of different validation strategies.
    """

    def validate(self, code: str) -> CodeValidationResult:
        """
        Validate generated Python code.

        Args:
            code: Python code string to validate

        Returns:
            ValidationResult indicating success/failure and any issues

        Raises:
            ValidationError: If validation fails with critical errors
        """
        ...


@dataclass
class ValidationResult:
    """Result of code validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


class DefaultValidator:
    """
    Default code validator implementation.

    Performs basic syntax and safety checks until Task 8 is complete.
    """

    # Dangerous patterns that should not appear in generated code
    FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
        (r"\bexec\s*\(", "exec() function is not allowed"),
        (r"\beval\s*\(", "eval() function is not allowed"),
        (r"\b__import__\s*\(", "__import__() is not allowed"),
        (r"\bcompile\s*\(", "compile() function is not allowed"),
        (r"\bos\.system\s*\(", "os.system() is not allowed"),
        (r"\bsubprocess\.", "subprocess module is not allowed"),
        (r"\bopen\s*\(.+['\"]w", "File writing is not allowed"),
        (r"\brequests\.", "HTTP requests are not allowed"),
        (r"\burllib\.", "urllib module is not allowed"),
        (r"\bsocket\.", "socket module is not allowed"),
    ]

    # Required elements in the code
    REQUIRED_PATTERNS: list[tuple[str, str]] = [
        (r"class\s+\w+.*Strategy", "Strategy class definition required"),
        (r"def\s+next\s*\(", "next() method required in Strategy"),
    ]

    def validate(self, code: str) -> ValidationResult:
        """
        Validate Python code for syntax and safety.

        Args:
            code: Python code string to validate

        Returns:
            ValidationResult with validation status and any issues
        """
        errors: list[str] = []
        warnings: list[str] = []

        # Check for forbidden patterns
        for pattern, message in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, code):
                errors.append(f"Security violation: {message}")

        # Check for required patterns
        for pattern, message in self.REQUIRED_PATTERNS:
            if not re.search(pattern, code):
                warnings.append(f"Missing element: {message}")

        # Syntax check
        try:
            compile(code, "<generated>", "exec")
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings)


# =============================================================================
# Backtest Code Generator
# =============================================================================


class BacktestCodeGenerator:
    """
    Converts natural language strategies into Python backtest code.

    This service orchestrates:
    1. Ticker extraction from strategy text
    2. Data availability verification
    3. Prompt construction from templates
    4. Agent-backed code generation
    5. Code validation and ticker reconciliation

    Follows SOLID principles with dependency injection for the generation backend
    and data providers.

    Attributes:
        data_provider: Data provider for availability checks
        validator: Code validator for safety checks
        generation_backend: Backend responsible for code/summary generation
        prompt_template_path: Path to the system prompt template

    Example:
        generator = BacktestCodeGenerator(
            data_provider=data_provider,
            generation_backend=backend,
        )
        result = await generator.generate(backtest_request)
        print(result.code)
    """

    # Common stock ticker patterns
    # Unicode-aware pattern that works with non-ASCII text (Korean, Japanese, Chinese, etc.)
    # (?<![A-Za-z]) = not preceded by ASCII letter (negative lookbehind)
    # [A-Z]{2,5} = 2-5 uppercase letters (changed from 1-5 to avoid single letters)
    # (?![A-Za-z]) = not followed by ASCII letter (negative lookahead)
    TICKER_PATTERN = re.compile(
        r"(?<![A-Za-z])([A-Z]{2,5})(?![A-Za-z])",
        re.UNICODE
    )

    # Korean stock code pattern (6 digits)
    KOREAN_TICKER_PATTERN = re.compile(
        r"\b(\d{6})\b"  # 6 digit code
    )

    # Common words that look like tickers but aren't
    TICKER_BLACKLIST: set[str] = {
        "I", "A", "AN", "THE", "AND", "OR", "IF", "FOR", "IN", "ON", "TO",
        "OF", "AT", "BY", "UP", "IT", "IS", "AS", "BE", "DO", "GO", "SO",
        "NO", "AM", "PM", "US", "UK", "EU", "USA", "GDP", "CEO", "CFO", "CTO",
        "API", "ETF", "IPO", "ROI", "EPS", "PE", "PB", "YTD", "QTD", "MTD",
        "BUY", "SELL", "HOLD", "LONG", "SHORT", "PUT", "CALL", "ATH", "ATL",
    }

    MAX_VALIDATION_RETRIES = 2

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        data_provider: DataProvider | None = None,
        validator: CodeValidator | None = None,
        generation_backend: CodeGenerationBackend | None = None,
    ):
        """
        Initialize the BacktestCodeGenerator.

        Args:
            llm_provider: Legacy constructor dependency kept only for compatibility
            data_provider: Data provider for checking data availability
            validator: Optional code validator (defaults to DefaultValidator)
            generation_backend: Backend that performs code generation
        """
        if data_provider is None:
            raise ValueError("data_provider is required")

        self.data_provider = data_provider
        self.validator = validator or DefaultValidator()
        self.generation_backend = generation_backend

    # =========================================================================
    # Ticker Extraction (Task 7.2)
    # =========================================================================

    def _extract_tickers(
        self,
        text: str,
        explicit_tickers: list[str] | None = None,
        benchmarks: list[str] | None = None,
    ) -> list[str]:
        """
        Extract stock ticker symbols from text using regex and heuristics.

        Identifies both US tickers (uppercase letters) and Korean stock codes
        (6-digit numbers) from strategy descriptions and benchmark lists.

        Supports explicit ticker override for non-English strategies or when
        automatic extraction fails.

        Args:
            text: Strategy description text to parse
            explicit_tickers: Optional explicit list of tickers (overrides extraction)
            benchmarks: Optional list of benchmark tickers to include

        Returns:
            List of unique ticker symbols found in the text

        Example:
            >>> generator._extract_tickers("Buy AAPL and TSLA when price drops")
            ['AAPL', 'TSLA']
            >>> generator._extract_tickers("QLD만 보유", explicit_tickers=["QLD"])
            ['QLD']
        """
        # Priority 1: Explicit tickers (if provided, skip extraction)
        if explicit_tickers:
            logger.info(f"Using explicit tickers (bypassing extraction): {explicit_tickers}")
            return sorted([ticker.upper().strip() for ticker in explicit_tickers if ticker.strip()])

        tickers: set[str] = set()

        # Priority 2: Extract US-style tickers (2-5 uppercase letters)
        us_matches = self.TICKER_PATTERN.findall(text)
        for match in us_matches:
            if match not in self.TICKER_BLACKLIST:
                tickers.add(match)
                logger.debug(f"Extracted US ticker from text: {match}")

        # Priority 3: Extract Korean stock codes (6 digits)
        kr_matches = self.KOREAN_TICKER_PATTERN.findall(text)
        if kr_matches:
            tickers.update(kr_matches)
            logger.debug(f"Extracted Korean stock codes: {kr_matches}")

        # Priority 4: Add benchmarks if provided
        if benchmarks:
            for benchmark in benchmarks:
                cleaned = benchmark.upper().strip()
                if cleaned:
                    tickers.add(cleaned)
                    logger.debug(f"Added benchmark ticker: {cleaned}")

        # Warning if no tickers found
        if not tickers:
            logger.warning(
                f"No tickers extracted from strategy: '{text[:100]}...'\n"
                f"Tip: Provide explicit tickers via params.explicit_tickers=['TICKER1', 'TICKER2']"
            )

        extracted_list = sorted(list(tickers))
        if extracted_list:
            logger.info(f"Final extracted tickers: {extracted_list}")

        return extracted_list

    # =========================================================================
    # Data Availability and Prompt Builder (Task 7.3)
    # =========================================================================

    async def _check_data_availability(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
    ) -> dict[str, DateRange]:
        """
        Check data availability for requested tickers.

        Queries the data provider to verify each ticker has data
        within the requested date range.

        Args:
            tickers: List of ticker symbols to check
            start_date: Requested start date
            end_date: Requested end date

        Returns:
            Dictionary mapping ticker to its available DateRange

        Raises:
            DataAvailabilityError: If no tickers have available data
        """
        results: dict[str, DateRange] = {}
        unavailable: list[str] = []

        # Check availability for each ticker concurrently
        async def check_ticker(ticker: str) -> tuple[str, DateRange | None]:
            try:
                date_range = await self.data_provider.get_available_date_range(ticker)
                return ticker, date_range
            except Exception:
                return ticker, None

        tasks = [check_ticker(ticker) for ticker in tickers]
        check_results = await asyncio.gather(*tasks)

        for ticker, date_range in check_results:
            if date_range is not None:
                results[ticker] = date_range
            else:
                unavailable.append(ticker)

        if not results:
            raise DataAvailabilityError(
                f"No data available for any of the requested tickers: {tickers}",
                unavailable,
            )

        return results

    @staticmethod
    def _build_prompt(
        strategy_text: str,
        available_tickers: list[str],
        data_ranges: dict[str, DateRange],
        *,
        params: BacktestParams | None = None,
        benchmarks: list[str] | None = None,
    ) -> str:
        """Build a user-message prompt for the agent.

        Includes the strategy description, ticker breakdown (strategy vs
        benchmark), requested backtest period, data availability window,
        and runtime parameter values so the model uses actual request
        values instead of hardcoded reference defaults.

        Args:
            strategy_text: Natural language strategy description.
            available_tickers: All tickers with confirmed data availability.
            data_ranges: Mapping of ticker to its available DateRange.
            params: Backtest parameters for capital/fees/contribution info.
            benchmarks: Benchmark tickers to separate from strategy tickers.
        """
        data_start = max(
            (dr.start_date for dr in data_ranges.values()),
            default=None,
        )
        data_end = min(
            (dr.end_date for dr in data_ranges.values()),
            default=None,
        )

        # Separate strategy tickers from benchmark tickers
        benchmark_set = {b.upper() for b in (benchmarks or [])}
        strategy_tickers = [t for t in available_tickers if t.upper() not in benchmark_set]
        benchmark_tickers = [t for t in available_tickers if t.upper() in benchmark_set]

        lines = [
            "## Strategy",
            strategy_text,
            "",
            "## Strategy Tickers",
            ", ".join(strategy_tickers) if strategy_tickers else "(none extracted)",
        ]

        if benchmark_tickers:
            lines.extend([
                "",
                "## Benchmark Tickers",
                ", ".join(benchmark_tickers),
            ])

        # Include user-requested backtest period
        if params is not None:
            lines.extend([
                "",
                "## Requested Backtest Period",
                f"{params.start_date.isoformat()} to {params.end_date.isoformat()}",
            ])

        if data_start and data_end:
            lines.extend([
                "",
                "## Data Availability Window",
                f"{data_start.isoformat()} to {data_end.isoformat()}",
            ])

        # Include actual request parameters so the model does not
        # fall back to hardcoded example defaults from the reference.
        if params is not None:
            contribution = params.contribution
            fees = params.fees
            lines.extend([
                "",
                "## Runtime Parameters (use via `params` dict)",
                f"- initial_capital: {params.initial_capital}",
                f"- trading_fee_percent: {fees.trading_fee_percent}",
                f"- slippage_percent: {fees.slippage_percent}",
                f"- contribution_frequency: {contribution.frequency.value}",
                f"- contribution_amount: {contribution.amount}",
                f"- dividend_reinvestment: {params.dividend_reinvestment}",
            ])

        return "\n".join(lines)

    def _require_generation_backend(self) -> CodeGenerationBackend:
        """Return the configured generation backend or raise a domain error."""
        if self.generation_backend is None:
            raise CodeGenerationError(
                "No code generation backend configured for BacktestCodeGenerator"
            )
        return self.generation_backend

    # =========================================================================
    # Main Generate Method (Task 7.4)
    # =========================================================================

    async def generate(self, request: BacktestRequest) -> GeneratedCode:
        """
        Generate backtest code from a natural language strategy.

        This is the main entry point that orchestrates the complete
        code generation workflow:
        1. Extract tickers from strategy text
        2. Check data availability for each ticker
        3. Build the prompt from template
        4. Delegate generation to the configured backend
        5. Validate and reconcile generated tickers
        6. Return the validated result

        Args:
            request: BacktestRequest containing strategy and parameters

        Returns:
            GeneratedCode with the validated Python code and metadata

        Raises:
            CodeGenerationError: If any step in the generation fails
            ValidationError: If the generated code fails validation
            DataAvailabilityError: If required data is not available
        """
        # Step 1: Extract tickers from strategy text and benchmarks
        # Use explicit_tickers if provided (for non-English strategies)
        tickers = self._extract_tickers(
            request.strategy,
            explicit_tickers=request.params.explicit_tickers,
            benchmarks=request.params.benchmarks,
        )

        if not tickers:
            raise CodeGenerationError(
                "No ticker symbols found in strategy. "
                "Please mention specific stock symbols (e.g., AAPL, TSLA, 005930)."
            )

        # Step 2: Check data availability concurrently
        data_ranges = await self._check_data_availability(
            tickers,
            request.params.start_date,
            request.params.end_date,
        )

        available_tickers = list(data_ranges.keys())

        # Step 3: Build the prompt with parameters and ticker breakdown
        prompt = self._build_prompt(
            request.strategy,
            available_tickers,
            data_ranges,
            params=request.params,
            benchmarks=request.params.benchmarks,
        )

        # Step 4: Delegate generation to the configured backend
        # Use the model's max_output_tokens from config, not hardcoded value
        generation_backend = self._require_generation_backend()
        model_info = generation_backend.get_model_info()

        # Request-level LLM settings currently own web search on/off behavior.
        # This lets the frontend's default `false` actually disable a globally
        # enabled provider setting.
        extra_config: dict[str, Any] = {
            "web_search_enabled": request.params.llm_settings.web_search_enabled,
        }

        # Per-request LLM overrides from the client request.
        # When not specified, config=None tells the adapter to use config.yaml defaults.
        llm_settings = request.params.llm_settings
        has_overrides = (
            llm_settings.temperature is not None
            or llm_settings.seed is not None
            or "web_search_enabled" in extra_config
        )

        generation_config: GenerationConfig | None = None
        if has_overrides:
            generation_config = GenerationConfig(
                temperature=llm_settings.temperature,
                max_tokens=model_info.max_output_tokens,
                seed=llm_settings.seed,
                extra=extra_config,
            )

        # Step 5: Generate + extract + validate (with retry on validation failure)
        backend_result = await generation_backend.generate(
            CodeGenerationBackendRequest(
                prompt=prompt,
                config=generation_config,
                validate_code=self.validator.validate,
                max_validation_retries=self.MAX_VALIDATION_RETRIES,
            )
        )
        result = backend_result.generation_result
        code = backend_result.code
        summary = backend_result.summary

        # Step 5a: DUAL-PHASE TICKER VALIDATION
        # Phase 1: Use backend-declared tickers from structured output
        llm_declared_tickers = backend_result.required_tickers

        # Phase 2: Extract tickers from generated code
        code_extracted_tickers = extract_tickers_from_code(code)

        # Phase 3: Merge all ticker sources with validation
        # Only include benchmarks that passed data availability check
        available_set = {t.upper() for t in available_tickers}
        available_benchmarks = [
            b for b in request.params.benchmarks if b.upper() in available_set
        ]
        ticker_merge_result = merge_ticker_sources(
            code_tickers=code_extracted_tickers,
            llm_tickers=llm_declared_tickers,
            params_tickers=available_tickers,
            benchmarks=available_benchmarks,
        )

        # Log detailed merge report
        logger.info("\n" + format_ticker_report(ticker_merge_result))

        # Use merged ticker list as final result
        final_tickers = ticker_merge_result["final"]

        # Step 6: Build and return the result
        backtest_model_info = self._convert_model_info(result.model_info)

        return GeneratedCode(
            code=code,
            strategy_summary=summary,
            model_info=backtest_model_info,
            tickers=final_tickers,  # Use dual-phase validated tickers
        )

    def _convert_model_info(self, llm_model_info: ModelInfo) -> BacktestModelInfo:
        """
        Convert LLM ModelInfo to Backtest ModelInfo.

        Args:
            llm_model_info: Model info from LLM provider

        Returns:
            BacktestModelInfo for the response DTO
        """
        return BacktestModelInfo(
            provider=llm_model_info.provider,
            model_id=llm_model_info.model_id,
            max_tokens=llm_model_info.max_output_tokens,
            supports_system_prompt=llm_model_info.supports_system_prompt,
            cost_per_1k_input=float(llm_model_info.cost_per_1k_input),
            cost_per_1k_output=float(llm_model_info.cost_per_1k_output),
        )
