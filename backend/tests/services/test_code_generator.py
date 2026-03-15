"""
Tests for BacktestCodeGenerator service.

Covers:
- Ticker extraction from text
- Data availability checking
- Prompt building
- Full generation workflow
- Validation integration
"""

import pytest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from backend.models.backtest import (
    BacktestParams,
    BacktestRequest,
    ContributionFrequency,
    ContributionPlan,
    FeeSettings,
    GeneratedCode,
)
from backend.providers.data.base import DateRange, DataProvider
from backend.providers.llm.base import (
    GenerationResult,
    LLMProvider,
    ModelInfo,
)
from backend.services.code_generation.base import CodeGenerationBackendResult
from backend.services.code_generator import (
    BacktestCodeGenerator,
    CodeGenerationError,
    DataAvailabilityError,
    DefaultValidator,
    ValidationError,
    ValidationResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """Create a mock LLM provider."""
    mock = MagicMock(spec=LLMProvider)
    mock.provider_name = "test_provider"
    mock.get_model_info.return_value = ModelInfo(
        model_id="test/model-v1",
        provider="test_provider",
        display_name="Test Model",
        max_context_tokens=128000,
        max_output_tokens=8000,
        cost_per_1k_input=Decimal("0.01"),
        cost_per_1k_output=Decimal("0.03"),
    )
    return mock


@pytest.fixture
def mock_data_provider() -> MagicMock:
    """Create a mock data provider."""
    mock = MagicMock(spec=DataProvider)
    mock.provider_name = "test_data_provider"
    return mock


@pytest.fixture
def sample_backtest_params() -> BacktestParams:
    """Create sample backtest parameters."""
    return BacktestParams(
        start_date=date(2020, 1, 1),
        end_date=date(2023, 12, 31),
        initial_capital=100000.0,
        contribution=ContributionPlan(
            frequency=ContributionFrequency.MONTHLY,
            amount=1000.0,
        ),
        fees=FeeSettings(
            trading_fee_percent=0.1,
            slippage_percent=0.05,
        ),
        dividend_reinvestment=True,
        benchmarks=["SPY"],
    )


@pytest.fixture
def sample_backtest_request(sample_backtest_params: BacktestParams) -> BacktestRequest:
    """Create sample backtest request."""
    return BacktestRequest(
        strategy="Buy AAPL and TSLA when their RSI drops below 30, sell when RSI exceeds 70.",
        params=sample_backtest_params,
    )


@pytest.fixture
def sample_generated_code() -> str:
    """Create sample generated code matching the sample response."""
    return """from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import ta

class RSIMeanReversionStrategy(Strategy):
    \"\"\"RSI-based mean reversion strategy for AAPL and TSLA.\"\"\"

    rsi_low = 30
    rsi_high = 70
    rsi_period = 14

    def init(self):
        close = self.data.Close
        self.rsi = self.I(
            lambda x: ta.momentum.RSIIndicator(pd.Series(x), window=self.rsi_period).rsi(),
            close
        )

    def next(self):
        if self.rsi[-1] < self.rsi_low and not self.position:
            self.buy()
        elif self.rsi[-1] > self.rsi_high and self.position:
            self.position.close()


def load_data(tickers: list[str], start_date: str, end_date: str) -> dict[str, pd.DataFrame]:
    pass


# Backtest execution
bt = Backtest(
    data,
    RSIMeanReversionStrategy,
    cash=100000,
    commission=0.001,
    exclusive_orders=True
)
results = bt.run()"""


@pytest.fixture
def mock_generation_backend(mock_llm_provider: MagicMock) -> MagicMock:
    """Create a mock code-generation backend."""
    mock = MagicMock()
    mock.get_model_info.return_value = mock_llm_provider.get_model_info.return_value
    mock.generate = AsyncMock()
    return mock


def _make_backend_result(
    *,
    model_info: ModelInfo,
    code: str,
    summary: str,
    required_tickers: list[str],
    content: str | None = None,
) -> CodeGenerationBackendResult:
    """Build a backend result for BacktestCodeGenerator tests."""
    payload = {
        "required_tickers": required_tickers,
        "summary": summary,
        "code": code,
    }
    return CodeGenerationBackendResult(
        generation_result=GenerationResult(
            content=content or str(payload),
            model_info=model_info,
        ),
        code=code,
        summary=summary,
        required_tickers=required_tickers,
    )


@pytest.fixture
def code_generator(
    mock_llm_provider: MagicMock,
    mock_data_provider: MagicMock,
    mock_generation_backend: MagicMock,
    tmp_path: Path,
) -> BacktestCodeGenerator:
    """Create a BacktestCodeGenerator instance with mocks."""
    # Create a temporary prompt template
    prompt_path = tmp_path / "backtest_system.txt"
    prompt_path.write_text("""Test prompt template.
Strategy: {strategy_description}
Start: {start_date}
End: {end_date}
Capital: {initial_capital}
Benchmarks: {benchmarks}
Contribution: {contribution_frequency} - {contribution_amount}
Trading Fee: {trading_fee_percent}%
Trading Fee Decimal: {trading_fee_decimal}
Slippage: {slippage_percent}%
Dividend Reinvestment: {dividend_reinvestment}
Available Tickers: {available_tickers}
Data Range: {data_start_date} to {data_end_date}
""")

    return BacktestCodeGenerator(
        llm_provider=mock_llm_provider,
        data_provider=mock_data_provider,
        generation_backend=mock_generation_backend,
    )


# =============================================================================
# Tests for _extract_tickers (Task 7.2)
# =============================================================================


class TestExtractTickers:
    """Tests for ticker extraction from text."""

    def test_extract_single_ticker(self, code_generator: BacktestCodeGenerator) -> None:
        """Test extracting a single ticker."""
        text = "I want to buy AAPL stock."
        tickers = code_generator._extract_tickers(text)
        assert "AAPL" in tickers

    def test_extract_multiple_tickers(self, code_generator: BacktestCodeGenerator) -> None:
        """Test extracting multiple tickers."""
        text = "Buy AAPL and TSLA when the market dips, also consider GOOGL."
        tickers = code_generator._extract_tickers(text)
        assert "AAPL" in tickers
        assert "TSLA" in tickers
        assert "GOOGL" in tickers

    def test_ignore_common_words(self, code_generator: BacktestCodeGenerator) -> None:
        """Test that common words are not extracted as tickers."""
        text = "I want to BUY and SELL stocks in the US market."
        tickers = code_generator._extract_tickers(text)
        assert "BUY" not in tickers
        assert "SELL" not in tickers
        assert "US" not in tickers

    def test_extract_korean_ticker(self, code_generator: BacktestCodeGenerator) -> None:
        """Test extracting Korean stock codes (6 digits)."""
        text = "Samsung Electronics 005930 and SK Hynix 000660."
        tickers = code_generator._extract_tickers(text)
        assert "005930" in tickers
        assert "000660" in tickers

    def test_extract_with_benchmarks(self, code_generator: BacktestCodeGenerator) -> None:
        """Test extracting tickers with benchmarks provided."""
        text = "Buy tech stocks."
        benchmarks = ["SPY", "QQQ"]
        tickers = code_generator._extract_tickers(text, benchmarks)
        assert "SPY" in tickers
        assert "QQQ" in tickers

    def test_deduplicate_tickers(self, code_generator: BacktestCodeGenerator) -> None:
        """Test that duplicate tickers are removed."""
        text = "Buy AAPL, then buy more AAPL when it dips."
        tickers = code_generator._extract_tickers(text)
        assert tickers.count("AAPL") == 1

    def test_sort_tickers(self, code_generator: BacktestCodeGenerator) -> None:
        """Test that tickers are sorted alphabetically."""
        text = "Consider TSLA, AAPL, and META."
        tickers = code_generator._extract_tickers(text)
        assert tickers == sorted(tickers)

    def test_single_letter_not_extracted(self, code_generator: BacktestCodeGenerator) -> None:
        """Test that single letters are not extracted as tickers."""
        text = "Buy stock A in market B."
        tickers = code_generator._extract_tickers(text)
        assert "A" not in tickers
        assert "B" not in tickers


# =============================================================================
# Tests for DefaultValidator
# =============================================================================


class TestDefaultValidator:
    """Tests for the default code validator."""

    @pytest.fixture
    def validator(self) -> DefaultValidator:
        """Create a validator instance."""
        return DefaultValidator()

    def test_valid_code(self, validator: DefaultValidator) -> None:
        """Test that valid code passes validation."""
        code = """
from backtesting import Strategy

class MyStrategy(Strategy):
    def init(self):
        pass

    def next(self):
        pass
"""
        result = validator.validate(code)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_forbidden_exec(self, validator: DefaultValidator) -> None:
        """Test that exec() is forbidden."""
        code = "exec('print(1)')"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("exec()" in e for e in result.errors)

    def test_forbidden_eval(self, validator: DefaultValidator) -> None:
        """Test that eval() is forbidden."""
        code = "result = eval('1+1')"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("eval()" in e for e in result.errors)

    def test_forbidden_os_system(self, validator: DefaultValidator) -> None:
        """Test that os.system() is forbidden."""
        code = "import os\nos.system('rm -rf /')"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("os.system()" in e for e in result.errors)

    def test_forbidden_subprocess(self, validator: DefaultValidator) -> None:
        """Test that subprocess module is forbidden."""
        code = "import subprocess\nsubprocess.run(['ls'])"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("subprocess" in e for e in result.errors)

    def test_syntax_error(self, validator: DefaultValidator) -> None:
        """Test that syntax errors are caught."""
        code = "def broken(:\n    pass"
        result = validator.validate(code)
        assert not result.is_valid
        assert any("Syntax error" in e for e in result.errors)

    def test_missing_strategy_warning(self, validator: DefaultValidator) -> None:
        """Test warning for missing Strategy class."""
        code = """
def calculate():
    return 42
"""
        result = validator.validate(code)
        assert any("Strategy class" in w for w in result.warnings)


# =============================================================================
# Tests for _check_data_availability (Task 7.3)
# =============================================================================


class TestCheckDataAvailability:
    """Tests for data availability checking."""

    @pytest.mark.asyncio
    async def test_check_availability_success(
        self,
        code_generator: BacktestCodeGenerator,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test successful data availability check."""
        mock_data_provider.get_available_date_range = AsyncMock(
            return_value=DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        )

        result = await code_generator._check_data_availability(
            ["AAPL"],
            date(2020, 1, 1),
            date(2023, 12, 31),
        )

        assert "AAPL" in result
        assert result["AAPL"].start_date == date(2015, 1, 1)

    @pytest.mark.asyncio
    async def test_check_availability_partial(
        self,
        code_generator: BacktestCodeGenerator,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test partial data availability (some tickers unavailable)."""

        async def mock_get_range(ticker: str, exchange=None) -> DateRange:
            if ticker == "AAPL":
                return DateRange(
                    start_date=date(2015, 1, 1),
                    end_date=date(2024, 1, 1),
                    ticker=ticker,
                )
            raise Exception(f"Ticker {ticker} not found")

        mock_data_provider.get_available_date_range = AsyncMock(
            side_effect=mock_get_range
        )

        result = await code_generator._check_data_availability(
            ["AAPL", "INVALID"],
            date(2020, 1, 1),
            date(2023, 12, 31),
        )

        assert "AAPL" in result
        assert "INVALID" not in result

    @pytest.mark.asyncio
    async def test_check_availability_none_available(
        self,
        code_generator: BacktestCodeGenerator,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test error when no tickers have available data."""
        mock_data_provider.get_available_date_range = AsyncMock(
            side_effect=Exception("Not found")
        )

        with pytest.raises(DataAvailabilityError, match="No data available"):
            await code_generator._check_data_availability(
                ["INVALID1", "INVALID2"],
                date(2020, 1, 1),
                date(2023, 12, 31),
            )


# =============================================================================
# Tests for _build_prompt (Task 7.3)
# =============================================================================


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_build_prompt_contains_strategy(
        self,
        code_generator: BacktestCodeGenerator,
        sample_backtest_params: BacktestParams,
    ) -> None:
        """Test that prompt contains strategy description."""
        data_ranges = {
            "AAPL": DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        }

        prompt = code_generator._build_prompt(
            "Buy AAPL when price drops",
            ["AAPL"],
            data_ranges,
        )

        assert "Buy AAPL when price drops" in prompt

    def test_build_prompt_contains_data_date_range(
        self,
        code_generator: BacktestCodeGenerator,
        sample_backtest_params: BacktestParams,
    ) -> None:
        """Test that prompt contains data date range from providers."""
        data_ranges = {
            "AAPL": DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        }

        prompt = code_generator._build_prompt(
            "Test strategy",
            ["AAPL"],
            data_ranges,
        )

        assert "2015-01-01" in prompt
        assert "2024-01-01" in prompt

    def test_build_prompt_contains_tickers(
        self,
        code_generator: BacktestCodeGenerator,
        sample_backtest_params: BacktestParams,
    ) -> None:
        """Test that prompt contains available tickers."""
        data_ranges = {
            "AAPL": DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        }

        prompt = code_generator._build_prompt(
            "Test strategy",
            ["AAPL"],
            data_ranges,
        )

        assert "AAPL" in prompt

    def test_build_prompt_separates_strategy_and_benchmark_tickers(
        self,
        code_generator: BacktestCodeGenerator,
        sample_backtest_params: BacktestParams,
    ) -> None:
        """Test that benchmark tickers are separated from strategy tickers."""
        data_ranges = {
            "AAPL": DateRange(start_date=date(2015, 1, 1), end_date=date(2024, 1, 1), ticker="AAPL"),
            "SPY": DateRange(start_date=date(2015, 1, 1), end_date=date(2024, 1, 1), ticker="SPY"),
        }

        prompt = code_generator._build_prompt(
            "Buy AAPL",
            ["AAPL", "SPY"],
            data_ranges,
            benchmarks=["SPY"],
        )

        assert "## Strategy Tickers" in prompt
        assert "## Benchmark Tickers" in prompt
        # SPY should be under benchmarks, not strategy tickers
        strategy_section = prompt.split("## Benchmark Tickers")[0]
        benchmark_section = prompt.split("## Benchmark Tickers")[1].split("##")[0]
        assert "AAPL" in strategy_section
        assert "SPY" in benchmark_section

    def test_build_prompt_includes_requested_period(
        self,
        code_generator: BacktestCodeGenerator,
        sample_backtest_params: BacktestParams,
    ) -> None:
        """Test that user-requested backtest period is in the prompt."""
        data_ranges = {
            "AAPL": DateRange(start_date=date(2015, 1, 1), end_date=date(2024, 1, 1), ticker="AAPL"),
        }

        prompt = code_generator._build_prompt(
            "Test strategy",
            ["AAPL"],
            data_ranges,
            params=sample_backtest_params,
        )

        assert "## Requested Backtest Period" in prompt
        assert "2020-01-01" in prompt
        assert "2023-12-31" in prompt

    def test_build_prompt_includes_runtime_parameters(
        self,
        code_generator: BacktestCodeGenerator,
        sample_backtest_params: BacktestParams,
    ) -> None:
        """Test that actual capital/fees/contribution values are in the prompt."""
        data_ranges = {
            "AAPL": DateRange(start_date=date(2015, 1, 1), end_date=date(2024, 1, 1), ticker="AAPL"),
        }

        prompt = code_generator._build_prompt(
            "Test strategy",
            ["AAPL"],
            data_ranges,
            params=sample_backtest_params,
        )

        assert "## Runtime Parameters" in prompt
        assert "initial_capital: 100000.0" in prompt
        assert "trading_fee_percent: 0.1" in prompt
        assert "slippage_percent: 0.05" in prompt
        assert "contribution_frequency: monthly" in prompt
        assert "contribution_amount: 1000.0" in prompt
        assert "dividend_reinvestment: True" in prompt

    def test_build_prompt_backward_compatible_without_params(
        self,
        code_generator: BacktestCodeGenerator,
    ) -> None:
        """Test that prompt still works without params (backward compatibility)."""
        data_ranges = {
            "AAPL": DateRange(start_date=date(2015, 1, 1), end_date=date(2024, 1, 1), ticker="AAPL"),
        }

        prompt = code_generator._build_prompt(
            "Test strategy",
            ["AAPL"],
            data_ranges,
        )

        assert "## Strategy" in prompt
        assert "AAPL" in prompt
        assert "## Runtime Parameters" not in prompt
        assert "## Requested Backtest Period" not in prompt


# =============================================================================
# Tests for generate (Task 7.4)
# =============================================================================


class TestGenerate:
    """Tests for the main generate method."""

    @pytest.mark.asyncio
    async def test_generate_success(
        self,
        code_generator: BacktestCodeGenerator,
        mock_generation_backend: MagicMock,
        mock_data_provider: MagicMock,
        sample_backtest_request: BacktestRequest,
        sample_generated_code: str,
    ) -> None:
        """Test successful code generation."""
        # Setup mocks
        mock_generation_backend.generate = AsyncMock(
            return_value=_make_backend_result(
                model_info=mock_generation_backend.get_model_info(),
                code=sample_generated_code,
                summary="RSI summary",
                required_tickers=["AAPL", "SPY", "TSLA"],
            )
        )
        mock_data_provider.get_available_date_range = AsyncMock(
            return_value=DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        )

        result = await code_generator.generate(sample_backtest_request)

        assert isinstance(result, GeneratedCode)
        assert "RSIMeanReversionStrategy" in result.code
        assert result.strategy_summary
        assert result.model_info.provider == "test_provider"

    @pytest.mark.asyncio
    async def test_generate_calls_validator(
        self,
        code_generator: BacktestCodeGenerator,
        mock_generation_backend: MagicMock,
        mock_data_provider: MagicMock,
        sample_backtest_request: BacktestRequest,
        sample_generated_code: str,
    ) -> None:
        """Test that generate calls the validator."""
        mock_validator = MagicMock()
        mock_validator.validate.return_value = ValidationResult(
            is_valid=True, errors=[], warnings=[]
        )
        code_generator.validator = mock_validator

        async def backend_generate(request):
            request.validate_code(sample_generated_code)
            return _make_backend_result(
                model_info=mock_generation_backend.get_model_info(),
                code=sample_generated_code,
                summary="RSI summary",
                required_tickers=["AAPL", "SPY", "TSLA"],
            )

        mock_generation_backend.generate = AsyncMock(side_effect=backend_generate)
        mock_data_provider.get_available_date_range = AsyncMock(
            return_value=DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        )

        await code_generator.generate(sample_backtest_request)

        mock_validator.validate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_validation_failure(
        self,
        code_generator: BacktestCodeGenerator,
        mock_generation_backend: MagicMock,
        mock_data_provider: MagicMock,
        sample_backtest_request: BacktestRequest,
    ) -> None:
        """Test that validation failure raises ValidationError."""
        mock_generation_backend.generate = AsyncMock(
            side_effect=ValidationError(
                "Generated code failed validation after retries",
                ["banned function"],
            )
        )
        mock_data_provider.get_available_date_range = AsyncMock(
            return_value=DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        )

        with pytest.raises(ValidationError, match="validation"):
            await code_generator.generate(sample_backtest_request)

    @pytest.mark.asyncio
    async def test_generate_passes_retry_budget_and_validator_to_backend(
        self,
        code_generator: BacktestCodeGenerator,
        mock_generation_backend: MagicMock,
        mock_data_provider: MagicMock,
        sample_backtest_request: BacktestRequest,
        sample_generated_code: str,
    ) -> None:
        """Service should pass retry budget and validation callable into the backend."""
        mock_validator = MagicMock()
        mock_validator.validate.side_effect = [
            ValidationResult(is_valid=False, errors=["banned function"], warnings=[]),
            ValidationResult(is_valid=True, errors=[], warnings=[]),
        ]
        code_generator.validator = mock_validator

        async def backend_generate(request):
            assert request.max_validation_retries == code_generator.MAX_VALIDATION_RETRIES
            assert request.validate_code is not None
            request.validate_code(sample_generated_code)
            request.validate_code(sample_generated_code)
            return _make_backend_result(
                model_info=mock_generation_backend.get_model_info(),
                code=sample_generated_code,
                summary="Retry summary",
                required_tickers=["AAPL", "SPY", "TSLA"],
            )

        mock_generation_backend.generate = AsyncMock(side_effect=backend_generate)
        mock_data_provider.get_available_date_range = AsyncMock(
            return_value=DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        )

        result = await code_generator.generate(sample_backtest_request)
        assert isinstance(result, GeneratedCode)
        assert mock_generation_backend.generate.await_count == 1
        assert mock_validator.validate.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_passes_web_search_override_to_backend(
        self,
        code_generator: BacktestCodeGenerator,
        mock_generation_backend: MagicMock,
        mock_data_provider: MagicMock,
        sample_backtest_request: BacktestRequest,
        sample_generated_code: str,
    ) -> None:
        """Request-level web search setting should override provider defaults."""

        async def backend_generate(request):
            assert request.config is not None
            assert request.config.extra["web_search_enabled"] is False
            return _make_backend_result(
                model_info=mock_generation_backend.get_model_info(),
                code=sample_generated_code,
                summary="Web search override summary",
                required_tickers=["AAPL", "SPY", "TSLA"],
            )

        mock_generation_backend.generate = AsyncMock(side_effect=backend_generate)
        mock_data_provider.get_available_date_range = AsyncMock(
            return_value=DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        )

        result = await code_generator.generate(sample_backtest_request)

        assert isinstance(result, GeneratedCode)
        assert mock_generation_backend.generate.await_count == 1

    @pytest.mark.asyncio
    async def test_generate_no_tickers_error(
        self,
        code_generator: BacktestCodeGenerator,
        sample_backtest_params: BacktestParams,
    ) -> None:
        """Test error when no tickers found in strategy."""
        request = BacktestRequest(
            strategy="Just make me money somehow.",
            params=sample_backtest_params,
        )
        # Remove benchmarks to have no tickers at all
        request.params.benchmarks = ["test"]  # This won't be extracted

        # Clear benchmarks after creation to test no-ticker scenario
        with patch.object(code_generator, "_extract_tickers", return_value=[]):
            with pytest.raises(CodeGenerationError, match="No ticker symbols found"):
                await code_generator.generate(request)

    @pytest.mark.asyncio
    async def test_generate_llm_failure(
        self,
        code_generator: BacktestCodeGenerator,
        mock_generation_backend: MagicMock,
        mock_data_provider: MagicMock,
        sample_backtest_request: BacktestRequest,
    ) -> None:
        """Test error handling when the generation backend fails."""
        mock_generation_backend.generate = AsyncMock(
            side_effect=CodeGenerationError("Agent generation failed: API timeout")
        )
        mock_data_provider.get_available_date_range = AsyncMock(
            return_value=DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        )

        with pytest.raises(CodeGenerationError, match="Agent generation failed"):
            await code_generator.generate(sample_backtest_request)


# =============================================================================
# Integration Tests
# =============================================================================


class TestCodeGeneratorIntegration:
    """Integration tests for the full generation workflow."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_real_template(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
        mock_generation_backend: MagicMock,
        sample_backtest_request: BacktestRequest,
        sample_generated_code: str,
    ) -> None:
        """Test full workflow using the actual prompt builder."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
            generation_backend=mock_generation_backend,
        )

        mock_generation_backend.generate = AsyncMock(
            return_value=_make_backend_result(
                model_info=mock_generation_backend.get_model_info(),
                code=sample_generated_code,
                summary="Real template summary",
                required_tickers=["AAPL", "SPY", "TSLA"],
            )
        )
        mock_data_provider.get_available_date_range = AsyncMock(
            return_value=DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        )

        result = await generator.generate(sample_backtest_request)

        assert result.code
        assert result.strategy_summary
        assert result.model_info

    def test_prompt_template_snapshot(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
        sample_backtest_params: BacktestParams,
    ) -> None:
        """Snapshot test verifying prompt contains required elements."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        data_ranges = {
            "AAPL": DateRange(
                start_date=date(2015, 1, 1),
                end_date=date(2024, 1, 1),
                ticker="AAPL",
            )
        }

        prompt = generator._build_prompt(
            "Buy AAPL when RSI is low",
            ["AAPL"],
            data_ranges,
        )

        # Verify prompt contains key elements (minimal: strategy + tickers + date range)
        assert "Buy AAPL when RSI is low" in prompt
        assert "AAPL" in prompt
        assert "2015-01-01" in prompt  # data range from provider, not params
        assert "2024-01-01" in prompt


# =============================================================================
# Multilingual Ticker Extraction Tests
# =============================================================================


class TestMultilingualTickerExtraction:
    """
    Tests for Unicode-aware ticker extraction.

    Verifies that ticker extraction works correctly with:
    - Korean text (한글)
    - Japanese text (日本語)
    - Chinese text (中文)
    - Mixed language text
    - Explicit ticker override
    """

    def test_extract_tickers_korean_text(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test ticker extraction from Korean strategy text."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # Korean text with ticker directly adjacent to Korean characters
        test_cases = [
            ("QLD만 계속 보유", ["QLD"]),
            ("SPY와 QQQ 매수", ["QQQ", "SPY"]),
            ("AAPL 주식을 사서 TSLA와 함께 보유", ["AAPL", "TSLA"]),
            ("매달 1000달러씩 VOO에 투자", ["VOO"]),
        ]

        for korean_text, expected_tickers in test_cases:
            result = generator._extract_tickers(korean_text)
            assert result == expected_tickers, f"Failed for: {korean_text}"

    def test_extract_tickers_japanese_text(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test ticker extraction from Japanese strategy text."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        test_cases = [
            ("QLDを買う", ["QLD"]),
            ("SPYとQQQを保有する", ["QQQ", "SPY"]),
            ("AAPLの株を購入", ["AAPL"]),
        ]

        for japanese_text, expected_tickers in test_cases:
            result = generator._extract_tickers(japanese_text)
            assert result == expected_tickers, f"Failed for: {japanese_text}"

    def test_extract_tickers_chinese_text(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test ticker extraction from Chinese strategy text."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        test_cases = [
            ("买入QLD", ["QLD"]),
            ("持有SPY和QQQ", ["QQQ", "SPY"]),
            ("每月投资1000美元到VOO", ["VOO"]),
        ]

        for chinese_text, expected_tickers in test_cases:
            result = generator._extract_tickers(chinese_text)
            assert result == expected_tickers, f"Failed for: {chinese_text}"

    def test_extract_tickers_mixed_language(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test ticker extraction from mixed language text."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # Mixed Korean and English
        result = generator._extract_tickers("Buy QLD and hold, QQQ도 매수")
        assert result == ["QLD", "QQQ"]

        # Mixed Japanese and English
        result = generator._extract_tickers("SPYを買って、VOOも purchase")
        assert result == ["SPY", "VOO"]

    def test_extract_tickers_with_spaces(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test that extraction works with spaces (backward compatibility)."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # Korean with spaces (original workaround)
        result = generator._extract_tickers("QLD 만 계속 보유")
        assert "QLD" in result

        # English (should still work)
        result = generator._extract_tickers("Buy AAPL and TSLA when price drops")
        assert result == ["AAPL", "TSLA"]

    def test_extract_tickers_explicit_override(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test explicit ticker parameter overrides extraction."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # Even if text has different tickers, explicit_tickers should win
        result = generator._extract_tickers(
            text="Buy AAPL",
            explicit_tickers=["QLD", "SPY"],
        )
        assert result == ["QLD", "SPY"]

        # Explicit tickers with Korean text
        result = generator._extract_tickers(
            text="어떤 텍스트든 상관없음",  # "Any text doesn't matter"
            explicit_tickers=["VOO"],
        )
        assert result == ["VOO"]

    def test_extract_tickers_korean_stock_codes(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test extraction of Korean 6-digit stock codes."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # Korean stock codes (6 digits)
        result = generator._extract_tickers("005930 매수")  # Samsung
        assert "005930" in result

        # Mixed US tickers and Korean codes
        result = generator._extract_tickers("SPY와 005930 보유")
        assert "SPY" in result
        assert "005930" in result

    def test_extract_tickers_blacklist_still_works(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test that blacklist filtering still works with new pattern."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # Blacklisted words should not be extracted
        result = generator._extract_tickers("Buy AAPL for the long term")
        assert "AAPL" in result
        # "FOR" and "THE" should be filtered (but note: new pattern requires 2+ chars)
        # So "FOR" might match, but should be in blacklist
        assert "FOR" not in result
        assert "THE" not in result

    def test_extract_tickers_with_punctuation(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test ticker extraction with various punctuation marks."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        test_cases = [
            ("QLD, SPY, QQQ 매수", ["QLD", "QQQ", "SPY"]),
            ("QLD. SPY. QQQ.", ["QLD", "QQQ", "SPY"]),
            ("QLD; SPY; QQQ;", ["QLD", "QQQ", "SPY"]),
            ("(QLD)와 (SPY) 보유", ["QLD", "SPY"]),
            ("QLD! SPY? QQQ~", ["QLD", "QQQ", "SPY"]),
        ]

        for text, expected_tickers in test_cases:
            result = generator._extract_tickers(text)
            assert result == expected_tickers, f"Failed for: {text}"

    def test_extract_tickers_no_false_positives(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test that common false positives are avoided."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # These should NOT extract non-ticker words
        result = generator._extract_tickers("The CEO of USA company")
        assert "CEO" not in result
        assert "USA" not in result

        # Single letters should not match (new pattern requires 2-5)
        result = generator._extract_tickers("I am a person")
        assert "I" not in result
        assert "AM" not in result

    def test_extract_tickers_empty_or_none(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test extraction with empty or whitespace-only text."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # Empty string
        result = generator._extract_tickers("")
        assert result == []

        # Only whitespace
        result = generator._extract_tickers("   \n\t  ")
        assert result == []

        # Only Korean text (no tickers)
        result = generator._extract_tickers("아무 티커도 없는 텍스트")
        assert result == []

    def test_extract_tickers_with_benchmarks(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test that benchmarks are added even if not in text."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # QLD in text, SPY in benchmarks
        result = generator._extract_tickers(
            text="QLD만 보유",
            benchmarks=["SPY"],
        )
        assert "QLD" in result
        assert "SPY" in result
        assert result == ["QLD", "SPY"]

    def test_extract_tickers_case_normalization(
        self,
        mock_llm_provider: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Test that explicit tickers are normalized to uppercase."""
        generator = BacktestCodeGenerator(
            llm_provider=mock_llm_provider,
            data_provider=mock_data_provider,
        )

        # Lowercase explicit tickers should be uppercased
        result = generator._extract_tickers(
            text="some text",
            explicit_tickers=["qld", "spy", "QQQ"],
        )
        assert result == ["QLD", "QQQ", "SPY"]
