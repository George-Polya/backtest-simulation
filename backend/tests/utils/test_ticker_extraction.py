"""
Comprehensive unit tests for ticker extraction utilities.

Tests dual-phase ticker validation system including:
- Code parsing for ticker extraction
- Multi-source ticker merging
- Data availability validation
"""

import pytest
import pandas as pd

from backend.utils.ticker_extraction import (
    extract_tickers_from_code,
    merge_ticker_sources,
    validate_required_tickers,
    format_ticker_report,
    TICKER_BLACKLIST,
)


class TestExtractTickersFromCode:
    """Test ticker extraction from Python code."""

    def test_load_data_with_tickers_param(self):
        """Test extraction from load_data(tickers=[...]) pattern."""
        code = '''
        data = load_data(
            tickers=["AAPL", "TSLA"],
            start_date="2020-01-01",
            end_date="2024-12-31"
        )
        '''
        result = extract_tickers_from_code(code)
        assert result == ["AAPL", "TSLA"]

    def test_load_data_positional_list(self):
        """Test extraction from load_data([...]) pattern."""
        code = '''
        data = load_data(["QLD", "SPY"], "2020-01-01", "2024-12-31")
        '''
        result = extract_tickers_from_code(code)
        assert result == ["QLD", "SPY"]

    def test_variable_assignment_ticker(self):
        """Test extraction from variable assignment patterns."""
        code = '''
        strategy_symbol = "QLD"
        benchmark_ticker = "SPY"
        '''
        result = extract_tickers_from_code(code)
        assert set(result) == {"QLD", "SPY"}

    def test_list_literal_pattern(self):
        """Test extraction from list literals."""
        code = '''
        tickers = ["AAPL", "MSFT", "GOOGL"]
        benchmark = ["SPY"]
        '''
        result = extract_tickers_from_code(code)
        assert set(result) == {"AAPL", "GOOGL", "MSFT", "SPY"}

    def test_blacklist_filtering(self):
        """Test that blacklisted words are filtered out."""
        code = '''
        # Common words that might match ticker pattern
        data = load_data(["AAPL", "IF", "AND", "TO"])
        '''
        result = extract_tickers_from_code(code)
        assert result == ["AAPL"]  # IF, AND, TO are blacklisted

    def test_deduplication(self):
        """Test that duplicate tickers are removed."""
        code = '''
        data1 = load_data(["AAPL", "SPY"])
        data2 = load_data(["AAPL", "TSLA"])
        ticker = "SPY"
        '''
        result = extract_tickers_from_code(code)
        assert result == ["AAPL", "SPY", "TSLA"]  # Sorted and deduplicated

    def test_case_sensitivity(self):
        """Test ticker normalization to uppercase."""
        code = '''
        # Variable assignments are normalized to uppercase
        ticker = "spy"
        symbol = "aapl"
        # load_data patterns only match uppercase in string literals
        data = load_data(["MSFT"])
        '''
        result = extract_tickers_from_code(code)
        # "spy" and "aapl" should be normalized to uppercase
        assert set(result) == {"AAPL", "MSFT", "SPY"}

    def test_empty_code(self):
        """Test handling of empty code."""
        result = extract_tickers_from_code("")
        assert result == []

    def test_no_tickers(self):
        """Test code without any tickers."""
        code = '''
        import pandas as pd
        def my_function():
            return 42
        '''
        result = extract_tickers_from_code(code)
        assert result == []

    def test_complex_real_world_code(self):
        """Test extraction from realistic backtest code."""
        code = '''
        from backtesting import Backtest, Strategy

        # Load market data
        data = load_data(
            tickers=["QLD", "SPY"],
            start_date="2020-01-01",
            end_date="2024-12-31"
        )

        strategy_symbol = "QLD"
        benchmark_symbol = "SPY"

        class MyStrategy(Strategy):
            def init(self):
                pass

            def next(self):
                if self.position.size == 0:
                    self.buy()

        def run_backtest(params):
            df = data["QLD"]
            bt = Backtest(df, MyStrategy, cash=100000)
            return bt.run()
        '''
        result = extract_tickers_from_code(code)
        assert set(result) == {"QLD", "SPY"}


class TestMergeTickerSources:
    """Test merging tickers from multiple sources."""

    def test_merge_all_sources(self):
        """Test merging tickers from all sources."""
        result = merge_ticker_sources(
            code_tickers=["AAPL", "TSLA"],
            llm_tickers=["AAPL", "MSFT"],
            params_tickers=["GOOGL"],
            benchmarks=["SPY"],
        )

        assert set(result["final"]) == {"AAPL", "GOOGL", "MSFT", "SPY", "TSLA"}
        assert result["by_source"]["code"] == ["AAPL", "TSLA"]
        assert result["by_source"]["llm"] == ["AAPL", "MSFT"]
        assert result["by_source"]["params"] == ["GOOGL"]
        assert result["by_source"]["benchmarks"] == ["SPY"]

    def test_merge_with_duplicates(self):
        """Test that duplicates are removed in final list."""
        result = merge_ticker_sources(
            code_tickers=["AAPL", "SPY"],
            llm_tickers=["AAPL"],
            params_tickers=["SPY"],
            benchmarks=["SPY"],
        )

        assert result["final"] == ["AAPL", "SPY"]  # Deduplicated

    def test_warning_code_not_in_llm(self):
        """Test warning when code uses tickers not declared by LLM."""
        result = merge_ticker_sources(
            code_tickers=["AAPL", "QLD"],  # QLD not in LLM
            llm_tickers=["AAPL"],
            params_tickers=[],
            benchmarks=[],
        )

        warnings = result["warnings"]
        assert len(warnings) > 0
        assert any("Code uses tickers not declared by LLM" in w for w in warnings)
        assert any("QLD" in w for w in warnings)

    def test_warning_llm_not_in_code(self):
        """Test warning when LLM declares tickers not used in code."""
        result = merge_ticker_sources(
            code_tickers=["AAPL"],
            llm_tickers=["AAPL", "TSLA"],  # TSLA not in code
            params_tickers=[],
            benchmarks=[],
        )

        warnings = result["warnings"]
        assert len(warnings) > 0
        assert any("LLM declared tickers not used in code" in w for w in warnings)
        assert any("TSLA" in w for w in warnings)

    def test_no_warnings_when_consistent(self):
        """Test no warnings when code and LLM are consistent."""
        result = merge_ticker_sources(
            code_tickers=["AAPL", "SPY"],
            llm_tickers=["AAPL", "SPY"],
            params_tickers=["AAPL"],
            benchmarks=["SPY"],
        )

        assert len(result["warnings"]) == 0

    def test_empty_sources(self):
        """Test handling of empty source lists."""
        result = merge_ticker_sources(
            code_tickers=[],
            llm_tickers=[],
            params_tickers=[],
            benchmarks=[],
        )

        assert result["final"] == []
        assert result["by_source"]["code"] == []

    def test_sorted_output(self):
        """Test that final list is sorted alphabetically."""
        result = merge_ticker_sources(
            code_tickers=["TSLA", "AAPL"],
            llm_tickers=["MSFT"],
            params_tickers=["GOOGL"],
            benchmarks=["SPY"],
        )

        # Should be sorted alphabetically
        assert result["final"] == ["AAPL", "GOOGL", "MSFT", "SPY", "TSLA"]


class TestValidateRequiredTickers:
    """Test ticker data availability validation."""

    def test_all_tickers_available(self):
        """Test validation when all required tickers are available."""
        required = ["AAPL", "SPY"]
        available_data = {
            "AAPL": pd.DataFrame({"close": [100, 101, 102]}),
            "SPY": pd.DataFrame({"close": [400, 401, 402]}),
        }

        # Should not raise any exception
        missing = validate_required_tickers(required, available_data)
        assert missing == []

    def test_missing_tickers(self):
        """Test validation when required tickers are missing."""
        required = ["AAPL", "TSLA", "SPY"]
        available_data = {
            "AAPL": pd.DataFrame({"close": [100, 101, 102]}),
            # TSLA and SPY missing
        }

        with pytest.raises(ValueError) as exc_info:
            validate_required_tickers(required, available_data)

        error_msg = str(exc_info.value)
        assert "SPY" in error_msg
        assert "TSLA" in error_msg
        assert "Required ticker data not available" in error_msg

    def test_empty_required_list(self):
        """Test validation with empty required list."""
        required = []
        available_data = {
            "AAPL": pd.DataFrame({"close": [100, 101, 102]}),
        }

        # Should pass without error
        missing = validate_required_tickers(required, available_data)
        assert missing == []

    def test_extra_available_tickers_ok(self):
        """Test that extra available tickers don't cause issues."""
        required = ["AAPL"]
        available_data = {
            "AAPL": pd.DataFrame({"close": [100, 101, 102]}),
            "SPY": pd.DataFrame({"close": [400, 401, 402]}),
            "TSLA": pd.DataFrame({"close": [200, 201, 202]}),
        }

        # Should pass - extra tickers are OK
        missing = validate_required_tickers(required, available_data)
        assert missing == []


class TestFormatTickerReport:
    """Test ticker merge report formatting."""

    def test_report_format(self):
        """Test that report includes all sections."""
        merge_result = merge_ticker_sources(
            code_tickers=["AAPL", "QLD"],
            llm_tickers=["AAPL"],
            params_tickers=["SPY"],
            benchmarks=["SPY"],
        )

        report = format_ticker_report(merge_result)

        # Check that report contains expected sections
        assert "=== Ticker Merge Report ===" in report
        assert "Final ticker list" in report
        assert "Breakdown by source:" in report
        assert "code" in report
        assert "llm" in report
        assert "params" in report
        assert "benchmarks" in report

    def test_report_with_warnings(self):
        """Test that warnings are included in report."""
        merge_result = merge_ticker_sources(
            code_tickers=["AAPL", "QLD"],
            llm_tickers=["AAPL", "TSLA"],
            params_tickers=[],
            benchmarks=[],
        )

        report = format_ticker_report(merge_result)

        # Check for warnings section
        assert "⚠️ Warnings:" in report
        assert "Code uses tickers not declared by LLM" in report
        assert "LLM declared tickers not used in code" in report

    def test_report_no_warnings(self):
        """Test report when no warnings exist."""
        merge_result = merge_ticker_sources(
            code_tickers=["AAPL"],
            llm_tickers=["AAPL"],
            params_tickers=["AAPL"],
            benchmarks=["SPY"],
        )

        report = format_ticker_report(merge_result)

        # Should not have warnings section
        assert "⚠️ Warnings:" not in report


class TestTickerBlacklist:
    """Test that blacklist constants are properly defined."""

    def test_blacklist_contains_common_words(self):
        """Test that common words are in blacklist."""
        assert "I" in TICKER_BLACKLIST
        assert "A" in TICKER_BLACKLIST
        assert "THE" in TICKER_BLACKLIST
        assert "AND" in TICKER_BLACKLIST
        assert "FOR" in TICKER_BLACKLIST

    def test_blacklist_contains_acronyms(self):
        """Test that common acronyms are in blacklist."""
        assert "API" in TICKER_BLACKLIST
        assert "CEO" in TICKER_BLACKLIST
        assert "GDP" in TICKER_BLACKLIST
        assert "ETF" in TICKER_BLACKLIST

    def test_blacklist_is_set(self):
        """Test that blacklist is a set for efficient lookup."""
        assert isinstance(TICKER_BLACKLIST, set)


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    def test_qld_spy_scenario(self):
        """Test the original QLD/SPY bug scenario."""
        # This is the scenario that caused the original bug
        code = '''
        data = load_data(["QLD"], start_date, end_date)
        strategy_symbol = "QLD"
        benchmark = load_data(["SPY"], start_date, end_date)
        '''

        # Extract tickers from code
        code_tickers = extract_tickers_from_code(code)
        assert set(code_tickers) == {"QLD", "SPY"}

        # Merge with params (simulating original bug where params only had SPY)
        merge_result = merge_ticker_sources(
            code_tickers=code_tickers,
            llm_tickers=[],  # LLM didn't declare tickers
            params_tickers=["SPY"],  # Only benchmark in params
            benchmarks=["SPY"],
        )

        # Final list should include both QLD and SPY
        assert set(merge_result["final"]) == {"QLD", "SPY"}

        # Should have warning about code using QLD
        assert len(merge_result["warnings"]) > 0

    def test_full_dual_phase_validation(self):
        """Test complete dual-phase validation workflow."""
        # Phase 1: LLM declares tickers
        llm_declared = ["AAPL", "TSLA", "SPY"]

        # Generated code
        code = '''
        data = load_data(["AAPL", "TSLA"], "2020-01-01", "2024-12-31")
        benchmark_data = load_data(["SPY"], "2020-01-01", "2024-12-31")
        '''

        # Phase 2: Extract from code
        code_tickers = extract_tickers_from_code(code)
        assert set(code_tickers) == {"AAPL", "SPY", "TSLA"}

        # Phase 3: Merge all sources
        merge_result = merge_ticker_sources(
            code_tickers=code_tickers,
            llm_tickers=llm_declared,
            params_tickers=["AAPL"],
            benchmarks=["SPY"],
        )

        # Should have no warnings (LLM and code match)
        assert len(merge_result["warnings"]) == 0
        assert set(merge_result["final"]) == {"AAPL", "SPY", "TSLA"}

        # Phase 4: Validate data availability
        available_data = {
            "AAPL": pd.DataFrame({"close": [100, 101]}),
            "TSLA": pd.DataFrame({"close": [200, 201]}),
            "SPY": pd.DataFrame({"close": [400, 401]}),
        }

        # Should pass validation
        missing = validate_required_tickers(merge_result["final"], available_data)
        assert missing == []

    def test_validation_failure_workflow(self):
        """Test workflow when data validation fails."""
        # Extract tickers from code
        code = '''
        data = load_data(["AAPL", "QLD", "SPY"])
        '''
        code_tickers = extract_tickers_from_code(code)

        # Merge tickers
        merge_result = merge_ticker_sources(
            code_tickers=code_tickers,
            llm_tickers=["AAPL", "QLD", "SPY"],
            params_tickers=["SPY"],
            benchmarks=["SPY"],
        )

        # Simulate data fetch failure (QLD data unavailable)
        available_data = {
            "AAPL": pd.DataFrame({"close": [100, 101]}),
            "SPY": pd.DataFrame({"close": [400, 401]}),
            # QLD missing!
        }

        # Validation should fail
        with pytest.raises(ValueError) as exc_info:
            validate_required_tickers(merge_result["final"], available_data)

        assert "QLD" in str(exc_info.value)
