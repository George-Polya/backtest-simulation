"""
CSV Exporter Service - Backtest Results to CSV Export.

This module provides comprehensive CSV export functionality for backtest
visualization data. It supports exporting:
1. Equity curves (strategy and benchmark)
2. Drawdown series
3. Monthly returns heatmap
4. Performance metrics comparison
5. Trade history

Follows SOLID principles with Protocol-based abstraction and comprehensive
type hints for maintainability.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from typing import Any, Protocol

import pandas as pd

from app.services.result_formatter import (
    FormattedResults,
    PerformanceMetrics,
    EquityCurveData,
    DrawdownData,
    MonthlyHeatmapData,
)


# =============================================================================
# Protocols and Interfaces
# =============================================================================


class CSVExportable(Protocol):
    """Protocol for objects that can be exported to CSV."""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for CSV export."""
        ...


class CSVFormatter(Protocol):
    """Protocol for CSV formatting strategies."""

    def format(self, df: pd.DataFrame) -> str:
        """Format DataFrame to CSV string."""
        ...


# =============================================================================
# CSV Formatter Implementations
# =============================================================================


@dataclass(frozen=True)
class StandardCSVFormatter:
    """
    Standard CSV formatter with configurable options.

    Attributes:
        include_index: Whether to include DataFrame index in output.
        float_format: Format string for float values.
        date_format: Format string for date values.
        encoding: Output encoding.
    """

    include_index: bool = False
    float_format: str = "%.2f"
    date_format: str = "%Y-%m-%d"
    encoding: str = "utf-8"

    def format(self, df: pd.DataFrame) -> str:
        """
        Format DataFrame to CSV string.

        Args:
            df: DataFrame to format.

        Returns:
            CSV formatted string.
        """
        buffer = StringIO()
        df.to_csv(
            buffer,
            index=self.include_index,
            float_format=self.float_format,
            date_format=self.date_format,
            encoding=self.encoding,
        )
        return buffer.getvalue()

    def format_bytes(self, df: pd.DataFrame) -> bytes:
        """
        Format DataFrame to CSV bytes.

        Args:
            df: DataFrame to format.

        Returns:
            CSV formatted bytes.
        """
        return self.format(df).encode(self.encoding)


# =============================================================================
# Abstract Base Exporter
# =============================================================================


class BaseCSVExporter(ABC):
    """Abstract base class for CSV exporters."""

    def __init__(self, formatter: CSVFormatter | None = None) -> None:
        """
        Initialize the exporter.

        Args:
            formatter: CSV formatter to use. Uses StandardCSVFormatter if None.
        """
        self._formatter = formatter or StandardCSVFormatter()

    @abstractmethod
    def to_dataframe(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Convert data to pandas DataFrame."""
        ...

    def to_csv_string(self, *args: Any, **kwargs: Any) -> str:
        """
        Export data to CSV string.

        Returns:
            CSV formatted string.
        """
        df = self.to_dataframe(*args, **kwargs)
        return self._formatter.format(df)

    def to_csv_bytes(self, *args: Any, **kwargs: Any) -> bytes:
        """
        Export data to CSV bytes.

        Returns:
            CSV formatted bytes.
        """
        if isinstance(self._formatter, StandardCSVFormatter):
            df = self.to_dataframe(*args, **kwargs)
            return self._formatter.format_bytes(df)
        return self.to_csv_string(*args, **kwargs).encode("utf-8")


# =============================================================================
# Equity Curve Exporter
# =============================================================================


class EquityCurveExporter(BaseCSVExporter):
    """
    Exporter for equity curve data.

    Exports strategy and benchmark equity values over time.

    Output format:
        date,strategy_value,benchmark_value
        2023-01-02,102500.50,101200.00
        ...
    """

    def to_dataframe(self, equity_curve: EquityCurveData) -> pd.DataFrame:
        """
        Convert equity curve data to DataFrame.

        Args:
            equity_curve: Equity curve data from FormattedResults.

        Returns:
            DataFrame with columns: date, strategy_value, benchmark_value.
        """
        # Extract strategy data
        strategy_data = [
            {"date": point.date, "strategy_value": point.value}
            for point in equity_curve.strategy
        ]

        df = pd.DataFrame(strategy_data)

        # Add benchmark if available
        if equity_curve.benchmark:
            benchmark_dict = {
                point.date: point.value for point in equity_curve.benchmark
            }
            df["benchmark_value"] = df["date"].map(benchmark_dict)
        else:
            df["benchmark_value"] = None

        return df


# =============================================================================
# Drawdown Exporter
# =============================================================================


class DrawdownExporter(BaseCSVExporter):
    """
    Exporter for drawdown series data.

    Exports drawdown percentages over time.

    Output format:
        date,drawdown_pct
        2023-03-15,-10.50
        ...
    """

    def to_dataframe(self, drawdown: DrawdownData) -> pd.DataFrame:
        """
        Convert drawdown data to DataFrame.

        Args:
            drawdown: Drawdown data from FormattedResults.

        Returns:
            DataFrame with columns: date, drawdown_pct.
        """
        data = [
            {"date": point.date, "drawdown_pct": point.value}
            for point in drawdown.data
        ]

        return pd.DataFrame(data)


# =============================================================================
# Monthly Returns Exporter
# =============================================================================


class MonthlyReturnsExporter(BaseCSVExporter):
    """
    Exporter for monthly returns heatmap data.

    Exports monthly returns organized by year and month.

    Output format:
        year,month,return_pct
        2023,Jan,2.35
        2023,Feb,-0.50
        ...
    """

    def to_dataframe(self, monthly_heatmap: MonthlyHeatmapData) -> pd.DataFrame:
        """
        Convert monthly heatmap data to DataFrame.

        Args:
            monthly_heatmap: Monthly heatmap data from FormattedResults.

        Returns:
            DataFrame with columns: year, month, return_pct.
        """
        rows: list[dict[str, Any]] = []

        for year_idx, year in enumerate(monthly_heatmap.years):
            if year_idx < len(monthly_heatmap.returns):
                year_returns = monthly_heatmap.returns[year_idx]

                for month_idx, month_name in enumerate(monthly_heatmap.months):
                    if month_idx < len(year_returns):
                        return_value = year_returns[month_idx]
                        # Only include months with data
                        if return_value is not None:
                            rows.append(
                                {
                                    "year": year,
                                    "month": month_name,
                                    "return_pct": return_value,
                                }
                            )

        return pd.DataFrame(rows)


# =============================================================================
# Performance Metrics Exporter
# =============================================================================


class PerformanceMetricsExporter(BaseCSVExporter):
    """
    Exporter for performance metrics comparison.

    Exports strategy and benchmark metrics with differences.

    Output format:
        metric,strategy_value,benchmark_value,difference
        Total Return (%),25.50,20.00,5.50
        CAGR (%),12.30,10.50,1.80
        ...
    """

    # Metric display names and their corresponding attribute names
    METRIC_MAPPINGS: list[tuple[str, str]] = [
        ("Total Return (%)", "total_return"),
        ("CAGR (%)", "cagr"),
        ("Max Drawdown (%)", "max_drawdown"),
        ("Volatility (%)", "volatility"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Sortino Ratio", "sortino_ratio"),
        ("Calmar Ratio", "calmar_ratio"),
        ("Total Trades", "total_trades"),
        ("Winning Trades", "winning_trades"),
        ("Losing Trades", "losing_trades"),
        ("Win Rate (%)", "win_rate"),
    ]

    def to_dataframe(
        self,
        strategy_metrics: PerformanceMetrics,
        benchmark_metrics: PerformanceMetrics | None = None,
    ) -> pd.DataFrame:
        """
        Convert performance metrics to DataFrame.

        Args:
            strategy_metrics: Strategy performance metrics.
            benchmark_metrics: Optional benchmark performance metrics.

        Returns:
            DataFrame with columns: metric, strategy_value, benchmark_value, difference.
        """
        rows: list[dict[str, Any]] = []

        for display_name, attr_name in self.METRIC_MAPPINGS:
            strategy_value = getattr(strategy_metrics, attr_name, 0.0)
            benchmark_value = (
                getattr(benchmark_metrics, attr_name, 0.0)
                if benchmark_metrics
                else None
            )
            difference = (
                strategy_value - benchmark_value
                if benchmark_value is not None
                else None
            )

            rows.append(
                {
                    "metric": display_name,
                    "strategy_value": strategy_value,
                    "benchmark_value": benchmark_value,
                    "difference": difference,
                }
            )

        return pd.DataFrame(rows)


# =============================================================================
# Trade History Exporter
# =============================================================================


class TradeHistoryExporter(BaseCSVExporter):
    """
    Exporter for trade history data.

    Exports individual trades with profit/loss information.

    Output format:
        date,symbol,action,quantity,price,profit_loss
        2023-01-15,AAPL,BUY,100,150.25,0.00
        2023-02-20,AAPL,SELL,50,165.00,737.50
        ...
    """

    def to_dataframe(self, trades: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Convert trade history to DataFrame.

        Args:
            trades: List of trade dictionaries with keys:
                - date: Trade date
                - symbol: Stock/asset symbol
                - action: "BUY" or "SELL"
                - quantity: Number of shares/units
                - price: Execution price
                - profit: Realized profit/loss

        Returns:
            DataFrame with columns: date, symbol, action, quantity, price, profit_loss.
        """
        if not trades:
            return pd.DataFrame(
                columns=["date", "symbol", "action", "quantity", "price", "profit_loss"]
            )

        rows: list[dict[str, Any]] = []

        for trade in trades:
            rows.append(
                {
                    "date": trade.get("date", ""),
                    "symbol": trade.get("symbol", ""),
                    "action": trade.get("action", ""),
                    "quantity": trade.get("quantity", 0),
                    "price": trade.get("price", 0.0),
                    "profit_loss": trade.get("profit", 0.0),
                }
            )

        return pd.DataFrame(rows)


# =============================================================================
# Unified CSV Export Service
# =============================================================================


class CSVExportService:
    """
    Unified service for exporting all backtest data to CSV.

    This is the main entry point for CSV exports, providing a clean interface
    to export any type of backtest data.

    Attributes:
        equity_exporter: Exporter for equity curve data.
        drawdown_exporter: Exporter for drawdown data.
        monthly_exporter: Exporter for monthly returns data.
        metrics_exporter: Exporter for performance metrics.
        trade_exporter: Exporter for trade history.
    """

    def __init__(self, formatter: CSVFormatter | None = None) -> None:
        """
        Initialize the CSV export service.

        Args:
            formatter: Optional custom CSV formatter for all exporters.
        """
        self.equity_exporter = EquityCurveExporter(formatter)
        self.drawdown_exporter = DrawdownExporter(formatter)
        self.monthly_exporter = MonthlyReturnsExporter(formatter)
        self.metrics_exporter = PerformanceMetricsExporter(formatter)
        self.trade_exporter = TradeHistoryExporter(formatter)

    def export_equity_curve(
        self,
        results: FormattedResults,
        as_bytes: bool = False,
    ) -> str | bytes:
        """
        Export equity curve data to CSV.

        Args:
            results: Formatted backtest results.
            as_bytes: If True, return bytes instead of string.

        Returns:
            CSV data as string or bytes.
        """
        if as_bytes:
            return self.equity_exporter.to_csv_bytes(results.equity_curve)
        return self.equity_exporter.to_csv_string(results.equity_curve)

    def export_drawdown(
        self,
        results: FormattedResults,
        as_bytes: bool = False,
    ) -> str | bytes:
        """
        Export drawdown data to CSV.

        Args:
            results: Formatted backtest results.
            as_bytes: If True, return bytes instead of string.

        Returns:
            CSV data as string or bytes.
        """
        if as_bytes:
            return self.drawdown_exporter.to_csv_bytes(results.drawdown)
        return self.drawdown_exporter.to_csv_string(results.drawdown)

    def export_monthly_returns(
        self,
        results: FormattedResults,
        as_bytes: bool = False,
    ) -> str | bytes:
        """
        Export monthly returns data to CSV.

        Args:
            results: Formatted backtest results.
            as_bytes: If True, return bytes instead of string.

        Returns:
            CSV data as string or bytes.
        """
        if as_bytes:
            return self.monthly_exporter.to_csv_bytes(results.monthly_heatmap)
        return self.monthly_exporter.to_csv_string(results.monthly_heatmap)

    def export_metrics(
        self,
        results: FormattedResults,
        benchmark_metrics: PerformanceMetrics | None = None,
        as_bytes: bool = False,
    ) -> str | bytes:
        """
        Export performance metrics to CSV.

        Args:
            results: Formatted backtest results.
            benchmark_metrics: Optional benchmark performance metrics.
            as_bytes: If True, return bytes instead of string.

        Returns:
            CSV data as string or bytes.
        """
        if as_bytes:
            return self.metrics_exporter.to_csv_bytes(
                results.metrics, benchmark_metrics
            )
        return self.metrics_exporter.to_csv_string(results.metrics, benchmark_metrics)

    def export_trades(
        self,
        trades: list[dict[str, Any]],
        as_bytes: bool = False,
    ) -> str | bytes:
        """
        Export trade history to CSV.

        Args:
            trades: List of trade dictionaries.
            as_bytes: If True, return bytes instead of string.

        Returns:
            CSV data as string or bytes.
        """
        if as_bytes:
            return self.trade_exporter.to_csv_bytes(trades)
        return self.trade_exporter.to_csv_string(trades)

    def get_equity_curve_dataframe(self, results: FormattedResults) -> pd.DataFrame:
        """
        Get equity curve as DataFrame.

        Args:
            results: Formatted backtest results.

        Returns:
            DataFrame with equity curve data.
        """
        return self.equity_exporter.to_dataframe(results.equity_curve)

    def get_drawdown_dataframe(self, results: FormattedResults) -> pd.DataFrame:
        """
        Get drawdown as DataFrame.

        Args:
            results: Formatted backtest results.

        Returns:
            DataFrame with drawdown data.
        """
        return self.drawdown_exporter.to_dataframe(results.drawdown)

    def get_monthly_returns_dataframe(self, results: FormattedResults) -> pd.DataFrame:
        """
        Get monthly returns as DataFrame.

        Args:
            results: Formatted backtest results.

        Returns:
            DataFrame with monthly returns data.
        """
        return self.monthly_exporter.to_dataframe(results.monthly_heatmap)

    def get_metrics_dataframe(
        self,
        results: FormattedResults,
        benchmark_metrics: PerformanceMetrics | None = None,
    ) -> pd.DataFrame:
        """
        Get performance metrics as DataFrame.

        Args:
            results: Formatted backtest results.
            benchmark_metrics: Optional benchmark performance metrics.

        Returns:
            DataFrame with performance metrics.
        """
        return self.metrics_exporter.to_dataframe(results.metrics, benchmark_metrics)

    def get_trades_dataframe(self, trades: list[dict[str, Any]]) -> pd.DataFrame:
        """
        Get trade history as DataFrame.

        Args:
            trades: List of trade dictionaries.

        Returns:
            DataFrame with trade history.
        """
        return self.trade_exporter.to_dataframe(trades)


# =============================================================================
# Factory Function
# =============================================================================


def create_csv_export_service(
    float_format: str = "%.2f",
    include_index: bool = False,
) -> CSVExportService:
    """
    Factory function to create a CSVExportService instance.

    Args:
        float_format: Format string for float values (default: 2 decimal places).
        include_index: Whether to include DataFrame index in CSV output.

    Returns:
        Configured CSVExportService instance.
    """
    formatter = StandardCSVFormatter(
        float_format=float_format,
        include_index=include_index,
    )
    return CSVExportService(formatter)
