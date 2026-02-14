"""
Local CSV Data Provider.

Provides market data from local CSV files with YFinance fallback.
When data is not available locally, fetches from YFinance and caches to CSV.
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal

from backend.providers.data.base import (
    CurrentPrice,
    DataProvider,
    DataProviderError,
    DateRange,
    Exchange,
    PriceData,
    TickerInfo,
    TickerNotFoundError,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=10)
def _get_exchange_calendar(exchange_code: str) -> mcal.MarketCalendar:
    """Get cached exchange calendar instance."""
    return mcal.get_calendar(exchange_code)


class LocalCSVDataProvider(DataProvider):
    """
    Local CSV-based data provider with YFinance fallback.

    Data flow:
    1. Query local CSV files for cached data
    2. Identify missing date ranges
    3. Fetch missing data from fallback provider (YFinance)
    4. Save fetched data to CSV for future queries
    5. Return combined data

    File structure:
    {storage_path}/
      ├── QQQ.csv
      ├── SPY.csv
      └── AAPL.csv

    CSV format:
    trade_date,open,high,low,close,volume,adjusted_close,source
    2024-01-01,100.00,101.50,99.00,100.50,1000000,100.50,yfinance
    """

    def __init__(
        self,
        storage_path: str | Path,
        fallback_provider: DataProvider | None = None,
    ) -> None:
        """
        Initialize the Local CSV data provider.

        Args:
            storage_path: Directory path for CSV file storage
            fallback_provider: Provider to use when data is not available locally
        """
        self._storage_path = Path(storage_path)
        self._fallback = fallback_provider
        self._initialized = False

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "local"

    async def initialize(self) -> None:
        """Initialize the provider and create storage directory."""
        if self._initialized:
            return

        # Create storage directory if it doesn't exist
        self._storage_path.mkdir(parents=True, exist_ok=True)

        if self._fallback and hasattr(self._fallback, "initialize"):
            await self._fallback.initialize()

        self._initialized = True
        logger.info(f"LocalCSVDataProvider initialized at {self._storage_path}")

    async def close(self) -> None:
        """Clean up resources."""
        if self._fallback:
            await self._fallback.close()
        self._initialized = False
        logger.info("LocalCSVDataProvider closed")

    async def _ensure_initialized(self) -> None:
        """Ensure the provider is initialized before use."""
        if not self._initialized:
            await self.initialize()

    def _get_csv_path(self, ticker: str) -> Path:
        """Get the CSV file path for a ticker."""
        return self._storage_path / f"{ticker.upper()}.csv"

    async def get_daily_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        exchange: Exchange | None = None,
    ) -> list[PriceData]:
        """
        Get daily OHLCV price data for a ticker.

        1. Query local CSV for cached data
        2. Identify missing date ranges
        3. Fetch missing data from fallback provider
        4. Save fetched data to CSV
        5. Return combined results

        Args:
            ticker: Ticker symbol (e.g., "QQQ", "SPY")
            start_date: Start date for historical data
            end_date: End date for historical data
            exchange: Exchange hint (optional)

        Returns:
            List of PriceData records sorted by date ascending
        """
        await self._ensure_initialized()
        ticker_upper = ticker.upper()

        # Step 1: Query local CSV
        cached_prices = await self._fetch_from_csv(ticker_upper, start_date, end_date)
        logger.debug(f"Local cache hit: {ticker_upper} ({len(cached_prices)} records)")

        # Step 2: Find missing date ranges (using exchange calendar for holidays)
        cached_dates = {p.date for p in cached_prices}
        missing_ranges = self._find_missing_ranges(start_date, end_date, cached_dates, exchange)

        if not missing_ranges:
            # All data available locally
            return sorted(cached_prices, key=lambda p: p.date)

        # Step 3: Fetch from fallback
        if self._fallback is None:
            logger.warning(f"No fallback provider for missing data: {ticker_upper}")
            return sorted(cached_prices, key=lambda p: p.date)

        fetched_prices: list[PriceData] = []
        for range_start, range_end in missing_ranges:
            # YFinance doesn't work well with single-day queries
            # Expand short ranges to at least 7 days for better API reliability
            fetch_start = range_start
            fetch_end = range_end
            if (range_end - range_start).days < 7:
                fetch_start = range_start - timedelta(days=3)
                fetch_end = range_end + timedelta(days=3)

            logger.info(f"Fetching from fallback: {ticker_upper} ({fetch_start} to {fetch_end})")
            try:
                prices = await self._fallback.get_daily_prices(
                    ticker_upper, fetch_start, fetch_end, exchange
                )
                # Filter to only include dates within the original requested range
                prices = [p for p in prices if range_start <= p.date <= range_end]
                fetched_prices.extend(prices)
            except Exception as e:
                logger.warning(f"Fallback fetch failed for {ticker_upper}: {e}")

        # Step 4: Save to CSV
        if fetched_prices:
            await self._save_to_csv(ticker_upper, fetched_prices)
            logger.info(f"Saved {len(fetched_prices)} records to CSV: {ticker_upper}")

        # Step 5: Combine and return
        all_prices = cached_prices + fetched_prices
        # Remove duplicates by date
        seen_dates: set[date] = set()
        unique_prices: list[PriceData] = []
        for p in sorted(all_prices, key=lambda x: x.date):
            if p.date not in seen_dates:
                seen_dates.add(p.date)
                unique_prices.append(p)

        return unique_prices

    async def _fetch_from_csv(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceData]:
        """Fetch price data from local CSV file."""
        csv_path = self._get_csv_path(ticker)

        if not csv_path.exists():
            return []

        try:
            df = pd.read_csv(csv_path, parse_dates=["trade_date"])
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

            # Filter by date range
            mask = (df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)
            filtered_df = df[mask]

            return [self._row_to_price_data(row) for _, row in filtered_df.iterrows()]
        except Exception as e:
            logger.warning(f"Error reading CSV for {ticker}: {e}")
            return []

    def _row_to_price_data(self, row: pd.Series) -> PriceData:
        """Convert a DataFrame row to PriceData."""
        return PriceData(
            date=row["trade_date"] if isinstance(row["trade_date"], date) else row["trade_date"].date(),
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=int(row["volume"]),
            adjusted_close=(
                Decimal(str(row["adjusted_close"])) if pd.notna(row.get("adjusted_close")) else None
            ),
            extra={"source": row.get("source", "local")},
        )

    async def _save_to_csv(
        self,
        ticker: str,
        prices: list[PriceData],
    ) -> None:
        """Save price data to CSV file with upsert logic."""
        csv_path = self._get_csv_path(ticker)

        # Ensure directory exists before saving
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare new data
        new_rows = [
            {
                "trade_date": p.date.isoformat(),
                "open": float(p.open),
                "high": float(p.high),
                "low": float(p.low),
                "close": float(p.close),
                "volume": p.volume,
                "adjusted_close": float(p.adjusted_close) if p.adjusted_close else None,
                "source": "yfinance",
            }
            for p in prices
        ]
        new_df = pd.DataFrame(new_rows)

        if csv_path.exists():
            # Load existing data and merge
            try:
                existing_df = pd.read_csv(csv_path)
                # Combine and remove duplicates (keep new data for conflicts)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.drop_duplicates(subset=["trade_date"], keep="last")
                combined_df = combined_df.sort_values("trade_date")
            except Exception as e:
                logger.warning(f"Error reading existing CSV for {ticker}: {e}")
                combined_df = new_df
        else:
            combined_df = new_df

        combined_df.to_csv(csv_path, index=False)

    def _find_missing_ranges(
        self,
        start_date: date,
        end_date: date,
        cached_dates: set[date],
        exchange: Exchange | None = None,
    ) -> list[tuple[date, date]]:
        """
        Find date ranges not covered by cached data.

        Uses pandas_market_calendars to identify actual trading days,
        excluding weekends AND holidays. Supports historical data from 1950+.

        Args:
            start_date: Start date of range
            end_date: End date of range
            cached_dates: Set of dates already in cache
            exchange: Exchange to get calendar for (defaults to NYSE)

        Returns:
            List of (start, end) tuples for missing date ranges
        """
        # Get appropriate exchange calendar
        calendar_code = self._get_calendar_code(exchange)
        calendar = _get_exchange_calendar(calendar_code)

        # Get trading days schedule from pandas_market_calendars (supports 1950+)
        schedule = calendar.schedule(
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )
        # Convert schedule index to set of dates for fast lookup
        trading_days: set[date] = {d.date() for d in schedule.index}

        # Find missing trading days not in cache
        missing_dates = sorted(trading_days - cached_dates)

        if not missing_dates:
            return []

        # Merge consecutive dates into ranges
        ranges: list[tuple[date, date]] = []
        range_start = missing_dates[0]
        prev = missing_dates[0]

        for d in missing_dates[1:]:
            # If gap is more than 5 days (accounting for long weekends/holidays), start new range
            if (d - prev).days > 5:
                ranges.append((range_start, prev))
                range_start = d
            prev = d

        ranges.append((range_start, prev))
        return ranges

    def _get_calendar_code(self, exchange: Exchange | None) -> str:
        """Map Exchange enum to pandas_market_calendars code."""
        if exchange is None:
            return "XNYS"  # Default to NYSE

        calendar_map = {
            Exchange.NYSE: "XNYS",
            Exchange.NASDAQ: "XNAS",
            Exchange.AMEX: "XNYS",  # AMEX uses NYSE calendar
            Exchange.KRX: "XKRX",
            Exchange.KRX_KOSPI: "XKRX",
            Exchange.KRX_KOSDAQ: "XKRX",
            Exchange.HKEX: "XHKG",
            Exchange.TSE: "XTKS",
            Exchange.SSE: "XSHG",
            Exchange.SZSE: "XSHE",
        }
        return calendar_map.get(exchange, "XNYS")

    async def get_current_price(
        self,
        ticker: str,
        exchange: Exchange | None = None,
        reference_date: date | None = None,
    ) -> CurrentPrice:
        """
        Get the current/latest price for a ticker.

        For real-time data, delegates to the fallback provider.
        If reference_date is provided, returns the closing price for that date.
        """
        await self._ensure_initialized()
        ticker_upper = ticker.upper()

        if reference_date is not None:
            # Get historical closing price for reference date
            prices = await self._fetch_from_csv(ticker_upper, reference_date, reference_date)
            if prices:
                p = prices[0]
                return CurrentPrice(
                    symbol=ticker_upper,
                    price=p.close,
                    change=Decimal("0"),
                    change_percent=Decimal("0"),
                    volume=p.volume,
                    timestamp=datetime.combine(p.date, datetime.min.time()),
                    extra={"source": "local", "is_historical": True},
                )

        # For real-time price, use fallback
        if self._fallback:
            return await self._fallback.get_current_price(ticker, exchange, reference_date)

        raise DataProviderError(
            f"No current price available for {ticker_upper}", self.provider_name
        )

    async def get_ticker_info(
        self,
        ticker: str,
        exchange: Exchange | None = None,
    ) -> TickerInfo:
        """
        Get metadata about a ticker.

        Delegates to fallback provider since local CSV doesn't store ticker metadata.
        """
        await self._ensure_initialized()

        if self._fallback:
            return await self._fallback.get_ticker_info(ticker, exchange)

        raise TickerNotFoundError(ticker.upper(), self.provider_name)

    async def get_available_date_range(
        self,
        ticker: str,
        exchange: Exchange | None = None,
        reference_date: date | None = None,
    ) -> DateRange:
        """
        Get the available date range for historical data.

        Queries min/max trade_date from local CSV file.
        """
        await self._ensure_initialized()
        ticker_upper = ticker.upper()
        csv_path = self._get_csv_path(ticker_upper)

        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path, parse_dates=["trade_date"])
                if not df.empty:
                    min_date = pd.to_datetime(df["trade_date"].min()).date()
                    max_date = pd.to_datetime(df["trade_date"].max()).date()
                    return DateRange(
                        start_date=min_date,
                        end_date=max_date,
                        ticker=ticker_upper,
                    )
            except Exception as e:
                logger.warning(f"Error reading CSV for date range: {e}")

        # Fallback
        if self._fallback:
            return await self._fallback.get_available_date_range(ticker, exchange, reference_date)

        raise TickerNotFoundError(ticker_upper, self.provider_name)

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            await self._ensure_initialized()
            # Check if storage directory is accessible
            return self._storage_path.exists() and self._storage_path.is_dir()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
