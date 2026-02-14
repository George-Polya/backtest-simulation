"""
Supabase Data Provider.

Provides market data from Supabase database with YFinance fallback.
When data is not available in Supabase, fetches from YFinance and caches to Supabase.
"""

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from functools import lru_cache
from typing import Any

import pandas_market_calendars as mcal
from supabase import AsyncClient, create_async_client


@lru_cache(maxsize=10)
def _get_exchange_calendar(exchange_code: str) -> mcal.MarketCalendar:
    """Get cached exchange calendar instance."""
    return mcal.get_calendar(exchange_code)

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


class SupabaseDataProvider(DataProvider):
    """
    Supabase-based data provider with YFinance fallback.

    Data flow:
    1. Query Supabase prices table
    2. If data missing, fetch from fallback provider (YFinance)
    3. Save fetched data to Supabase for future queries
    4. Return combined data

    This approach uses Supabase as a persistent cache while ensuring
    data availability through the fallback mechanism.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
        fallback_provider: DataProvider | None = None,
    ) -> None:
        """
        Initialize the Supabase data provider.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase anonymous or service role key
            fallback_provider: Provider to use when data is not in Supabase
        """
        self._url = supabase_url
        self._key = supabase_key
        self._client: AsyncClient | None = None
        self._fallback = fallback_provider
        self._initialized = False

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "supabase"

    async def initialize(self) -> None:
        """Initialize the Supabase async client."""
        if self._initialized:
            return

        self._client = await create_async_client(self._url, self._key)

        if self._fallback and hasattr(self._fallback, "initialize"):
            await self._fallback.initialize()

        self._initialized = True
        logger.info("SupabaseDataProvider initialized")

    async def close(self) -> None:
        """Clean up resources."""
        if self._fallback:
            await self._fallback.close()
        self._initialized = False
        logger.info("SupabaseDataProvider closed")

    async def _ensure_initialized(self) -> None:
        """Ensure the provider is initialized before use."""
        if not self._initialized:
            await self.initialize()

    async def get_daily_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        exchange: Exchange | None = None,
    ) -> list[PriceData]:
        """
        Get daily OHLCV price data for a ticker.

        1. Query Supabase for cached data
        2. Identify missing date ranges
        3. Fetch missing data from fallback provider
        4. Save fetched data to Supabase
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

        # Step 1: Query Supabase
        cached_prices = await self._fetch_from_supabase(ticker_upper, start_date, end_date)
        logger.debug(f"Supabase cache hit: {ticker_upper} ({len(cached_prices)} records)")

        # Step 2: Find missing date ranges (using exchange calendar for holidays)
        cached_dates = {p.date for p in cached_prices}
        missing_ranges = self._find_missing_ranges(start_date, end_date, cached_dates, exchange)

        if not missing_ranges:
            # All data available in Supabase
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
                prices = [
                    p for p in prices if range_start <= p.date <= range_end
                ]
                fetched_prices.extend(prices)
            except Exception as e:
                logger.warning(f"Fallback fetch failed for {ticker_upper}: {e}")

        # Step 4: Save to Supabase
        if fetched_prices:
            await self._save_to_supabase(ticker_upper, fetched_prices)
            logger.info(f"Saved {len(fetched_prices)} records to Supabase: {ticker_upper}")

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

    async def _fetch_from_supabase(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
    ) -> list[PriceData]:
        """Fetch price data from Supabase prices table."""
        if self._client is None:
            raise DataProviderError("Supabase client not initialized", self.provider_name)

        response = await (
            self._client.table("prices")
            .select("*")
            .eq("ticker", ticker)
            .gte("trade_date", start_date.isoformat())
            .lte("trade_date", end_date.isoformat())
            .order("trade_date")
            .execute()
        )

        return [self._row_to_price_data(row) for row in response.data]

    def _row_to_price_data(self, row: dict[str, Any]) -> PriceData:
        """Convert a database row to PriceData."""
        return PriceData(
            date=date.fromisoformat(row["trade_date"]),
            open=Decimal(str(row["open"])),
            high=Decimal(str(row["high"])),
            low=Decimal(str(row["low"])),
            close=Decimal(str(row["close"])),
            volume=int(row["volume"]),
            adjusted_close=(
                Decimal(str(row["adjusted_close"])) if row.get("adjusted_close") else None
            ),
            extra={"source": row.get("source", "supabase")},
        )

    async def _save_to_supabase(
        self,
        ticker: str,
        prices: list[PriceData],
    ) -> None:
        """Save price data to Supabase with upsert."""
        if self._client is None:
            raise DataProviderError("Supabase client not initialized", self.provider_name)

        rows = [
            {
                "ticker": ticker,
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

        # Batch upsert - Supabase handles conflict on (ticker, trade_date)
        await (
            self._client.table("prices")
            .upsert(rows, on_conflict="ticker,trade_date")
            .execute()
        )

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
            prices = await self._fetch_from_supabase(
                ticker_upper, reference_date, reference_date
            )
            if prices:
                p = prices[0]
                return CurrentPrice(
                    symbol=ticker_upper,
                    price=p.close,
                    change=Decimal("0"),
                    change_percent=Decimal("0"),
                    volume=p.volume,
                    timestamp=datetime.combine(p.date, datetime.min.time()),
                    extra={"source": "supabase", "is_historical": True},
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

        Queries the tickers table first, falls back to fallback provider.
        """
        await self._ensure_initialized()
        ticker_upper = ticker.upper()

        if self._client is None:
            raise DataProviderError("Supabase client not initialized", self.provider_name)

        # Query tickers table
        response = await (
            self._client.table("tickers")
            .select("*")
            .eq("symbol", ticker_upper)
            .maybe_single()
            .execute()
        )

        if response.data:
            row = response.data
            return TickerInfo(
                symbol=row["symbol"],
                name=row.get("name") or row["symbol"],
                exchange=self._parse_exchange(row.get("exchange")),
                currency=row.get("currency", "USD"),
                security_type=row.get("security_type", "etf"),
                extra={"source": "supabase"},
            )

        # Fallback
        if self._fallback:
            return await self._fallback.get_ticker_info(ticker, exchange)

        raise TickerNotFoundError(ticker_upper, self.provider_name)

    def _parse_exchange(self, exchange_str: str | None) -> Exchange:
        """Parse exchange string to Exchange enum."""
        if not exchange_str:
            return Exchange.NASDAQ

        exchange_map = {
            "NYSE": Exchange.NYSE,
            "NASDAQ": Exchange.NASDAQ,
            "NAS": Exchange.NASDAQ,
            "AMEX": Exchange.AMEX,
            "AMS": Exchange.AMEX,
        }
        return exchange_map.get(exchange_str.upper(), Exchange.NASDAQ)

    async def get_available_date_range(
        self,
        ticker: str,
        exchange: Exchange | None = None,
        reference_date: date | None = None,
    ) -> DateRange:
        """
        Get the available date range for historical data.

        Queries min/max trade_date from prices table.
        """
        await self._ensure_initialized()
        ticker_upper = ticker.upper()

        if self._client is None:
            raise DataProviderError("Supabase client not initialized", self.provider_name)

        # Get min and max dates from prices table
        # Using separate queries since Supabase doesn't have built-in aggregate functions
        min_response = await (
            self._client.table("prices")
            .select("trade_date")
            .eq("ticker", ticker_upper)
            .order("trade_date", desc=False)
            .limit(1)
            .execute()
        )

        max_response = await (
            self._client.table("prices")
            .select("trade_date")
            .eq("ticker", ticker_upper)
            .order("trade_date", desc=True)
            .limit(1)
            .execute()
        )

        if min_response.data and max_response.data:
            return DateRange(
                start_date=date.fromisoformat(min_response.data[0]["trade_date"]),
                end_date=date.fromisoformat(max_response.data[0]["trade_date"]),
                ticker=ticker_upper,
            )

        # Fallback
        if self._fallback:
            return await self._fallback.get_available_date_range(
                ticker, exchange, reference_date
            )

        raise TickerNotFoundError(ticker_upper, self.provider_name)

    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible."""
        try:
            await self._ensure_initialized()
            if self._client is None:
                return False

            # Simple query to verify connection
            await self._client.table("tickers").select("id").limit(1).execute()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
