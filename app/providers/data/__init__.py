"""
Data Provider package.

Provides abstractions and implementations for market data providers.
"""

from app.providers.data.base import (
    AuthenticationError,
    CurrentPrice,
    DataProvider,
    DataProviderError,
    DataUnavailableError,
    DateRange,
    Exchange,
    InvalidDateRangeError,
    PriceData,
    RateLimitError,
    TickerInfo,
    TickerNotFoundError,
)
from app.providers.data.factory import (
    CachedDataProvider,
    DataProviderFactory,
    LRUCache,
    get_data_provider,
)
from app.providers.data.fallback import (
    FallbackDataProvider,
    get_resilient_data_provider,
)
from app.providers.data.mock import MockDataConfig, MockDataProvider
from app.providers.data.supabase import SupabaseDataProvider
from app.providers.data.yfinance import YFinanceDataProvider
from app.providers.data.local import LocalCSVDataProvider

__all__ = [
    # Base classes and types
    "DataProvider",
    "PriceData",
    "CurrentPrice",
    "TickerInfo",
    "DateRange",
    "Exchange",
    # Errors
    "DataProviderError",
    "TickerNotFoundError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidDateRangeError",
    "DataUnavailableError",
    # Implementations
    "YFinanceDataProvider",
    "SupabaseDataProvider",
    "LocalCSVDataProvider",
    "MockDataProvider",
    "MockDataConfig",
    # Factory and utilities
    "DataProviderFactory",
    "CachedDataProvider",
    "LRUCache",
    "get_data_provider",
    # Fallback
    "FallbackDataProvider",
    "get_resilient_data_provider",
]
