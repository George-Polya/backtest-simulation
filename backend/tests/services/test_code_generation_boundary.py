"""
Tests for the code-generation backend seam.
"""

from datetime import date
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.models.backtest import (
    BacktestParams,
    BacktestRequest,
    ContributionFrequency,
    ContributionPlan,
    FeeSettings,
)
from backend.providers.data.base import DataProvider, DateRange
from backend.providers.llm.base import GenerationResult, LLMProvider, ModelInfo
from backend.services.code_generation.base import CodeGenerationBackendResult
from backend.services.code_generator import BacktestCodeGenerator


@pytest.fixture
def sample_request() -> BacktestRequest:
    """Create a small request for backend seam tests."""
    return BacktestRequest(
        strategy="Buy AAPL when RSI is low.",
        params=BacktestParams(
            start_date=date(2020, 1, 1),
            end_date=date(2021, 1, 1),
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
        ),
    )


@pytest.mark.asyncio
async def test_generate_uses_injected_generation_backend(
    sample_request: BacktestRequest,
) -> None:
    """BacktestCodeGenerator should delegate candidate generation to the backend seam."""
    llm_provider = MagicMock(spec=LLMProvider)
    llm_provider.get_model_info.return_value = ModelInfo(
        model_id="test/model-v1",
        provider="test_provider",
        display_name="Test Model",
        max_context_tokens=128000,
        max_output_tokens=8000,
        cost_per_1k_input=Decimal("0.01"),
        cost_per_1k_output=Decimal("0.03"),
    )
    llm_provider.generate = AsyncMock(side_effect=AssertionError("direct LLM path should not run"))

    data_provider = MagicMock(spec=DataProvider)
    data_provider.get_available_date_range = AsyncMock(
        return_value=DateRange(
            start_date=date(2019, 1, 1),
            end_date=date(2022, 12, 31),
            ticker="AAPL",
        )
    )

    backend = MagicMock()
    backend.get_model_info.return_value = llm_provider.get_model_info.return_value
    generated_model_info = ModelInfo(
        model_id="agent/model-v2",
        provider="agent_provider",
        display_name="Agent Model",
        max_context_tokens=256000,
        max_output_tokens=16000,
        cost_per_1k_input=Decimal("0.02"),
        cost_per_1k_output=Decimal("0.04"),
    )
    backend.generate = AsyncMock(
        return_value=CodeGenerationBackendResult(
            generation_result=GenerationResult(
                content='{"code": "class ExampleStrategy(Strategy):\\n    def next(self):\\n        pass", "summary": "test"}',
                model_info=generated_model_info,
            ),
            code="class ExampleStrategy(Strategy):\n    def next(self):\n        pass",
            summary="test summary",
            required_tickers=["AAPL"],
        )
    )

    generator = BacktestCodeGenerator(
        llm_provider=llm_provider,
        data_provider=data_provider,
        generation_backend=backend,
    )

    result = await generator.generate(sample_request)

    assert result.code == "class ExampleStrategy(Strategy):\n    def next(self):\n        pass"
    assert result.strategy_summary == "test summary"
    assert "AAPL" in result.tickers
    assert "SPY" in result.tickers
    assert result.model_info.model_id == "agent/model-v2"
    assert result.model_info.provider == "agent_provider"
    assert backend.generate.await_count == 1
    llm_provider.generate.assert_not_awaited()

    backend_request = backend.generate.await_args.args[0]
    assert "Buy AAPL when RSI is low." in backend_request.prompt
    assert backend_request.max_validation_retries == generator.MAX_VALIDATION_RETRIES
    assert backend_request.validate_code is not None
    assert backend_request.validate_code("def broken(").is_valid is False
