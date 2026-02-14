import {
  BacktestParams,
  BacktestRequest,
  BacktestResultResponse,
  ContributionFrequency,
  GeneratedCode,
  JobStatus
} from '@/types';

describe('API types', () => {
  it('matches ContributionFrequency enum values', () => {
    expect(Object.values(ContributionFrequency)).toEqual([
      'monthly',
      'quarterly',
      'semiannual',
      'annual'
    ]);
  });

  it('matches JobStatus enum values including cancelled and timeout', () => {
    expect(Object.values(JobStatus)).toEqual([
      'pending',
      'running',
      'completed',
      'failed',
      'cancelled',
      'timeout'
    ]);
  });

  it('defines BacktestRequest with strategy and params', () => {
    const params: BacktestParams = {
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      initial_capital: 10000,
      contribution: {
        frequency: ContributionFrequency.Monthly,
        amount: 500
      },
      fees: {
        trading_fee_percent: 0.1,
        slippage_percent: 0.05
      },
      dividend_reinvestment: true,
      benchmarks: ['SPY'],
      llm_settings: {
        provider: 'openrouter',
        model: 'anthropic/claude-3.5-sonnet',
        web_search_enabled: false,
        seed: 42
      }
    };

    const request: BacktestRequest = {
      strategy: 'Buy SPY monthly and rebalance annually.',
      params
    };

    expect(request.strategy).toContain('SPY');
    expect(request.params.initial_capital).toBe(10000);
  });

  it('defines BacktestParams structure with nested fields', () => {
    const params: BacktestParams = {
      start_date: '2023-01-01',
      end_date: '2023-12-31',
      initial_capital: 50000,
      contribution: {
        frequency: ContributionFrequency.Quarterly,
        amount: 1000
      },
      fees: {
        trading_fee_percent: 0.2,
        slippage_percent: 0.1
      },
      dividend_reinvestment: false,
      benchmarks: ['SPY', 'QQQ'],
      explicit_tickers: ['AAPL', 'MSFT'],
      llm_settings: {
        provider: 'openai',
        model: 'gpt-4.1',
        web_search_enabled: true,
        seed: 7,
        temperature: 0.2
      },
      reference_date: '2024-01-15'
    };

    expect(params.benchmarks).toHaveLength(2);
    expect(params.llm_settings.provider).toBe('openai');
    expect(params.llm_settings.seed).toBe(7);
  });

  it('defines GeneratedCode shape with model_id and tickers', () => {
    const generated: GeneratedCode = {
      code: 'def run_backtest(params):\n  return {"status": "ok"}',
      strategy_summary: 'Momentum strategy using SPY and QQQ.',
      model_info: {
        provider: 'openrouter',
        model_id: 'anthropic/claude-3.5-sonnet',
        max_tokens: 8000,
        supports_system_prompt: true,
        cost_per_1k_input: 0.003,
        cost_per_1k_output: 0.015
      },
      tickers: ['SPY', 'QQQ']
    };

    expect(generated.model_info.model_id).toContain('claude');
    expect(generated.tickers).toEqual(['SPY', 'QQQ']);
  });

  it('defines BacktestResultResponse fields', () => {
    const result: BacktestResultResponse = {
      job_id: 'job-123',
      status: JobStatus.Completed,
      metrics: {
        total_return: 12.3,
        cagr: 6.1,
        max_drawdown: 8.4,
        sharpe_ratio: 1.2,
        sortino_ratio: 1.6,
        calmar_ratio: 0.73,
        volatility: 15.2,
        total_trades: 24,
        winning_trades: 14,
        losing_trades: 10,
        win_rate: 58.3
      },
      equity_curve: {
        strategy: [{ date: '2024-01-01', value: 10000 }],
        benchmark: [{ date: '2024-01-01', value: 10000 }],
        log_scale: false
      },
      drawdown: {
        data: [{ date: '2024-01-01', value: -0.01 }]
      },
      monthly_heatmap: {
        years: [2024],
        months: ['Jan'],
        returns: [[1.2]]
      },
      trades: [{ symbol: 'SPY' }],
      logs: 'Execution complete'
    };

    expect(result.status).toBe(JobStatus.Completed);
    expect(result.metrics.total_trades).toBe(24);
    expect(result.equity_curve.strategy[0]?.value).toBe(10000);
  });
});
