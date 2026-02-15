import {
  mapBacktestResultToMetricsProps,
  mapExecutionResultToCharts,
  mapJobStatusToUiStatus,
  mapTradesToTableRows
} from '@/lib/api/mappers';
import { BacktestResultResponse, ExecutionResult, JobStatus } from '@/types';

function createResult(benchmark = true): BacktestResultResponse {
  return {
    job_id: 'job-777',
    status: JobStatus.Completed,
    metrics: {
      total_return: 24.5,
      cagr: 11.2,
      max_drawdown: 6.4,
      sharpe_ratio: 1.2,
      sortino_ratio: 1.8,
      calmar_ratio: 0.9,
      volatility: 13.1,
      total_trades: 20,
      winning_trades: 11,
      losing_trades: 9,
      win_rate: 55
    },
    equity_curve: {
      strategy: [
        { date: '2024-01-01', value: 10000 },
        { date: '2024-12-31', value: 12450 }
      ],
      benchmark: benchmark
        ? [
            { date: '2024-01-01', value: 10000 },
            { date: '2024-12-31', value: 11100 }
          ]
        : null,
      log_scale: false
    },
    drawdown: {
      data: [{ date: '2024-06-01', value: -6.4 }]
    },
    monthly_heatmap: {
      years: [2024],
      months: ['Jan'],
      returns: [[1.2]]
    },
    trades: [{ symbol: 'SPY', action: 'BUY' }],
    logs: 'done'
  };
}

describe('mapBacktestResultToMetricsProps', () => {
  it('maps full result with formatted summary and period label', () => {
    const viewModel = mapBacktestResultToMetricsProps(createResult(true));

    expect(viewModel.metrics?.total_return).toBe(24.5);
    expect(viewModel.strategyCurve?.length).toBe(2);
    expect(viewModel.benchmarkCurve?.length).toBe(2);
    expect(viewModel.periodLabel).toContain('2024');
    expect(viewModel.summary?.totalReturn).toBe('24.50%');
    expect(viewModel.summary?.maxDrawdown).toBe('-6.40%');
  });

  it('handles missing benchmark and null input safely', () => {
    const withoutBenchmark = mapBacktestResultToMetricsProps(createResult(false));
    expect(withoutBenchmark.benchmarkCurve).toBeNull();

    const empty = mapBacktestResultToMetricsProps(null);
    expect(empty.metrics).toBeNull();
    expect(empty.summary).toBeNull();
    expect(empty.periodLabel).toBeNull();
  });
});

describe('mapExecutionResultToCharts', () => {
  it('maps charts from legacy execution payload keys', () => {
    const payload: ExecutionResult = {
      success: true,
      job_id: 'job-legacy',
      status: JobStatus.Completed,
      logs: '',
      data: {
        equity_series: [
          { date: '2024-01-01', value: 10000 },
          { date: '2024-12-31', value: 12100 }
        ],
        drawdown_series: [{ date: '2024-08-01', value: -4.5 }],
        trades: [{ symbol: 'AAPL', side: 'BUY' }]
      },
      error: null,
      duration_seconds: 1.1
    };

    const mapped = mapExecutionResultToCharts(payload);

    expect(mapped.equityCurve?.strategy.length).toBe(2);
    expect(mapped.equityCurve?.benchmark).toBeNull();
    expect(mapped.drawdown?.data[0]?.value).toBe(-4.5);
    expect(mapped.trades.length).toBe(1);
  });

  it('returns safe empty structures for malformed payload', () => {
    const mapped = mapExecutionResultToCharts({
      success: false,
      job_id: 'job-empty',
      status: JobStatus.Failed,
      data: null,
      error: 'error',
      logs: 'failed'
    });

    expect(mapped.equityCurve).toBeNull();
    expect(mapped.drawdown).toBeNull();
    expect(mapped.trades).toEqual([]);
  });
});

describe('mapTradesToTableRows', () => {
  it('formats partial trade rows with fallbacks and display strings', () => {
    const rows = mapTradesToTableRows([
      {
        ticker: 'QQQ',
        side: 'sell',
        date: '2024-11-01',
        pnl: -123.4
      },
      {
        symbol: 'SPY',
        action: 'BUY',
        entry_date: '2024-02-01',
        exit_date: '2024-03-01',
        profit: 240,
        profit_pct: 2.45
      }
    ]);

    expect(rows).toHaveLength(2);
    expect(rows[0]?.symbol).toBe('QQQ');
    expect(rows[0]?.action).toBe('SELL');
    expect(rows[0]?.profitLabel).toBe('-$123.40');
    expect(rows[0]?.entryDateLabel).toContain('2024');
    expect(rows[0]?.profitPctLabel).toBe('-');
    expect(rows[1]?.profitPctLabel).toBe('+2.45%');
  });
});

describe('mapJobStatusToUiStatus', () => {
  it('maps job status payload to user-friendly labels', () => {
    expect(mapJobStatusToUiStatus(JobStatus.Pending).label).toBe('Backtest pending');
    expect(mapJobStatusToUiStatus(JobStatus.Completed).tone).toBe('success');
    expect(mapJobStatusToUiStatus(JobStatus.Timeout).isTerminal).toBe(true);
    expect(mapJobStatusToUiStatus(null).label).toBe('Backtest not started');
  });
});
