import baseline from '../regression/metrics-baseline.json';
import { mapBacktestResultToMetricsProps } from '@/lib/api';
import { BacktestResultResponse, JobStatus } from '@/types';

function createBaselineResult(): BacktestResultResponse {
  return {
    job_id: 'baseline-job',
    status: JobStatus.Completed,
    metrics: baseline.metrics,
    equity_curve: {
      strategy: [
        { date: '2024-01-01', value: 10000 },
        { date: '2024-12-31', value: 12450 }
      ],
      benchmark: [
        { date: '2024-01-01', value: 10000 },
        { date: '2024-12-31', value: 11100 }
      ],
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
    logs: 'baseline'
  };
}

describe('metrics regression against baseline', () => {
  it('keeps mapped summary values aligned with baseline output', () => {
    const viewModel = mapBacktestResultToMetricsProps(createBaselineResult());

    expect(viewModel.summary).toEqual(baseline.summary);
    expect(viewModel.metrics).toEqual(baseline.metrics);
  });
});
