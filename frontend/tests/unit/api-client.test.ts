import { http, HttpResponse } from 'msw';
import {
  executeBacktest,
  generateCode,
  getFormattedResult,
  getJobResult,
  getJobStatus
} from '@/lib/api';
import { API_BASE_URL, server } from '../mocks/server';
import { BacktestRequest, ContributionFrequency, JobStatus } from '@/types';

const generateResponse = {
  generated_code: {
    code: 'def run_backtest(params):\n  return {"equity_series": []}',
    strategy_summary: 'Simple SPY strategy.',
    model_info: {
      provider: 'openrouter',
      model_id: 'anthropic/claude-3.5-sonnet',
      max_tokens: 8000,
      supports_system_prompt: true,
      cost_per_1k_input: 0.003,
      cost_per_1k_output: 0.015
    },
    tickers: ['SPY']
  },
  tickers_found: ['SPY'],
  generation_time_seconds: 1.5
};

describe('api client', () => {
  beforeEach(() => {
    server.use(
      http.post(`${API_BASE_URL}/backtest/generate`, () => HttpResponse.json(generateResponse)),
      http.post(`${API_BASE_URL}/backtest/execute`, () =>
        HttpResponse.json({
          job_id: 'job-123',
          status: JobStatus.Pending,
          message: 'Backtest submitted successfully. Use job_id to track progress.',
          result: null
        })
      ),
      http.get(`${API_BASE_URL}/backtest/status/:jobId`, ({ params }) =>
        HttpResponse.json({
          job_id: params.jobId,
          status: JobStatus.Running
        })
      ),
      http.get(`${API_BASE_URL}/backtest/result/:jobId`, ({ params }) =>
        HttpResponse.json({
          success: true,
          job_id: params.jobId,
          status: JobStatus.Completed,
          data: {
            equity_series: [{ date: '2024-01-01', value: 10000 }]
          },
          error: null,
          logs: 'ok',
          duration_seconds: 2.3
        })
      ),
      http.get(`${API_BASE_URL}/backtest/:jobId/result`, ({ params }) =>
        HttpResponse.json({
          job_id: params.jobId,
          status: JobStatus.Completed,
          metrics: {
            total_return: 10,
            cagr: 5,
            max_drawdown: 7,
            sharpe_ratio: 1,
            sortino_ratio: 1.2,
            calmar_ratio: 0.7,
            volatility: 12,
            total_trades: 10,
            winning_trades: 6,
            losing_trades: 4,
            win_rate: 60
          },
          equity_curve: {
            strategy: [{ date: '2024-01-01', value: 10000 }],
            benchmark: null,
            log_scale: false
          },
          drawdown: {
            data: [{ date: '2024-01-01', value: -0.02 }]
          },
          monthly_heatmap: {
            years: [2024],
            months: ['Jan'],
            returns: [[1.2]]
          },
          trades: [{ symbol: 'SPY' }],
          logs: 'done'
        })
      )
    );
  });

  it('calls generateCode and returns wrapped generated code response', async () => {
    const payload: BacktestRequest = {
      strategy: 'Buy SPY and hold for one year.',
      params: {
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        initial_capital: 10000,
        contribution: {
          frequency: ContributionFrequency.Monthly,
          amount: 100
        },
        fees: {
          trading_fee_percent: 0.1,
          slippage_percent: 0.05
        },
        dividend_reinvestment: true,
        benchmarks: ['SPY'],
        llm_settings: {
          provider: 'openrouter',
          web_search_enabled: false
        }
      }
    };

    const result = await generateCode(payload);

    expect(result.generated_code.code).toContain('run_backtest');
    expect(result.generated_code.model_info.model_id).toContain('claude');
    expect(result.tickers_found).toEqual(['SPY']);
  });

  it('calls executeBacktest', async () => {
    const result = await executeBacktest({
      code: 'print("hello")',
      params: {
        start_date: '2024-01-01',
        end_date: '2024-12-31',
        initial_capital: 10000,
        contribution: {
          frequency: ContributionFrequency.Monthly,
          amount: 0
        },
        fees: {
          trading_fee_percent: 0.1,
          slippage_percent: 0.05
        },
        dividend_reinvestment: true,
        benchmarks: ['SPY'],
        llm_settings: {
          provider: 'openrouter',
          web_search_enabled: false
        }
      },
      async_mode: true
    });

    expect(result.job_id).toBe('job-123');
    expect(result.status).toBe(JobStatus.Pending);
  });

  it('calls getJobStatus', async () => {
    const result = await getJobStatus('job-123');

    expect(result.job_id).toBe('job-123');
    expect(result.status).toBe(JobStatus.Running);
  });

  it('calls getJobResult', async () => {
    const result = await getJobResult('job-123');

    expect(result.success).toBe(true);
    expect(result.job_id).toBe('job-123');
    expect(result.status).toBe(JobStatus.Completed);
  });

  it('calls getFormattedResult', async () => {
    const result = await getFormattedResult('job-123');

    expect(result.job_id).toBe('job-123');
    expect(result.status).toBe(JobStatus.Completed);
    expect(result.metrics.total_return).toBe(10);
    expect(result.drawdown.data[0]?.value).toBe(-0.02);
  });
});
