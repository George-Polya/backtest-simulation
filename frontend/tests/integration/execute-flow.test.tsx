import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { BacktestWorkspace } from '@/components/backtest';
import { useBacktestStore } from '@/stores';
import { BacktestRequest, ContributionFrequency, JobStatus } from '@/types';
import { API_BASE_URL, server } from '../mocks/server';

vi.mock('@monaco-editor/react', () => ({
  default: ({
    value,
    onChange
  }: {
    value: string;
    onChange?: (value: string | undefined) => void;
  }) => (
    <textarea
      aria-label="Code Editor"
      data-testid="monaco-editor"
      onChange={(event) => onChange?.(event.target.value)}
      value={value}
    />
  )
}));

function createRequestConfig(): BacktestRequest {
  return {
    strategy: 'Use momentum with monthly rebalance and stop-loss.',
    params: {
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      initial_capital: 100000,
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
        web_search_enabled: false
      }
    }
  };
}

function createFormattedResult(jobId: string) {
  return {
    job_id: jobId,
    status: JobStatus.Completed,
    metrics: {
      total_return: 10,
      cagr: 5,
      max_drawdown: 6,
      sharpe_ratio: 1.1,
      sortino_ratio: 1.3,
      calmar_ratio: 0.8,
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
  };
}

function renderWorkspace() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BacktestWorkspace />
    </QueryClientProvider>
  );
}

describe('integration: execute flow', () => {
  beforeEach(() => {
    act(() => {
      useBacktestStore.getState().reset();
      useBacktestStore.getState().setRequestConfig(createRequestConfig());
      useBacktestStore
        .getState()
        .setGeneratedCode('def run_backtest(params):\n    return {"status": "ok"}');
    });
  });

  afterEach(() => {
    act(() => {
      useBacktestStore.getState().reset();
    });
  });

  it('runs execute flow from submit through polling to completed result', async () => {
    const jobId = 'job-200';
    let statusCallCount = 0;

    server.use(
      http.post(`${API_BASE_URL}/backtest/execute`, () =>
        HttpResponse.json({
          job_id: jobId,
          status: JobStatus.Pending,
          message: 'submitted',
          result: null
        })
      ),
      http.get(`${API_BASE_URL}/backtest/status/:jobId`, () => {
        statusCallCount += 1;

        if (statusCallCount === 1) {
          return HttpResponse.json({ job_id: jobId, status: JobStatus.Pending });
        }
        if (statusCallCount === 2) {
          return HttpResponse.json({ job_id: jobId, status: JobStatus.Running });
        }
        return HttpResponse.json({ job_id: jobId, status: JobStatus.Completed });
      }),
      http.get(`${API_BASE_URL}/backtest/result/:jobId`, ({ params }) =>
        HttpResponse.json({
          success: true,
          job_id: params.jobId,
          status: JobStatus.Completed,
          data: {
            equity_series: [{ date: '2024-01-01', value: 10000 }]
          },
          error: null,
          logs: 'execution done',
          duration_seconds: 2.1
        })
      ),
      http.get(`${API_BASE_URL}/backtest/:jobId/result`, ({ params }) =>
        HttpResponse.json(createFormattedResult(String(params.jobId)))
      )
    );

    renderWorkspace();
    fireEvent.click(screen.getByRole('button', { name: 'Execute Backtest' }));

    await waitFor(
      () => {
        expect(screen.getByText('Backtest pending')).toBeInTheDocument();
      },
      { timeout: 5000 }
    );

    await waitFor(
      () => {
        expect(screen.getByText('Backtest running')).toBeInTheDocument();
      },
      { timeout: 7000 }
    );

    await waitFor(
      () => {
        expect(screen.getByText('Backtest completed')).toBeInTheDocument();
      },
      { timeout: 9000 }
    );

    await waitFor(
      () => {
        expect(screen.getByText('Available')).toBeInTheDocument();
      },
      { timeout: 9000 }
    );
  });

  it('surfaces failure state and logs when terminal status is failed', async () => {
    const jobId = 'job-500';
    let statusCallCount = 0;

    server.use(
      http.post(`${API_BASE_URL}/backtest/execute`, () =>
        HttpResponse.json({
          job_id: jobId,
          status: JobStatus.Pending,
          message: 'submitted',
          result: null
        })
      ),
      http.get(`${API_BASE_URL}/backtest/status/:jobId`, () => {
        statusCallCount += 1;

        if (statusCallCount === 1) {
          return HttpResponse.json({ job_id: jobId, status: JobStatus.Pending });
        }

        return HttpResponse.json({ job_id: jobId, status: JobStatus.Failed });
      }),
      http.get(`${API_BASE_URL}/backtest/result/:jobId`, () =>
        HttpResponse.json({
          success: false,
          job_id: jobId,
          status: JobStatus.Failed,
          data: null,
          error: 'SyntaxError at line 12',
          logs: 'Traceback:\n  File "script.py", line 12',
          duration_seconds: 0.8
        })
      )
    );

    renderWorkspace();
    fireEvent.click(screen.getByRole('button', { name: 'Execute Backtest' }));

    await waitFor(
      () => {
        expect(screen.getByText('Backtest failed')).toBeInTheDocument();
      },
      { timeout: 7000 }
    );

    expect(screen.getByText('SyntaxError at line 12')).toBeInTheDocument();
    expect(screen.getByText(/Traceback/)).toBeInTheDocument();
  });
});
