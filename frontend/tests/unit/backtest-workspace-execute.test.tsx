import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { BacktestWorkspace } from '@/components/backtest';
import { useBacktestStore } from '@/stores';
import { BackendErrorCode, BacktestRequest, ContributionFrequency, JobStatus } from '@/types';
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

describe('BacktestWorkspace execute flow', () => {
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

  it('submits, polls status, and completes with results', async () => {
    const jobId = 'job-200';
    let statusCallCount = 0;

    server.use(
      http.post(`${API_BASE_URL}/backtest/execute`, async () => {
        return HttpResponse.json({
          job_id: jobId,
          status: JobStatus.Pending,
          message: 'submitted',
          result: null
        });
      }),
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

  it('shows failed state with error logs', async () => {
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

  it('shows job-not-found message on 404 status polling', async () => {
    const jobId = 'job-404';

    server.use(
      http.post(`${API_BASE_URL}/backtest/execute`, () =>
        HttpResponse.json({
          job_id: jobId,
          status: JobStatus.Pending,
          message: 'submitted',
          result: null
        })
      ),
      http.get(`${API_BASE_URL}/backtest/status/:jobId`, () =>
        HttpResponse.json({ detail: `Job not found: ${jobId}` }, { status: 404 })
      )
    );

    renderWorkspace();
    fireEvent.click(screen.getByRole('button', { name: 'Execute Backtest' }));

    await waitFor(
      () => {
        expect(screen.getByText('Job not found')).toBeInTheDocument();
      },
      { timeout: 5000 }
    );

    expect(screen.getByText(`Job not found: ${jobId}`)).toBeInTheDocument();
  });

  it(
    'auto-retries once with original payload when polling detects Docker error',
    async () => {
      const failedJobId = 'job-docker-fail';
      const retryJobId = 'job-docker-retry-ok';
      let executeCallCount = 0;
      const capturedPayloads: unknown[] = [];

      server.use(
        http.post(`${API_BASE_URL}/backtest/execute`, async ({ request }) => {
          executeCallCount += 1;
          capturedPayloads.push(await request.json());

          const jobId = executeCallCount === 1 ? failedJobId : retryJobId;
          return HttpResponse.json({
            job_id: jobId,
            status: JobStatus.Pending,
            message: 'submitted',
            result: null
          });
        }),
        http.get(`${API_BASE_URL}/backtest/status/:jobId`, ({ params }) => {
          const id = String(params.jobId);

          if (id === failedJobId) {
            return HttpResponse.json({ job_id: id, status: JobStatus.Failed });
          }

          return HttpResponse.json({ job_id: id, status: JobStatus.Completed });
        }),
        http.get(`${API_BASE_URL}/backtest/result/:jobId`, ({ params }) => {
          const id = String(params.jobId);

          if (id === failedJobId) {
            return HttpResponse.json({
              success: false,
              job_id: id,
              status: JobStatus.Failed,
              data: null,
              error: "Docker image 'backtest-runner:latest' is not available.",
              error_code: BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE,
              logs: '',
              duration_seconds: 0.1
            });
          }

          return HttpResponse.json({
            success: true,
            job_id: id,
            status: JobStatus.Completed,
            data: { equity_series: [{ date: '2024-01-01', value: 10000 }] },
            error: null,
            error_code: null,
            logs: 'ok',
            duration_seconds: 2.0
          });
        }),
        http.get(`${API_BASE_URL}/backtest/:jobId/result`, ({ params }) =>
          HttpResponse.json(createFormattedResult(String(params.jobId)))
        )
      );

      renderWorkspace();

      const originalCode = 'def run_backtest(params):\n    return {"status": "ok"}';

      fireEvent.click(screen.getByRole('button', { name: 'Execute Backtest' }));

      // After first submit, switch to Code tab and mutate the editor to a different value.
      // If retry correctly uses the captured payload, this change must NOT appear in the second call.
      await waitFor(() => {
        expect(executeCallCount).toBeGreaterThanOrEqual(1);
      });

      // Execute switches to results tab; switch back to code tab to access editor
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: 'Code' }));
      });

      const editor = screen.getByTestId('monaco-editor') as HTMLTextAreaElement;
      await act(async () => {
        fireEvent.change(editor, { target: { value: 'CHANGED_AFTER_SUBMIT' } });
      });

      // Wait for retry job to complete
      await waitFor(
        () => {
          expect(screen.getByText('Backtest completed')).toBeInTheDocument();
        },
        { timeout: 15_000 }
      );

      // Should have called /execute exactly twice (original + 1 retry)
      expect(executeCallCount).toBe(2);

      // First call: original code
      const firstPayload = capturedPayloads[0] as { code: string };
      expect(firstPayload.code).toBe(originalCode);

      // Second call (retry): must also be original code, NOT 'CHANGED_AFTER_SUBMIT'
      const secondPayload = capturedPayloads[1] as { code: string };
      expect(secondPayload.code).toBe(originalCode);
      expect(secondPayload.code).not.toBe('CHANGED_AFTER_SUBMIT');
    },
    20_000
  );

  it(
    'does not retry more than once on repeated Docker failures',
    async () => {
      let executeCallCount = 0;

      server.use(
        http.post(`${API_BASE_URL}/backtest/execute`, () => {
          executeCallCount += 1;
          return HttpResponse.json({
            job_id: `job-docker-${executeCallCount}`,
            status: JobStatus.Pending,
            message: 'submitted',
            result: null
          });
        }),
        http.get(`${API_BASE_URL}/backtest/status/:jobId`, ({ params }) =>
          HttpResponse.json({ job_id: String(params.jobId), status: JobStatus.Failed })
        ),
        http.get(`${API_BASE_URL}/backtest/result/:jobId`, ({ params }) =>
          HttpResponse.json({
            success: false,
            job_id: String(params.jobId),
            status: JobStatus.Failed,
            data: null,
            error: "Docker image not available.",
            error_code: BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE,
            logs: '',
            duration_seconds: 0.1
          })
        )
      );

      renderWorkspace();
      fireEvent.click(screen.getByRole('button', { name: 'Execute Backtest' }));

      // Wait for the retry to settle — both the original and retry job fail with Docker error
      await waitFor(
        () => {
          expect(executeCallCount).toBe(2);
        },
        { timeout: 12_000 }
      );

      // Wait another retry-delay cycle (2s) + margin to prove no 3rd call happens
      await act(async () => {
        await new Promise((r) => setTimeout(r, 3_500));
      });

      // Original + exactly 1 retry = 2 calls total, no infinite loop
      expect(executeCallCount).toBe(2);
    },
    20_000
  );
});
