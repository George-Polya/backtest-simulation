import React, { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useJobPolling } from '@/hooks';
import {
  ApiError,
  getFormattedResult,
  getJobResult,
  getJobStatus
} from '@/lib/api';
import { useBacktestStore } from '@/stores';
import { JobStatus } from '@/types';

vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return {
    ...actual,
    getJobStatus: vi.fn(),
    getJobResult: vi.fn(),
    getFormattedResult: vi.fn()
  };
});

const mockedGetJobStatus = vi.mocked(getJobStatus);
const mockedGetJobResult = vi.mocked(getJobResult);
const mockedGetFormattedResult = vi.mocked(getFormattedResult);

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  });

  return function Wrapper({ children }: PropsWithChildren) {
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

describe('useJobPolling', () => {
  beforeEach(() => {
    mockedGetJobStatus.mockReset();
    mockedGetJobResult.mockReset();
    mockedGetFormattedResult.mockReset();
    act(() => {
      useBacktestStore.getState().reset();
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    act(() => {
      useBacktestStore.getState().reset();
    });
  });

  it('polls every 2s and hydrates results on completed status', async () => {
    let statusCallCount = 0;
    mockedGetJobStatus.mockImplementation(async () => {
      statusCallCount += 1;

      if (statusCallCount === 1) {
        return { job_id: 'job-1', status: JobStatus.Pending };
      }

      if (statusCallCount === 2) {
        return { job_id: 'job-1', status: JobStatus.Running };
      }

      return { job_id: 'job-1', status: JobStatus.Completed };
    });

    mockedGetJobResult.mockResolvedValue({
      success: true,
      job_id: 'job-1',
      status: JobStatus.Completed,
      data: {
        equity_series: [{ date: '2024-01-01', value: 10000 }]
      },
      error: null,
      logs: 'execution finished',
      duration_seconds: 2.4
    });

    mockedGetFormattedResult.mockResolvedValue({
      job_id: 'job-1',
      status: JobStatus.Completed,
      metrics: {
        total_return: 10,
        cagr: 5,
        max_drawdown: 3,
        sharpe_ratio: 1.1,
        sortino_ratio: 1.2,
        calmar_ratio: 0.8,
        volatility: 12,
        total_trades: 8,
        winning_trades: 5,
        losing_trades: 3,
        win_rate: 62.5
      },
      equity_curve: {
        strategy: [{ date: '2024-01-01', value: 10000 }],
        benchmark: null,
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
      logs: 'formatted'
    });

    const { result } = renderHook(() => useJobPolling('job-1'), {
      wrapper: createWrapper()
    });

    await waitFor(() => {
      expect(result.current.status).toBe(JobStatus.Pending);
    });

    await waitFor(
      () => {
        expect(result.current.status).toBe(JobStatus.Running);
      },
      { timeout: 6_000 }
    );

    await waitFor(
      () => {
        expect(result.current.status).toBe(JobStatus.Completed);
      },
      { timeout: 9_000 }
    );

    await waitFor(() => {
      expect(mockedGetJobResult).toHaveBeenCalledWith('job-1');
      expect(mockedGetFormattedResult).toHaveBeenCalledWith('job-1');
    });

    expect(mockedGetJobStatus).toHaveBeenCalledTimes(3);
    expect(useBacktestStore.getState().results?.job_id).toBe('job-1');
  }, 15_000);

  it('handles 404 job-not-found without retries', async () => {
    mockedGetJobStatus.mockRejectedValue(new ApiError('Job not found: job-404', 'HTTP_ERROR', 404));

    const { result } = renderHook(() => useJobPolling('job-404'), {
      wrapper: createWrapper()
    });

    await waitFor(() => {
      expect(result.current.isJobNotFound).toBe(true);
    });

    expect(mockedGetJobStatus).toHaveBeenCalledTimes(1);
    await waitFor(() => {
      expect(useBacktestStore.getState().executionResult?.error).toContain('Job not found');
    });
  });

  it('retries network errors up to 3 times with backoff', async () => {
    mockedGetJobStatus.mockRejectedValue(new ApiError('Network unreachable', 'NETWORK_ERROR'));

    const { result } = renderHook(() => useJobPolling('job-net'), {
      wrapper: createWrapper()
    });

    await waitFor(() => {
      expect(mockedGetJobStatus).toHaveBeenCalledTimes(1);
    });

    await waitFor(
      () => {
        expect(mockedGetJobStatus).toHaveBeenCalledTimes(4);
      },
      { timeout: 6_000 }
    );

    expect(result.current.error?.message).toContain('Network');
  }, 10_000);
});
