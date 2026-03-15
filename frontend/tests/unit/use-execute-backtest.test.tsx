import React, { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useExecuteBacktest } from '@/hooks';
import { ApiError, executeBacktest } from '@/lib/api';
import { useBacktestStore } from '@/stores';
import { BackendErrorCode, ContributionFrequency, ExecuteBacktestRequest, JobStatus } from '@/types';

vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return {
    ...actual,
    executeBacktest: vi.fn()
  };
});

const mockedExecuteBacktest = vi.mocked(executeBacktest);

function createPayload(): ExecuteBacktestRequest {
  return {
    code: 'def run_backtest(params):\n    return {"status": "ok"}',
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
    },
    async_mode: false
  };
}

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

describe('useExecuteBacktest', () => {
  beforeEach(() => {
    mockedExecuteBacktest.mockReset();
    useBacktestStore.getState().reset();
  });

  afterEach(() => {
    useBacktestStore.getState().reset();
  });

  it('submits async execution and stores job id/status on success', async () => {
    mockedExecuteBacktest.mockResolvedValue({
      job_id: 'job-123',
      status: JobStatus.Pending,
      message: 'submitted',
      result: null
    });

    const { result } = renderHook(() => useExecuteBacktest(), {
      wrapper: createWrapper()
    });

    act(() => {
      result.current.mutate(createPayload());
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    expect(mockedExecuteBacktest).toHaveBeenCalledWith(
      expect.objectContaining({
        code: expect.stringContaining('run_backtest'),
        async_mode: true
      })
    );

    const store = useBacktestStore.getState();
    expect(store.jobId).toBe('job-123');
    expect(store.jobStatus).toBe(JobStatus.Pending);
    expect(store.jobStartedAt).not.toBeNull();
  });

  it('surfaces execute API errors for non-Docker errors without retrying', async () => {
    mockedExecuteBacktest.mockRejectedValue(new Error('Execution failed'));

    const { result } = renderHook(() => useExecuteBacktest(), {
      wrapper: createWrapper()
    });

    act(() => {
      result.current.mutate(createPayload());
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe('Execution failed');
    expect(mockedExecuteBacktest).toHaveBeenCalledTimes(1);
    expect(result.current.isRetrying).toBe(false);
  });

  it(
    'retries once on Docker error and succeeds on the second attempt',
    async () => {
      const dockerError = new ApiError(
        'DockerError: container failed to start',
        'HTTP_ERROR',
        500,
        undefined,
        BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE
      );
      const successResponse = {
        job_id: 'job-retry-ok',
        status: JobStatus.Pending as const,
        message: 'submitted',
        result: null
      };

      mockedExecuteBacktest
        .mockRejectedValueOnce(dockerError)
        .mockResolvedValueOnce(successResponse);

      const { result } = renderHook(() => useExecuteBacktest(), {
        wrapper: createWrapper()
      });

      act(() => {
        result.current.mutate(createPayload());
      });

      // The retry includes a 2s delay, so we wait for the full mutation to settle.
      await waitFor(
        () => {
          expect(result.current.isSuccess).toBe(true);
        },
        { timeout: 10_000 }
      );

      expect(mockedExecuteBacktest).toHaveBeenCalledTimes(2);
      expect(result.current.isRetrying).toBe(false);

      const store = useBacktestStore.getState();
      expect(store.jobId).toBe('job-retry-ok');
      expect(store.jobStatus).toBe(JobStatus.Pending);
    },
    15_000
  );

  it(
    'surfaces error when Docker retry also fails',
    async () => {
      const dockerError = new ApiError(
        'Docker image pull failed',
        'HTTP_ERROR',
        500,
        undefined,
        BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE
      );
      const secondError = new ApiError(
        'Docker image still unavailable',
        'HTTP_ERROR',
        500,
        undefined,
        BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE
      );

      mockedExecuteBacktest
        .mockRejectedValueOnce(dockerError)
        .mockRejectedValueOnce(secondError);

      const { result } = renderHook(() => useExecuteBacktest(), {
        wrapper: createWrapper()
      });

      act(() => {
        result.current.mutate(createPayload());
      });

      await waitFor(
        () => {
          expect(result.current.isError).toBe(true);
        },
        { timeout: 10_000 }
      );

      expect(mockedExecuteBacktest).toHaveBeenCalledTimes(2);
      expect(result.current.error?.message).toBe('Docker image still unavailable');
      expect(result.current.isRetrying).toBe(false);
    },
    15_000
  );

  it('does not retry for non-Docker errors', async () => {
    mockedExecuteBacktest.mockRejectedValue(new Error('SyntaxError at line 12'));

    const { result } = renderHook(() => useExecuteBacktest(), {
      wrapper: createWrapper()
    });

    act(() => {
      result.current.mutate(createPayload());
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(mockedExecuteBacktest).toHaveBeenCalledTimes(1);
    expect(result.current.isRetrying).toBe(false);
  });

  it('exposes isRetrying as false initially', () => {
    const { result } = renderHook(() => useExecuteBacktest(), {
      wrapper: createWrapper()
    });

    expect(result.current.isRetrying).toBe(false);
  });
});
