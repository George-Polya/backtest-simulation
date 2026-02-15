import React, { PropsWithChildren } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, renderHook, waitFor } from '@testing-library/react';
import { useGenerateCode } from '@/hooks';
import { generateCode } from '@/lib/api';
import { useBacktestStore } from '@/stores';
import { BacktestRequest, ContributionFrequency } from '@/types';

vi.mock('@/lib/api', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/api')>();
  return {
    ...actual,
    generateCode: vi.fn()
  };
});

const mockedGenerateCode = vi.mocked(generateCode);

function createRequest(): BacktestRequest {
  return {
    strategy: 'Use momentum with risk controls and monthly rebalancing.',
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

describe('useGenerateCode', () => {
  const originalGenerationTimeout = process.env.NEXT_PUBLIC_CODE_GENERATION_TIMEOUT_MS;

  beforeEach(() => {
    mockedGenerateCode.mockReset();
    delete process.env.NEXT_PUBLIC_CODE_GENERATION_TIMEOUT_MS;
    useBacktestStore.getState().reset();
  });

  afterEach(() => {
    vi.useRealTimers();
    if (originalGenerationTimeout) {
      process.env.NEXT_PUBLIC_CODE_GENERATION_TIMEOUT_MS = originalGenerationTimeout;
    } else {
      delete process.env.NEXT_PUBLIC_CODE_GENERATION_TIMEOUT_MS;
    }
    useBacktestStore.getState().reset();
  });

  it('updates store with generated code and metadata on success', async () => {
    mockedGenerateCode.mockResolvedValue({
      generated_code: {
        code: 'def run_backtest(params):\n    return {"status": "ok"}',
        strategy_summary: 'Momentum strategy for SPY.',
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
      generation_time_seconds: 1.2
    });

    const { result } = renderHook(() => useGenerateCode(), {
      wrapper: createWrapper()
    });

    act(() => {
      result.current.mutate(createRequest());
    });

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true);
    });

    const store = useBacktestStore.getState();
    expect(store.generatedCode).toContain('run_backtest');
    expect(store.generationMetadata?.model_info.model_id).toContain('claude');
    expect(store.generationMetadata?.tickers_found).toEqual(['SPY']);
  });

  it('returns timeout error after 120 seconds by default', async () => {
    const setTimeoutSpy = vi
      .spyOn(global, 'setTimeout')
      .mockImplementation(((handler: TimerHandler) => {
        if (typeof handler === 'function') {
          handler();
        }
        return 1 as unknown as ReturnType<typeof setTimeout>;
      }) as unknown as typeof setTimeout);

    const clearTimeoutSpy = vi
      .spyOn(global, 'clearTimeout')
      .mockImplementation(() => undefined);

    mockedGenerateCode.mockRejectedValue(new Error('Request was aborted'));

    const { result } = renderHook(() => useGenerateCode(), {
      wrapper: createWrapper()
    });

    await act(async () => {
      await expect(result.current.mutateAsync(createRequest())).rejects.toThrow(
        'timed out after 120 seconds'
      );
    });

    expect(result.current.error?.message).toContain('timed out after 120 seconds');
    expect(setTimeoutSpy).toHaveBeenCalled();

    setTimeoutSpy.mockRestore();
    clearTimeoutSpy.mockRestore();
  });

  it('surfaces API errors', async () => {
    mockedGenerateCode.mockRejectedValue(new Error('Invalid strategy format'));

    const { result } = renderHook(() => useGenerateCode(), {
      wrapper: createWrapper()
    });

    act(() => {
      result.current.mutate(createRequest());
    });

    await waitFor(() => {
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.error?.message).toBe('Invalid strategy format');
  });
});
