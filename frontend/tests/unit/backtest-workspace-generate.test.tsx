import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { BacktestWorkspace } from '@/components/backtest';
import { useBacktestStore } from '@/stores';
import { BacktestRequest, ContributionFrequency } from '@/types';
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
    strategy: 'Use momentum with a volatility filter and monthly rebalance.',
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

describe('BacktestWorkspace generate flow', () => {
  beforeEach(() => {
    act(() => {
      useBacktestStore.getState().reset();
      useBacktestStore.getState().setRequestConfig(createRequestConfig());
    });
    server.use(
      http.post(`${API_BASE_URL}/backtest/generate`, () =>
        HttpResponse.json({
          generated_code: {
            code: 'def run_backtest(params):\n    return {"status": "ok"}',
            strategy_summary: 'Momentum strategy summary.',
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
          generation_time_seconds: 1.3
        })
      )
    );
  });

  afterEach(() => {
    act(() => {
      useBacktestStore.getState().reset();
    });
  });

  it('disables execute when generated code is not available', () => {
    renderWorkspace();

    const executeButton = screen.getByRole('button', { name: 'Execute Backtest' });
    expect(executeButton).toBeDisabled();
  });

  it('generates code and displays metadata', async () => {
    renderWorkspace();

    fireEvent.click(screen.getByRole('button', { name: 'Generate Code' }));

    await waitFor(() => {
      expect(screen.getByDisplayValue(/def run_backtest/)).toBeInTheDocument();
    });

    expect(screen.getByText('Momentum strategy summary.')).toBeInTheDocument();
    expect(screen.getByText('openrouter / anthropic/claude-3.5-sonnet')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Execute Backtest' })).not.toBeDisabled();
  });

  it('shows loading state while generation is in progress', async () => {
    server.use(
      http.post(`${API_BASE_URL}/backtest/generate`, async () => {
        await new Promise((resolve) => setTimeout(resolve, 120));
        return HttpResponse.json({
          generated_code: {
            code: 'def run_backtest(params):\n    return {"status": "ok"}',
            strategy_summary: 'Delayed summary.',
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
          generation_time_seconds: 1.8
        });
      })
    );

    renderWorkspace();
    fireEvent.click(screen.getByRole('button', { name: 'Generate Code' }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Generating...' })).toBeInTheDocument();
    });
  });

  it('shows error message when generation fails', async () => {
    server.use(
      http.post(`${API_BASE_URL}/backtest/generate`, () =>
        HttpResponse.json({ detail: 'Generation failed' }, { status: 500 })
      )
    );

    renderWorkspace();
    fireEvent.click(screen.getByRole('button', { name: 'Generate Code' }));

    await waitFor(() => {
      expect(screen.getByText('Code generation failed')).toBeInTheDocument();
    });

    expect(screen.getByText('Generation failed')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Execute Backtest' })).toBeDisabled();
  });
});
