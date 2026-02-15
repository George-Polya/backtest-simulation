import React from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { BacktestWorkspace } from '@/components/backtest';
import { useBacktestStore } from '@/stores';
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

describe('integration: generate flow', () => {
  beforeEach(() => {
    act(() => {
      useBacktestStore.getState().reset();
    });

    server.use(
      http.post(`${API_BASE_URL}/backtest/generate`, () =>
        HttpResponse.json({
          generated_code: {
            code: 'def run_backtest(params):\n    return {"status": "ok"}',
            strategy_summary: 'Integration summary.',
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
          generation_time_seconds: 1.1
        })
      )
    );
  });

  afterEach(() => {
    act(() => {
      useBacktestStore.getState().reset();
    });
  });

  it('validates form then generates code and metadata', async () => {
    renderWorkspace();

    fireEvent.change(screen.getByLabelText('Strategy'), { target: { value: 'short' } });
    fireEvent.click(screen.getByRole('button', { name: 'Save Configuration' }));

    await waitFor(() => {
      expect(screen.getByText('Please fix the following validation errors')).toBeInTheDocument();
    });

    expect(screen.getAllByText('Strategy must be at least 10 characters.').length).toBeGreaterThan(0);
    expect(screen.queryByText('Ready for code generation')).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Strategy'), {
      target: { value: 'Momentum strategy with monthly rebalance and risk controls.' }
    });
    fireEvent.click(screen.getByRole('button', { name: 'Save Configuration' }));

    await waitFor(() => {
      expect(screen.getByText('Ready for code generation')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: 'Generate Code' }));

    await waitFor(() => {
      expect(screen.getByDisplayValue(/def run_backtest/)).toBeInTheDocument();
    });

    expect(screen.getByText('Integration summary.')).toBeInTheDocument();
    expect(screen.getByText('openrouter / anthropic/claude-3.5-sonnet')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Execute Backtest' })).not.toBeDisabled();
  });
});
