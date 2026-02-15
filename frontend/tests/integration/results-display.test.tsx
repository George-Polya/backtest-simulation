import React, { ReactNode } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { act, render, screen } from '@testing-library/react';
import { BacktestWorkspace } from '@/components/backtest';
import { useBacktestStore } from '@/stores';
import { BacktestResultResponse, JobStatus } from '@/types';

vi.mock('@/hooks', async () => {
  const actual = await vi.importActual<typeof import('@/hooks')>('@/hooks');
  return {
    ...actual,
    useJobPolling: () => ({
      status: null,
      data: undefined,
      isLoading: false,
      isPolling: false,
      error: null,
      isJobNotFound: false
    })
  };
});

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

vi.mock('recharts', () => {
  const MockWrapper = ({
    children,
    as = 'div',
    testId
  }: {
    children?: ReactNode;
    as?: 'div' | 'svg';
    testId: string;
  }) =>
    as === 'svg' ? (
      <svg data-testid={testId}>{children}</svg>
    ) : (
      <div data-testid={testId}>{children}</div>
    );

  return {
    ResponsiveContainer: ({ children }: { children?: ReactNode }) => (
      <MockWrapper testId="responsive-container">{children}</MockWrapper>
    ),
    LineChart: ({ children }: { children?: ReactNode }) => (
      <MockWrapper as="svg" testId="line-chart">
        {children}
      </MockWrapper>
    ),
    AreaChart: ({ children }: { children?: ReactNode }) => (
      <MockWrapper as="svg" testId="area-chart">
        {children}
      </MockWrapper>
    ),
    CartesianGrid: () => <g data-testid="cartesian-grid" />,
    XAxis: () => <g data-testid="x-axis" />,
    YAxis: () => <g data-testid="y-axis" />,
    Tooltip: () => <g data-testid="tooltip" />,
    Legend: () => <g data-testid="legend" />,
    Line: ({ dataKey }: { dataKey?: string }) => <g data-testid={`line-${String(dataKey)}`} />,
    Area: ({ dataKey }: { dataKey?: string }) => <g data-testid={`area-${String(dataKey)}`} />
  };
});

function createResult(overrides: Partial<BacktestResultResponse> = {}): BacktestResultResponse {
  return {
    job_id: 'job-results',
    status: JobStatus.Completed,
    metrics: {
      total_return: 18.2,
      cagr: 9.4,
      max_drawdown: 7.1,
      sharpe_ratio: 1.22,
      sortino_ratio: 1.54,
      calmar_ratio: 0.86,
      volatility: 12.3,
      total_trades: 4,
      winning_trades: 3,
      losing_trades: 1,
      win_rate: 75
    },
    equity_curve: {
      strategy: [
        { date: '2024-01-01', value: 10000 },
        { date: '2024-12-31', value: 11820 }
      ],
      benchmark: [
        { date: '2024-01-01', value: 10000 },
        { date: '2024-12-31', value: 11100 }
      ],
      log_scale: false
    },
    drawdown: {
      data: [{ date: '2024-06-01', value: -2.4 }]
    },
    monthly_heatmap: {
      years: [2024],
      months: ['Jan'],
      returns: [[1.1]]
    },
    trades: [{ date: '2024-04-01', symbol: 'AAPL', action: 'BUY', profit: 0 }],
    logs: 'done',
    ...overrides
  };
}

function renderWorkspaceWithResults(results: BacktestResultResponse) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false }
    }
  });

  act(() => {
    const state = useBacktestStore.getState();
    state.reset();
    state.setJobState(results.job_id, JobStatus.Completed);
    state.setResults(results);
    state.setUiToggle('isCodeEditorOpen', false);
    state.setUiToggle('isResultsOpen', true);
    state.setSelectedTab('results');
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <BacktestWorkspace />
    </QueryClientProvider>
  );
}

describe('integration: results display', () => {
  afterEach(() => {
    act(() => {
      useBacktestStore.getState().reset();
    });
  });

  it('renders metrics/charts/trades and enables csv exports when results exist', () => {
    renderWorkspaceWithResults(createResult());

    expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    expect(screen.getByText('Equity Curve')).toBeInTheDocument();
    expect(screen.getByText('Drawdown')).toBeInTheDocument();
    expect(screen.getByText('Monthly Returns Heatmap')).toBeInTheDocument();
    expect(screen.getByText('Trade History')).toBeInTheDocument();

    expect(screen.getByText('Total Return')).toBeInTheDocument();
    expect(screen.getByText('1 trade(s)')).toBeInTheDocument();

    expect(screen.getByRole('button', { name: 'Export Equity' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Export Drawdown' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Export Monthly' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Export Metrics' })).not.toBeDisabled();
    expect(screen.getByRole('button', { name: 'Export Trades' })).not.toBeDisabled();
  });

  it('shows graceful empty states for partial result payloads', () => {
    renderWorkspaceWithResults(
      createResult({
        equity_curve: {
          strategy: [],
          benchmark: null,
          log_scale: false
        },
        drawdown: { data: [] },
        monthly_heatmap: {
          years: [],
          months: [],
          returns: []
        },
        trades: []
      })
    );

    expect(screen.getByText('Equity curve unavailable')).toBeInTheDocument();
    expect(screen.getByText('Drawdown unavailable')).toBeInTheDocument();
    expect(screen.getByText('Monthly returns unavailable')).toBeInTheDocument();
    expect(screen.getByText('No trades available')).toBeInTheDocument();
  });
});
