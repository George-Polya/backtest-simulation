import { fireEvent, render, screen, within } from '@testing-library/react';
import { ReactNode } from 'react';
import {
  DrawdownChart,
  EquityChart,
  MetricsCards,
  MonthlyHeatmap,
  TradeTable
} from '@/components/results';
import {
  DrawdownData,
  EquityCurveData,
  JobStatus,
  MonthlyHeatmapData,
  PerformanceMetrics
} from '@/types';

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

const metrics: PerformanceMetrics = {
  total_return: 22.34,
  cagr: 10.51,
  max_drawdown: 15.2,
  sharpe_ratio: 1.34,
  sortino_ratio: 1.89,
  calmar_ratio: 0.69,
  volatility: 18.2,
  total_trades: 8,
  winning_trades: 5,
  losing_trades: 3,
  win_rate: 62.5
};

const equityCurve: EquityCurveData = {
  strategy: [
    { date: '2024-01-01', value: 10000 },
    { date: '2024-12-31', value: 12340 }
  ],
  benchmark: [
    { date: '2024-01-01', value: 10000 },
    { date: '2024-12-31', value: 11800 }
  ],
  log_scale: false
};

const drawdown: DrawdownData = {
  data: [
    { date: '2024-01-01', value: 0 },
    { date: '2024-04-01', value: -4.2 },
    { date: '2024-08-01', value: -9.8 }
  ]
};

const monthlyHeatmap: MonthlyHeatmapData = {
  years: [2024],
  months: ['Jan', 'Feb', 'Mar'],
  returns: [[1.2, -2.1, null]]
};

describe('MetricsCards', () => {
  it('renders primary and secondary metrics with benchmark delta', () => {
    render(
      <MetricsCards
        benchmarkCurve={equityCurve.benchmark}
        metrics={metrics}
        strategyCurve={equityCurve.strategy}
      />
    );

    expect(screen.getByText('Total Return')).toBeInTheDocument();
    expect(screen.getByText('CAGR')).toBeInTheDocument();
    expect(screen.getByText('Max Drawdown')).toBeInTheDocument();
    expect(screen.getByText('Sharpe Ratio')).toBeInTheDocument();
    expect(screen.getAllByText(/vs benchmark/i).length).toBeGreaterThan(0);
  });

  it('renders loading and empty states', () => {
    const { rerender } = render(<MetricsCards metrics={null} isLoading />);
    expect(screen.getByText('Loading performance metrics...')).toBeInTheDocument();

    rerender(<MetricsCards metrics={null} isLoading={false} />);
    expect(screen.getByText('Metrics unavailable')).toBeInTheDocument();
  });
});

describe('EquityChart', () => {
  it('renders chart with log scale toggle', () => {
    render(<EquityChart equityCurve={equityCurve} />);

    expect(screen.getByText('Log Scale')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    expect(screen.getByTestId('line-strategy')).toBeInTheDocument();
    expect(screen.getByTestId('line-benchmark')).toBeInTheDocument();
  });

  it('disables log scale when data has non-positive values', () => {
    render(
      <EquityChart
        equityCurve={{
          ...equityCurve,
          strategy: [
            { date: '2024-01-01', value: 0 },
            { date: '2024-12-31', value: 10000 }
          ]
        }}
      />
    );

    expect(screen.getByRole('switch')).toBeDisabled();
  });

  it('renders empty state when data is missing', () => {
    render(<EquityChart equityCurve={null} />);
    expect(screen.getByText('Equity curve unavailable')).toBeInTheDocument();
  });
});

describe('DrawdownChart', () => {
  it('renders chart data', () => {
    render(<DrawdownChart drawdown={drawdown} />);
    expect(screen.getByTestId('area-chart')).toBeInTheDocument();
    expect(screen.getByTestId('area-value')).toBeInTheDocument();
  });

  it('renders loading and empty states', () => {
    const { rerender } = render(<DrawdownChart drawdown={null} isLoading />);
    expect(screen.getByText('Loading drawdown chart...')).toBeInTheDocument();

    rerender(<DrawdownChart drawdown={null} isLoading={false} />);
    expect(screen.getByText('Drawdown unavailable')).toBeInTheDocument();
  });
});

describe('MonthlyHeatmap', () => {
  it('renders year/month grid and formatted values', () => {
    render(<MonthlyHeatmap monthlyHeatmap={monthlyHeatmap} />);

    expect(screen.getByText('2024')).toBeInTheDocument();
    expect(screen.getByText('Jan')).toBeInTheDocument();
    expect(screen.getByText('1.2%')).toBeInTheDocument();
    expect(screen.getByText('-2.1%')).toBeInTheDocument();
    expect(screen.getByText('-')).toBeInTheDocument();
  });

  it('renders empty state', () => {
    render(<MonthlyHeatmap monthlyHeatmap={null} />);
    expect(screen.getByText('Monthly returns unavailable')).toBeInTheDocument();
  });
});

describe('TradeTable', () => {
  const trades = [
    {
      date: '2024-05-01',
      symbol: 'AAPL',
      action: 'BUY',
      profit: 0,
      profit_pct: 0
    },
    {
      date: '2024-06-01',
      symbol: 'SPY',
      action: 'SELL',
      profit: 240.12,
      profit_pct: 2.4
    },
    {
      date: '2024-04-01',
      symbol: 'QQQ',
      action: 'SELL',
      profit: -120.5,
      profit_pct: -1.4
    }
  ];

  it('renders sortable trade table and action/profit formatting', () => {
    render(<TradeTable trades={trades} />);

    expect(screen.getByText('3 trade(s)')).toBeInTheDocument();
    expect(screen.getByText('Buy')).toBeInTheDocument();
    expect(screen.getAllByText('Sell').length).toBeGreaterThan(0);
    expect(screen.getByText('+$240.12')).toBeInTheDocument();
    expect(screen.getByText('-$120.50')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /profit\/loss/i }));
    const rows = screen.getAllByRole('row');
    const firstBodyRow = rows[1];

    expect(within(firstBodyRow).getByText('SPY')).toBeInTheDocument();
  });

  it('supports collapse/expand and empty state', () => {
    const { rerender } = render(<TradeTable trades={trades} />);

    fireEvent.click(screen.getByRole('button', { name: 'Collapse' }));
    expect(screen.queryByRole('table')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(screen.getByRole('table')).toBeInTheDocument();

    rerender(<TradeTable trades={[]} />);
    expect(screen.getByText('No trades available')).toBeInTheDocument();
  });
});

describe('Task7 result payload compatibility', () => {
  it('keeps JobStatus type contract untouched', () => {
    expect(JobStatus.Completed).toBe('completed');
  });
});
