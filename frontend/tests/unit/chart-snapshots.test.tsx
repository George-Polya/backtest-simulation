import { ReactNode } from 'react';
import { render } from '@testing-library/react';
import { DrawdownChart, EquityChart, MonthlyHeatmap } from '@/components/results';

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

describe('chart snapshots', () => {
  it('matches equity chart snapshot', () => {
    const { asFragment } = render(
      <EquityChart
        equityCurve={{
          strategy: [
            { date: '2024-01-01', value: 10000 },
            { date: '2024-12-31', value: 12340 }
          ],
          benchmark: [
            { date: '2024-01-01', value: 10000 },
            { date: '2024-12-31', value: 11800 }
          ],
          log_scale: false
        }}
      />
    );

    expect(asFragment()).toMatchSnapshot();
  });

  it('matches drawdown chart snapshot', () => {
    const { asFragment } = render(
      <DrawdownChart
        drawdown={{
          data: [
            { date: '2024-01-01', value: 0 },
            { date: '2024-04-01', value: -4.2 },
            { date: '2024-08-01', value: -9.8 }
          ]
        }}
      />
    );

    expect(asFragment()).toMatchSnapshot();
  });

  it('matches monthly heatmap snapshot', () => {
    const { asFragment } = render(
      <MonthlyHeatmap
        monthlyHeatmap={{
          years: [2024],
          months: ['Jan', 'Feb', 'Mar'],
          returns: [[1.2, -2.1, null]]
        }}
      />
    );

    expect(asFragment()).toMatchSnapshot();
  });
});
