'use client';

import { useEffect, useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { Alert, Card, LoadingSpinner, Toggle } from '@/components/ui';
import { formatCurrency, formatDate } from '@/lib/utils';
import { ChartDataPoint, EquityCurveData } from '@/types';

interface EquityChartProps {
  equityCurve: EquityCurveData | null;
  isLoading?: boolean;
}

interface EquityChartRow {
  date: string;
  strategy?: number;
  benchmark?: number;
}

function buildChartRows(strategy: ChartDataPoint[], benchmark: ChartDataPoint[] | null): EquityChartRow[] {
  const rowsByDate = new Map<string, EquityChartRow>();

  for (const point of strategy) {
    const row = rowsByDate.get(point.date) ?? { date: point.date };
    row.strategy = point.value;
    rowsByDate.set(point.date, row);
  }

  for (const point of benchmark ?? []) {
    const row = rowsByDate.get(point.date) ?? { date: point.date };
    row.benchmark = point.value;
    rowsByDate.set(point.date, row);
  }

  return Array.from(rowsByDate.values()).sort((a, b) => a.date.localeCompare(b.date));
}

export function EquityChart({ equityCurve, isLoading = false }: EquityChartProps) {
  const [isLogScale, setIsLogScale] = useState(false);

  useEffect(() => {
    setIsLogScale(Boolean(equityCurve?.log_scale));
  }, [equityCurve?.log_scale]);

  const chartRows = useMemo(
    () => buildChartRows(equityCurve?.strategy ?? [], equityCurve?.benchmark ?? null),
    [equityCurve?.benchmark, equityCurve?.strategy]
  );
  const hasBenchmark = Boolean(equityCurve?.benchmark && equityCurve.benchmark.length > 0);
  const hasEquityData = chartRows.length > 0;
  const allValues = useMemo(() => {
    const values: number[] = [];
    for (const row of chartRows) {
      if (typeof row.strategy === 'number') {
        values.push(row.strategy);
      }
      if (typeof row.benchmark === 'number') {
        values.push(row.benchmark);
      }
    }
    return values;
  }, [chartRows]);
  const canUseLogScale = allValues.every((value) => value > 0);

  useEffect(() => {
    if (!canUseLogScale && isLogScale) {
      setIsLogScale(false);
    }
  }, [canUseLogScale, isLogScale]);

  return (
    <Card title="Equity Curve">
      {isLoading ? (
        <div className="py-6">
          <LoadingSpinner centered size="md" />
          <p className="mt-2 text-center text-sm text-slate-600">Loading equity chart...</p>
        </div>
      ) : null}

      {!isLoading && !hasEquityData ? (
        <Alert title="Equity curve unavailable" variant="info">
          Equity curve data is not available for this result.
        </Alert>
      ) : null}

      {!isLoading && hasEquityData ? (
        <div className="space-y-3">
          <Toggle
            checked={isLogScale}
            className="max-w-xs"
            description={
              canUseLogScale
                ? 'Switch between linear and logarithmic y-axis.'
                : 'Log scale requires positive values. Linear scale is used.'
            }
            disabled={!canUseLogScale}
            label="Log Scale"
            onChange={(event) => setIsLogScale(event.target.checked)}
          />

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={chartRows}
                margin={{ top: 12, right: 12, left: 0, bottom: 12 }}
              >
                <CartesianGrid stroke="#e2e8f0" strokeDasharray="4 4" />
                <XAxis dataKey="date" tickFormatter={formatDate} />
                <YAxis
                  domain={isLogScale ? ['dataMin', 'dataMax'] : ['auto', 'auto']}
                  scale={isLogScale ? 'log' : 'auto'}
                  tickFormatter={(value: number) =>
                    formatCurrency(value, { minimumFractionDigits: 0, maximumFractionDigits: 0 })
                  }
                  type="number"
                />
                <Tooltip
                  formatter={(value: number | string, name: string) => [
                    typeof value === 'number'
                      ? formatCurrency(value, { minimumFractionDigits: 2, maximumFractionDigits: 2 })
                      : value,
                    name === 'strategy' ? 'Strategy' : 'Benchmark'
                  ]}
                  labelFormatter={(label) => formatDate(String(label))}
                />
                <Legend />
                <Line
                  dataKey="strategy"
                  dot={false}
                  name="Strategy"
                  stroke="#1d7de0"
                  strokeWidth={2}
                  type="monotone"
                />
                {hasBenchmark ? (
                  <Line
                    dataKey="benchmark"
                    dot={false}
                    name="Benchmark"
                    stroke="#64748b"
                    strokeWidth={1.8}
                    type="monotone"
                  />
                ) : null}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : null}
    </Card>
  );
}
