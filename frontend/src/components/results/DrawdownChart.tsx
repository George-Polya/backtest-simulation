'use client';

import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import { Alert, Card, LoadingSpinner } from '@/components/ui';
import { formatDate, formatPercentage } from '@/lib/utils';
import { DrawdownData } from '@/types';

interface DrawdownChartProps {
  drawdown: DrawdownData | null;
  isLoading?: boolean;
}

export function DrawdownChart({ drawdown, isLoading = false }: DrawdownChartProps) {
  const data = drawdown?.data ?? [];

  return (
    <Card title="Drawdown">
      {isLoading ? (
        <div className="py-6">
          <LoadingSpinner centered size="md" />
          <p className="mt-2 text-center text-sm text-slate-600">Loading drawdown chart...</p>
        </div>
      ) : null}

      {!isLoading && data.length === 0 ? (
        <Alert title="Drawdown unavailable" variant="info">
          Drawdown data is not available for this result.
        </Alert>
      ) : null}

      {!isLoading && data.length > 0 ? (
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 12, right: 12, left: 0, bottom: 12 }}>
              <defs>
                <linearGradient id="drawdownFill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#1d7de0" stopOpacity={0.45} />
                  <stop offset="100%" stopColor="#1d7de0" stopOpacity={0.05} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#e2e8f0" strokeDasharray="4 4" />
              <XAxis dataKey="date" tickFormatter={formatDate} />
              <YAxis
                tickFormatter={(value: number) => formatPercentage(value, { maximumFractionDigits: 1 })}
              />
              <Tooltip
                formatter={(value: number | string) => [
                  typeof value === 'number'
                    ? formatPercentage(value, { maximumFractionDigits: 2 })
                    : value,
                  'Drawdown'
                ]}
                labelFormatter={(label) => formatDate(String(label))}
              />
              <Area
                dataKey="value"
                fill="url(#drawdownFill)"
                stroke="#1d7de0"
                strokeWidth={1.5}
                type="monotone"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      ) : null}
    </Card>
  );
}
