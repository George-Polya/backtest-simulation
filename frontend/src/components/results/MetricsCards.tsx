'use client';

import { useMemo } from 'react';
import { Alert, Card, LoadingSpinner } from '@/components/ui';
import { formatNumber, formatPercentage } from '@/lib/utils';
import { ChartDataPoint, PerformanceMetrics } from '@/types';

interface MetricsCardsProps {
  metrics: PerformanceMetrics | null;
  strategyCurve?: ChartDataPoint[] | null;
  benchmarkCurve?: ChartDataPoint[] | null;
  isLoading?: boolean;
}

interface PrimaryMetricCard {
  id: 'total_return' | 'cagr' | 'max_drawdown';
  label: string;
  value: number;
}

interface BenchmarkPrimaryMetrics {
  total_return: number;
  cagr: number;
  max_drawdown: number;
}

function calculateTotalReturn(series: ChartDataPoint[]): number {
  if (series.length < 2) {
    return 0;
  }

  const first = series[0]?.value ?? 0;
  const last = series[series.length - 1]?.value ?? 0;

  if (first <= 0) {
    return 0;
  }

  return ((last - first) / first) * 100;
}

function calculateCagr(series: ChartDataPoint[]): number {
  if (series.length < 2) {
    return 0;
  }

  const first = series[0]?.value ?? 0;
  const last = series[series.length - 1]?.value ?? 0;

  if (first <= 0 || last <= 0) {
    return 0;
  }

  const firstDate = new Date(series[0]?.date ?? '');
  const lastDate = new Date(series[series.length - 1]?.date ?? '');
  const hasValidDates = !Number.isNaN(firstDate.getTime()) && !Number.isNaN(lastDate.getTime());
  const days = hasValidDates
    ? Math.max(1, (lastDate.getTime() - firstDate.getTime()) / (1000 * 60 * 60 * 24))
    : Math.max(1, series.length - 1);
  const years = days / 365.25;

  if (years <= 0) {
    return 0;
  }

  return (Math.pow(last / first, 1 / years) - 1) * 100;
}

function calculateMaxDrawdown(series: ChartDataPoint[]): number {
  if (series.length < 2) {
    return 0;
  }

  let runningMax = Number.NEGATIVE_INFINITY;
  let maxDrawdown = 0;

  for (const point of series) {
    runningMax = Math.max(runningMax, point.value);
    if (runningMax <= 0) {
      continue;
    }

    const drawdown = ((point.value - runningMax) / runningMax) * 100;
    maxDrawdown = Math.min(maxDrawdown, drawdown);
  }

  return Math.abs(maxDrawdown);
}

function deriveBenchmarkMetrics(benchmarkCurve: ChartDataPoint[] | null | undefined): BenchmarkPrimaryMetrics | null {
  if (!benchmarkCurve || benchmarkCurve.length < 2) {
    return null;
  }

  return {
    total_return: calculateTotalReturn(benchmarkCurve),
    cagr: calculateCagr(benchmarkCurve),
    max_drawdown: calculateMaxDrawdown(benchmarkCurve)
  };
}

function getPrimaryValueLabel(metric: PrimaryMetricCard): string {
  if (metric.id === 'max_drawdown') {
    return `-${formatPercentage(Math.abs(metric.value))}`;
  }

  return formatPercentage(metric.value);
}

function isDeltaFavorable(metricId: PrimaryMetricCard['id'], delta: number): boolean {
  if (metricId === 'max_drawdown') {
    return delta < 0;
  }

  return delta > 0;
}

export function MetricsCards({
  metrics,
  strategyCurve,
  benchmarkCurve,
  isLoading = false
}: MetricsCardsProps) {
  const benchmarkMetrics = useMemo(
    () => deriveBenchmarkMetrics(benchmarkCurve),
    [benchmarkCurve]
  );
  const primaryMetrics: PrimaryMetricCard[] = useMemo(() => {
    if (!metrics) {
      return [];
    }

    return [
      { id: 'total_return', label: 'Total Return', value: metrics.total_return },
      { id: 'cagr', label: 'CAGR', value: metrics.cagr },
      { id: 'max_drawdown', label: 'Max Drawdown', value: metrics.max_drawdown }
    ];
  }, [metrics]);
  const hasBenchmarkComparison = Boolean(
    benchmarkMetrics && strategyCurve && strategyCurve.length > 1
  );

  return (
    <Card title="Performance Metrics">
      {isLoading ? (
        <div className="py-4">
          <LoadingSpinner centered size="md" />
          <p className="mt-2 text-center text-sm text-slate-600">Loading performance metrics...</p>
        </div>
      ) : null}

      {!isLoading && !metrics ? (
        <Alert title="Metrics unavailable" variant="info">
          Run a completed backtest to view performance metrics.
        </Alert>
      ) : null}

      {!isLoading && metrics ? (
        <div className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            {primaryMetrics.map((metric) => {
              const benchmarkValue = benchmarkMetrics?.[metric.id];
              const hasDelta = typeof benchmarkValue === 'number';
              const delta = hasDelta ? metric.value - benchmarkValue : null;
              const deltaClass =
                delta === null
                  ? 'text-slate-500'
                  : isDeltaFavorable(metric.id, delta)
                    ? 'text-emerald-700'
                    : 'text-red-700';

              return (
                <Card className="bg-slate-50/70" key={metric.id} title={metric.label}>
                  <p className="text-2xl font-semibold text-slate-900">{getPrimaryValueLabel(metric)}</p>
                  {hasDelta && delta !== null ? (
                    <p className={`mt-2 text-xs font-medium ${deltaClass}`}>
                      {formatPercentage(delta, { signed: true })} vs benchmark
                    </p>
                  ) : (
                    <p className="mt-2 text-xs text-slate-500">
                      {hasBenchmarkComparison
                        ? 'Benchmark comparison unavailable.'
                        : 'No benchmark series provided.'}
                    </p>
                  )}
                </Card>
              );
            })}
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <Card className="bg-white/90" title="Sharpe Ratio">
              <p className="text-xl font-semibold text-slate-900">{formatNumber(metrics.sharpe_ratio)}</p>
            </Card>
            <Card className="bg-white/90" title="Sortino Ratio">
              <p className="text-xl font-semibold text-slate-900">{formatNumber(metrics.sortino_ratio)}</p>
            </Card>
            <Card className="bg-white/90" title="Calmar Ratio">
              <p className="text-xl font-semibold text-slate-900">{formatNumber(metrics.calmar_ratio)}</p>
            </Card>
          </div>
        </div>
      ) : null}
    </Card>
  );
}
