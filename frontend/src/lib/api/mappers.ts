import { formatCurrency, formatDate, formatNumber, formatPercentage } from '@/lib/utils';
import {
  BacktestResultResponse,
  ChartDataPoint,
  DrawdownData,
  EquityCurveData,
  ExecutionResult,
  JobStatus,
  JobStatusResponse,
  PerformanceMetrics
} from '@/types';

export type UiStatusTone = 'neutral' | 'info' | 'success' | 'error';

export interface JobStatusUi {
  status: JobStatus | null;
  label: string;
  detail: string;
  tone: UiStatusTone;
  isTerminal: boolean;
}

export interface MetricsCardsViewModel {
  metrics: PerformanceMetrics | null;
  strategyCurve: ChartDataPoint[] | null;
  benchmarkCurve: ChartDataPoint[] | null;
  periodLabel: string | null;
  summary: {
    totalReturn: string;
    cagr: string;
    maxDrawdown: string;
    sharpeRatio: string;
    winRate: string;
  } | null;
}

export interface TradeRowViewModel {
  id: string;
  symbol: string;
  action: string;
  entryDateLabel: string;
  exitDateLabel: string;
  profitLabel: string;
  profitPctLabel: string;
  raw: Record<string, unknown>;
}

export interface ResultChartsViewModel {
  equityCurve: EquityCurveData | null;
  drawdown: DrawdownData | null;
  trades: Array<Record<string, unknown>>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function toFiniteNumber(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function toStringValue(value: unknown): string | null {
  if (typeof value === 'string' && value.trim().length > 0) {
    return value.trim();
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  return null;
}

function parseChartSeries(value: unknown): ChartDataPoint[] {
  if (!Array.isArray(value)) {
    return [];
  }

  return value
    .map((entry) => {
      if (!isRecord(entry)) {
        return null;
      }

      const date = toStringValue(entry.date);
      const pointValue = toFiniteNumber(entry.value);
      if (!date || pointValue === null) {
        return null;
      }

      return {
        date,
        value: pointValue
      };
    })
    .filter((point): point is ChartDataPoint => Boolean(point));
}

function parseEquityCurveFromData(data: Record<string, unknown>): EquityCurveData | null {
  const equityCurveValue = data.equity_curve;

  if (isRecord(equityCurveValue)) {
    const strategy = parseChartSeries(equityCurveValue.strategy);
    if (strategy.length === 0) {
      return null;
    }

    const benchmark = parseChartSeries(equityCurveValue.benchmark);

    return {
      strategy,
      benchmark: benchmark.length > 0 ? benchmark : null,
      log_scale: Boolean(equityCurveValue.log_scale)
    };
  }

  const strategy = parseChartSeries(data.equity_series ?? data.strategy);
  if (strategy.length === 0) {
    return null;
  }

  const benchmark = parseChartSeries(data.benchmark);

  return {
    strategy,
    benchmark: benchmark.length > 0 ? benchmark : null,
    log_scale: false
  };
}

function parseDrawdownFromData(data: Record<string, unknown>): DrawdownData | null {
  const drawdownValue = data.drawdown;

  if (isRecord(drawdownValue)) {
    const points = parseChartSeries(drawdownValue.data ?? drawdownValue.series);
    return points.length > 0 ? { data: points } : null;
  }

  const points = parseChartSeries(data.drawdown_series ?? drawdownValue);
  return points.length > 0 ? { data: points } : null;
}

function parseTradesFromData(data: Record<string, unknown>): Array<Record<string, unknown>> {
  const candidate = data.trades ?? data.trade_log;

  if (!Array.isArray(candidate)) {
    return [];
  }

  return candidate.filter((entry): entry is Record<string, unknown> => isRecord(entry));
}

export function mapJobStatusToUiStatus(
  statusInput: JobStatusResponse | JobStatus | null | undefined
): JobStatusUi {
  const status =
    typeof statusInput === 'string'
      ? statusInput
      : statusInput && typeof statusInput.status === 'string'
        ? statusInput.status
        : null;

  if (status === JobStatus.Pending) {
    return {
      status,
      label: 'Backtest pending',
      detail: 'Job is queued and waiting for execution.',
      tone: 'info',
      isTerminal: false
    };
  }

  if (status === JobStatus.Running) {
    return {
      status,
      label: 'Backtest running',
      detail: 'Execution in progress.',
      tone: 'info',
      isTerminal: false
    };
  }

  if (status === JobStatus.Completed) {
    return {
      status,
      label: 'Backtest completed',
      detail: 'Execution completed successfully.',
      tone: 'success',
      isTerminal: true
    };
  }

  if (status === JobStatus.Failed) {
    return {
      status,
      label: 'Backtest failed',
      detail: 'Execution failed.',
      tone: 'error',
      isTerminal: true
    };
  }

  if (status === JobStatus.Cancelled) {
    return {
      status,
      label: 'Backtest cancelled',
      detail: 'Execution was cancelled.',
      tone: 'neutral',
      isTerminal: true
    };
  }

  if (status === JobStatus.Timeout) {
    return {
      status,
      label: 'Backtest timed out',
      detail: 'Execution exceeded the timeout.',
      tone: 'error',
      isTerminal: true
    };
  }

  return {
    status: null,
    label: 'Backtest not started',
    detail: 'No execution status available.',
    tone: 'neutral',
    isTerminal: false
  };
}

export function mapBacktestResultToMetricsProps(
  result: BacktestResultResponse | null | undefined
): MetricsCardsViewModel {
  if (!result) {
    return {
      metrics: null,
      strategyCurve: null,
      benchmarkCurve: null,
      periodLabel: null,
      summary: null
    };
  }

  const strategyCurve = result.equity_curve.strategy.length > 0 ? result.equity_curve.strategy : null;
  const benchmarkCurve =
    result.equity_curve.benchmark && result.equity_curve.benchmark.length > 0
      ? result.equity_curve.benchmark
      : null;
  const firstDate = strategyCurve?.[0]?.date ?? null;
  const lastDate = strategyCurve?.[strategyCurve.length - 1]?.date ?? null;
  const periodLabel = firstDate && lastDate ? `${formatDate(firstDate)} - ${formatDate(lastDate)}` : null;

  return {
    metrics: result.metrics,
    strategyCurve,
    benchmarkCurve,
    periodLabel,
    summary: {
      totalReturn: formatPercentage(result.metrics.total_return, { maximumFractionDigits: 2 }),
      cagr: formatPercentage(result.metrics.cagr, { maximumFractionDigits: 2 }),
      maxDrawdown: `-${formatPercentage(Math.abs(result.metrics.max_drawdown), { maximumFractionDigits: 2 })}`,
      sharpeRatio: formatNumber(result.metrics.sharpe_ratio, { maximumFractionDigits: 2 }),
      winRate: formatPercentage(result.metrics.win_rate, {
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
      })
    }
  };
}

export function mapExecutionResultToCharts(
  executionResult: ExecutionResult | null | undefined
): ResultChartsViewModel {
  if (!executionResult || !isRecord(executionResult.data)) {
    return {
      equityCurve: null,
      drawdown: null,
      trades: []
    };
  }

  const data = executionResult.data;

  return {
    equityCurve: parseEquityCurveFromData(data),
    drawdown: parseDrawdownFromData(data),
    trades: parseTradesFromData(data)
  };
}

export function mapTradesToTableRows(
  trades: Array<Record<string, unknown>> | null | undefined
): TradeRowViewModel[] {
  return (trades ?? []).map((trade, index) => {
    const symbol = toStringValue(trade.symbol) ?? toStringValue(trade.ticker) ?? '-';
    const action = (toStringValue(trade.action) ?? toStringValue(trade.side) ?? '-').toUpperCase();
    const entryDateRaw =
      toStringValue(trade.entry_date) ?? toStringValue(trade.entryDate) ?? toStringValue(trade.date);
    const exitDateRaw =
      toStringValue(trade.exit_date) ?? toStringValue(trade.exitDate) ?? toStringValue(trade.date);
    const profit =
      toFiniteNumber(trade.profit) ?? toFiniteNumber(trade.profit_loss) ?? toFiniteNumber(trade.pnl);
    const profitPct =
      toFiniteNumber(trade.profit_pct) ??
      toFiniteNumber(trade.profitPercent) ??
      toFiniteNumber(trade.return_pct) ??
      toFiniteNumber(trade.pnl_pct);

    return {
      id: `${symbol}-${entryDateRaw ?? 'na'}-${index}`,
      symbol,
      action,
      entryDateLabel: entryDateRaw ? formatDate(entryDateRaw) : '-',
      exitDateLabel: exitDateRaw ? formatDate(exitDateRaw) : '-',
      profitLabel:
        profit === null
          ? '-'
          : formatCurrency(profit, { signed: true, minimumFractionDigits: 2, maximumFractionDigits: 2 }),
      profitPctLabel:
        profitPct === null
          ? '-'
          : formatPercentage(profitPct, { signed: true, minimumFractionDigits: 2, maximumFractionDigits: 2 }),
      raw: trade
    };
  });
}
