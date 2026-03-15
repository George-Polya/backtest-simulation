import { BacktestResultResponse, DrawdownData, EquityCurveData, MonthlyHeatmapData, PerformanceMetrics } from '@/types';

export const CSV_BOM = '\uFEFF';

function formatDateToken(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${year}${month}${day}`;
}

function normalizeLineEndings(value: string): string {
  return value.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

export function escapeCSVValue(value: unknown): string {
  if (value === null || value === undefined) {
    return '';
  }

  if (typeof value === 'number') {
    return Number.isFinite(value) ? String(value) : '';
  }

  if (typeof value === 'boolean') {
    return value ? 'true' : 'false';
  }

  const normalized = normalizeLineEndings(String(value));
  const escaped = normalized.replace(/"/g, '""');
  const needsWrapping = /[",\n]/.test(escaped);

  return needsWrapping ? `"${escaped}"` : escaped;
}

export function generateFilename(
  type: 'equity' | 'drawdown' | 'monthly' | 'metrics' | 'trades',
  jobId: string,
  date = new Date()
): string {
  return `backtest_${type}_${jobId}_${formatDateToken(date)}.csv`;
}

function toCSVString(rows: Array<Array<unknown>>, includeBom: boolean): string {
  const body = rows
    .map((row) => row.map((cell) => escapeCSVValue(cell)).join(','))
    .join('\n');

  return includeBom ? `${CSV_BOM}${body}` : body;
}

function collectSeriesDates(equityCurve: EquityCurveData): string[] {
  const dates = new Set<string>();

  for (const point of equityCurve.strategy) {
    dates.add(point.date);
  }

  for (const point of equityCurve.benchmark ?? []) {
    dates.add(point.date);
  }

  return Array.from(dates).sort((a, b) => a.localeCompare(b));
}

export function generateEquityCSV(
  equityCurve: EquityCurveData,
  options: { includeBom?: boolean } = {}
): string {
  const includeBom = options.includeBom ?? true;
  const strategyByDate = new Map(equityCurve.strategy.map((point) => [point.date, point.value]));
  const benchmarkByDate = new Map((equityCurve.benchmark ?? []).map((point) => [point.date, point.value]));
  const hasBenchmark = (equityCurve.benchmark ?? []).length > 0;
  const headers = hasBenchmark
    ? ['date', 'strategy_value', 'benchmark_value']
    : ['date', 'strategy_value'];

  const rows: Array<Array<unknown>> = [headers];

  for (const date of collectSeriesDates(equityCurve)) {
    const row: Array<unknown> = [date, strategyByDate.get(date) ?? ''];
    if (hasBenchmark) {
      row.push(benchmarkByDate.get(date) ?? '');
    }
    rows.push(row);
  }

  return toCSVString(rows, includeBom);
}

export function generateDrawdownCSV(
  drawdown: DrawdownData,
  options: { includeBom?: boolean } = {}
): string {
  const includeBom = options.includeBom ?? true;
  const rows: Array<Array<unknown>> = [['date', 'drawdown_percent']];

  for (const point of drawdown.data) {
    rows.push([point.date, point.value]);
  }

  return toCSVString(rows, includeBom);
}

export function generateMonthlyCSV(
  monthlyHeatmap: MonthlyHeatmapData,
  options: { includeBom?: boolean } = {}
): string {
  const includeBom = options.includeBom ?? true;
  const rows: Array<Array<unknown>> = [['year', 'month', 'return_percent']];

  for (let yearIndex = 0; yearIndex < monthlyHeatmap.years.length; yearIndex += 1) {
    const year = monthlyHeatmap.years[yearIndex];

    for (let monthIndex = 0; monthIndex < monthlyHeatmap.months.length; monthIndex += 1) {
      const month = monthlyHeatmap.months[monthIndex];
      const value = monthlyHeatmap.returns[yearIndex]?.[monthIndex] ?? '';
      rows.push([year, month, value]);
    }
  }

  return toCSVString(rows, includeBom);
}

const METRIC_LABELS: Record<keyof PerformanceMetrics, string> = {
  total_return: 'total_return',
  cagr: 'cagr',
  max_drawdown: 'max_drawdown',
  sharpe_ratio: 'sharpe_ratio',
  sortino_ratio: 'sortino_ratio',
  calmar_ratio: 'calmar_ratio',
  volatility: 'volatility',
  total_trades: 'total_trades',
  winning_trades: 'winning_trades',
  losing_trades: 'losing_trades',
  win_rate: 'win_rate'
};

export function generateMetricsCSV(
  metrics: PerformanceMetrics,
  options: { includeBom?: boolean } = {}
): string {
  const includeBom = options.includeBom ?? true;
  const rows: Array<Array<unknown>> = [['metric_name', 'value']];

  (Object.keys(METRIC_LABELS) as Array<keyof PerformanceMetrics>).forEach((key) => {
    rows.push([METRIC_LABELS[key], metrics[key]]);
  });

  return toCSVString(rows, includeBom);
}

function uniqueTradeColumns(trades: Array<Record<string, unknown>>): string[] {
  const columnSet = new Set<string>();

  for (const trade of trades) {
    Object.keys(trade).forEach((key) => columnSet.add(key));
  }

  return Array.from(columnSet).sort((a, b) => a.localeCompare(b));
}

export function generateTradesCSV(
  trades: Array<Record<string, unknown>>,
  options: { includeBom?: boolean } = {}
): string {
  const includeBom = options.includeBom ?? true;

  if (trades.length === 0) {
    return toCSVString([['no_trades']], includeBom);
  }

  const columns = uniqueTradeColumns(trades);
  const rows: Array<Array<unknown>> = [columns];

  for (const trade of trades) {
    rows.push(columns.map((column) => trade[column] ?? ''));
  }

  return toCSVString(rows, includeBom);
}

export function downloadCSV(content: string, filename: string): void {
  const blob = new Blob([content], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');

  anchor.href = url;
  anchor.setAttribute('download', filename);
  anchor.style.display = 'none';

  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  URL.revokeObjectURL(url);
}

export function exportBacktestResultsCsv(
  result: BacktestResultResponse,
  type: 'equity' | 'drawdown' | 'monthly' | 'metrics' | 'trades',
  date = new Date()
): { filename: string; csv: string } {
  const filename = generateFilename(type, result.job_id, date);

  if (type === 'equity') {
    return {
      filename,
      csv: generateEquityCSV(result.equity_curve)
    };
  }

  if (type === 'drawdown') {
    return {
      filename,
      csv: generateDrawdownCSV(result.drawdown)
    };
  }

  if (type === 'monthly') {
    return {
      filename,
      csv: generateMonthlyCSV(result.monthly_heatmap)
    };
  }

  if (type === 'metrics') {
    return {
      filename,
      csv: generateMetricsCSV(result.metrics)
    };
  }

  return {
    filename,
    csv: generateTradesCSV(result.trades)
  };
}
