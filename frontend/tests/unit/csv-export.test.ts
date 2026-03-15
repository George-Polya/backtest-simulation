import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';
import { ExportButtons } from '@/components/results';
import {
  CSV_BOM,
  escapeCSVValue,
  generateDrawdownCSV,
  generateEquityCSV,
  generateFilename,
  generateMetricsCSV,
  generateMonthlyCSV,
  generateTradesCSV
} from '@/lib/export';
import * as exportLib from '@/lib/export';
import { BacktestResultResponse, JobStatus } from '@/types';

function sampleResult(): BacktestResultResponse {
  return {
    job_id: 'job-321',
    status: JobStatus.Completed,
    metrics: {
      total_return: 12.4,
      cagr: 8.2,
      max_drawdown: 5.1,
      sharpe_ratio: 1.31,
      sortino_ratio: 1.8,
      calmar_ratio: 0.9,
      volatility: 14.3,
      total_trades: 5,
      winning_trades: 3,
      losing_trades: 2,
      win_rate: 60
    },
    equity_curve: {
      strategy: [
        { date: '2024-01-01', value: 10000 },
        { date: '2024-12-31', value: 11420 }
      ],
      benchmark: [
        { date: '2024-01-01', value: 10000 },
        { date: '2024-12-31', value: 10880 }
      ],
      log_scale: false
    },
    drawdown: {
      data: [
        { date: '2024-01-01', value: 0 },
        { date: '2024-06-01', value: -3.2 }
      ]
    },
    monthly_heatmap: {
      years: [2024],
      months: ['Jan', 'Feb'],
      returns: [[1.2, -0.4]]
    },
    trades: [
      {
        symbol: 'SPY',
        side: 'BUY',
        notes: 'alpha, core',
        memo: 'He said "go"',
        pnl: 120.5
      }
    ],
    logs: 'done'
  };
}

describe('csv export utilities', () => {
  it('escapes commas, quotes, and newlines', () => {
    expect(escapeCSVValue('alpha,beta')).toBe('"alpha,beta"');
    expect(escapeCSVValue('hello "world"')).toBe('"hello ""world"""');
    expect(escapeCSVValue('line1\nline2')).toBe('"line1\nline2"');
  });

  it('creates deterministic filename format', () => {
    const filename = generateFilename('equity', 'job-123', new Date('2026-01-05T10:00:00Z'));
    expect(filename).toBe('backtest_equity_job-123_20260105.csv');
  });

  it('generates equity csv with benchmark column and BOM', () => {
    const result = sampleResult();
    const csv = generateEquityCSV(result.equity_curve);

    expect(csv.startsWith(CSV_BOM)).toBe(true);
    expect(csv).toContain('date,strategy_value,benchmark_value');
    expect(csv).toContain('2024-01-01,10000,10000');
    expect(csv).toContain('2024-12-31,11420,10880');
  });

  it('generates drawdown, monthly, metrics and trades csv content', () => {
    const result = sampleResult();

    const drawdownCsv = generateDrawdownCSV(result.drawdown, { includeBom: false });
    expect(drawdownCsv).toContain('date,drawdown_percent');
    expect(drawdownCsv).toContain('2024-06-01,-3.2');

    const monthlyCsv = generateMonthlyCSV(result.monthly_heatmap, { includeBom: false });
    expect(monthlyCsv).toContain('year,month,return_percent');
    expect(monthlyCsv).toContain('2024,Jan,1.2');

    const metricsCsv = generateMetricsCSV(result.metrics, { includeBom: false });
    expect(metricsCsv).toContain('metric_name,value');
    expect(metricsCsv).toContain('total_return,12.4');
    expect(metricsCsv).toContain('win_rate,60');

    const tradesCsv = generateTradesCSV(result.trades, { includeBom: false });
    expect(tradesCsv).toContain('memo,notes,pnl,side,symbol');
    expect(tradesCsv).toContain('"alpha, core"');
    expect(tradesCsv).toContain('"He said ""go"""');
  });

  it('handles empty trades edge case', () => {
    const csv = generateTradesCSV([], { includeBom: false });
    expect(csv).toBe('no_trades');
  });
});

describe('ExportButtons', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('keeps all export buttons disabled until results and job id are available', () => {
    const { rerender } = render(React.createElement(ExportButtons, { results: null, jobId: 'job-1' }));

    screen.getAllByRole('button').forEach((button) => {
      expect(button).toBeDisabled();
    });

    rerender(React.createElement(ExportButtons, { results: sampleResult(), jobId: null }));

    screen.getAllByRole('button').forEach((button) => {
      expect(button).toBeDisabled();
    });
  });

  it('downloads csv for each export button', () => {
    const downloadSpy = vi
      .spyOn(exportLib, 'downloadCSV')
      .mockImplementation(() => undefined);

    render(React.createElement(ExportButtons, { results: sampleResult(), jobId: 'job-1' }));

    fireEvent.click(screen.getByRole('button', { name: 'Export Equity' }));
    fireEvent.click(screen.getByRole('button', { name: 'Export Drawdown' }));
    fireEvent.click(screen.getByRole('button', { name: 'Export Monthly' }));
    fireEvent.click(screen.getByRole('button', { name: 'Export Metrics' }));
    fireEvent.click(screen.getByRole('button', { name: 'Export Trades' }));

    expect(downloadSpy).toHaveBeenCalledTimes(5);
    expect(downloadSpy.mock.calls[0]?.[0].startsWith(CSV_BOM)).toBe(true);
    expect(downloadSpy.mock.calls[0]?.[1]).toContain('backtest_equity_job-1_');
    expect(downloadSpy.mock.calls[4]?.[1]).toContain('backtest_trades_job-1_');
  });
});
