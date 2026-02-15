'use client';

import { useMemo, useState } from 'react';
import { Alert, Button, Card, LoadingSpinner } from '@/components/ui';
import { formatCurrency, formatDate, formatPercentage } from '@/lib/utils';

interface TradeTableProps {
  trades: Array<Record<string, unknown>> | null;
  isLoading?: boolean;
}

type SortColumn = 'entryDate' | 'exitDate' | 'ticker' | 'action' | 'profit';
type SortDirection = 'asc' | 'desc';

interface NormalizedTrade {
  id: string;
  entryDate: string | null;
  exitDate: string | null;
  ticker: string;
  action: string;
  profit: number | null;
  profitPct: number | null;
}

function toStringValue(value: unknown): string {
  if (typeof value === 'string') {
    return value.trim();
  }

  if (typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }

  return '';
}

function toNumberValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }

  return null;
}

function toTimestamp(value: string | null): number {
  if (!value) {
    return Number.NEGATIVE_INFINITY;
  }

  const timestamp = new Date(value).getTime();
  return Number.isNaN(timestamp) ? Number.NEGATIVE_INFINITY : timestamp;
}

function normalizeTrade(trade: Record<string, unknown>, index: number): NormalizedTrade {
  const ticker =
    toStringValue(trade.ticker) ||
    toStringValue(trade.symbol) ||
    toStringValue(trade.asset) ||
    toStringValue(trade.instrument) ||
    '-';
  const action = (toStringValue(trade.action) || toStringValue(trade.side) || '-').toUpperCase();
  const entryDate = toStringValue(trade.entry_date) || toStringValue(trade.entryDate) || toStringValue(trade.date);
  const exitDate = toStringValue(trade.exit_date) || toStringValue(trade.exitDate) || toStringValue(trade.date);
  const profit =
    toNumberValue(trade.profit) ?? toNumberValue(trade.profit_loss) ?? toNumberValue(trade.pnl);
  const profitPct =
    toNumberValue(trade.profit_pct) ??
    toNumberValue(trade.profitPercent) ??
    toNumberValue(trade.return_pct) ??
    toNumberValue(trade.pnl_pct);

  return {
    id: `${ticker}-${entryDate}-${exitDate}-${index}`,
    entryDate: entryDate || null,
    exitDate: exitDate || null,
    ticker,
    action,
    profit,
    profitPct
  };
}

function compareTrades(
  a: NormalizedTrade,
  b: NormalizedTrade,
  column: SortColumn,
  direction: SortDirection
): number {
  const directionMultiplier = direction === 'asc' ? 1 : -1;

  if (column === 'entryDate') {
    return (toTimestamp(a.entryDate) - toTimestamp(b.entryDate)) * directionMultiplier;
  }

  if (column === 'exitDate') {
    return (toTimestamp(a.exitDate) - toTimestamp(b.exitDate)) * directionMultiplier;
  }

  if (column === 'profit') {
    const left = a.profit ?? Number.NEGATIVE_INFINITY;
    const right = b.profit ?? Number.NEGATIVE_INFINITY;
    return (left - right) * directionMultiplier;
  }

  const left = (column === 'ticker' ? a.ticker : a.action).toLowerCase();
  const right = (column === 'ticker' ? b.ticker : b.action).toLowerCase();
  return left.localeCompare(right) * directionMultiplier;
}

function getActionBadgeClass(action: string): string {
  if (action === 'BUY') {
    return 'border-emerald-200 bg-emerald-50 text-emerald-800';
  }

  if (action === 'SELL') {
    return 'border-red-200 bg-red-50 text-red-800';
  }

  return 'border-slate-200 bg-slate-50 text-slate-700';
}

function getProfitClass(profit: number | null): string {
  if (profit === null) {
    return 'text-slate-500';
  }

  if (profit > 0) {
    return 'text-emerald-700';
  }

  if (profit < 0) {
    return 'text-red-700';
  }

  return 'text-slate-700';
}

function SortButton({
  activeColumn,
  column,
  direction,
  label,
  onSort
}: {
  activeColumn: SortColumn;
  column: SortColumn;
  direction: SortDirection;
  label: string;
  onSort: (column: SortColumn) => void;
}) {
  const isActive = activeColumn === column;
  const icon = isActive ? (direction === 'asc' ? '▲' : '▼') : '↕';

  return (
    <button
      className="inline-flex items-center gap-1 font-semibold text-slate-700 hover:text-slate-900"
      onClick={() => onSort(column)}
      type="button"
    >
      <span>{label}</span>
      <span aria-hidden="true" className="text-[10px]">
        {icon}
      </span>
    </button>
  );
}

export function TradeTable({ trades, isLoading = false }: TradeTableProps) {
  const [isExpanded, setIsExpanded] = useState(true);
  const [sortColumn, setSortColumn] = useState<SortColumn>('entryDate');
  const [sortDirection, setSortDirection] = useState<SortDirection>('desc');

  const normalizedTrades = useMemo(
    () => (trades ?? []).map((trade, index) => normalizeTrade(trade, index)),
    [trades]
  );
  const sortedTrades = useMemo(() => {
    const copy = [...normalizedTrades];
    copy.sort((a, b) => compareTrades(a, b, sortColumn, sortDirection));
    return copy;
  }, [normalizedTrades, sortColumn, sortDirection]);

  const handleSort = (column: SortColumn) => {
    if (sortColumn === column) {
      setSortDirection((previous) => (previous === 'asc' ? 'desc' : 'asc'));
      return;
    }

    setSortColumn(column);
    setSortDirection(column === 'profit' ? 'desc' : 'asc');
  };

  return (
    <Card title="Trade History">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-slate-600">{normalizedTrades.length} trade(s)</p>
        <Button onClick={() => setIsExpanded((previous) => !previous)} type="button" variant="secondary">
          {isExpanded ? 'Collapse' : 'Expand'}
        </Button>
      </div>

      {isLoading ? (
        <div className="py-6">
          <LoadingSpinner centered size="md" />
          <p className="mt-2 text-center text-sm text-slate-600">Loading trade history...</p>
        </div>
      ) : null}

      {!isLoading && normalizedTrades.length === 0 ? (
        <Alert title="No trades available" variant="info">
          This backtest did not return any trade rows.
        </Alert>
      ) : null}

      {!isLoading && normalizedTrades.length > 0 && isExpanded ? (
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-left text-sm">
            <thead>
              <tr className="bg-slate-50">
                <th className="border border-slate-200 px-3 py-2">
                  <SortButton
                    activeColumn={sortColumn}
                    column="entryDate"
                    direction={sortDirection}
                    label="Entry"
                    onSort={handleSort}
                  />
                </th>
                <th className="border border-slate-200 px-3 py-2">
                  <SortButton
                    activeColumn={sortColumn}
                    column="exitDate"
                    direction={sortDirection}
                    label="Exit"
                    onSort={handleSort}
                  />
                </th>
                <th className="border border-slate-200 px-3 py-2">
                  <SortButton
                    activeColumn={sortColumn}
                    column="ticker"
                    direction={sortDirection}
                    label="Ticker"
                    onSort={handleSort}
                  />
                </th>
                <th className="border border-slate-200 px-3 py-2">
                  <SortButton
                    activeColumn={sortColumn}
                    column="action"
                    direction={sortDirection}
                    label="Action"
                    onSort={handleSort}
                  />
                </th>
                <th className="border border-slate-200 px-3 py-2">
                  <SortButton
                    activeColumn={sortColumn}
                    column="profit"
                    direction={sortDirection}
                    label="Profit/Loss"
                    onSort={handleSort}
                  />
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedTrades.map((trade) => (
                <tr className="odd:bg-white even:bg-slate-50/40" key={trade.id}>
                  <td className="border border-slate-200 px-3 py-2">
                    {trade.entryDate ? formatDate(trade.entryDate) : '-'}
                  </td>
                  <td className="border border-slate-200 px-3 py-2">
                    {trade.exitDate ? formatDate(trade.exitDate) : '-'}
                  </td>
                  <td className="border border-slate-200 px-3 py-2 font-medium text-slate-900">
                    {trade.ticker}
                  </td>
                  <td className="border border-slate-200 px-3 py-2">
                    <span className={`inline-flex rounded-full border px-2 py-0.5 text-xs font-semibold ${getActionBadgeClass(trade.action)}`}>
                      {trade.action === 'BUY' ? 'Buy' : trade.action === 'SELL' ? 'Sell' : trade.action}
                    </span>
                  </td>
                  <td className={`border border-slate-200 px-3 py-2 font-semibold ${getProfitClass(trade.profit)}`}>
                    {trade.profit === null ? (
                      '-'
                    ) : (
                      <div className="space-y-0.5">
                        <p>
                          {formatCurrency(trade.profit, {
                            signed: true,
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2
                          })}
                        </p>
                        <p className="text-xs font-medium">
                          {trade.profitPct === null
                            ? '-'
                            : formatPercentage(trade.profitPct, {
                                signed: true,
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2
                              })}
                        </p>
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </Card>
  );
}
