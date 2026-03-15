'use client';

import { Alert, Card, LoadingSpinner } from '@/components/ui';
import { formatPercentage } from '@/lib/utils';
import { MonthlyHeatmapData } from '@/types';

interface MonthlyHeatmapProps {
  monthlyHeatmap: MonthlyHeatmapData | null;
  isLoading?: boolean;
}

function toHeatmapCellStyle(value: number | null): { backgroundColor: string; color: string } {
  if (value === null) {
    return {
      backgroundColor: '#e2e8f0',
      color: '#475569'
    };
  }

  if (value === 0) {
    return {
      backgroundColor: '#f1f5f9',
      color: '#334155'
    };
  }

  const intensity = Math.min(Math.abs(value) / 20, 1);

  if (value > 0) {
    const lightness = 92 - intensity * 42;
    return {
      backgroundColor: `hsl(152 58% ${lightness}%)`,
      color: intensity > 0.5 ? '#14532d' : '#166534'
    };
  }

  const lightness = 94 - intensity * 44;
  return {
    backgroundColor: `hsl(0 72% ${lightness}%)`,
    color: intensity > 0.45 ? '#7f1d1d' : '#991b1b'
  };
}

export function MonthlyHeatmap({ monthlyHeatmap, isLoading = false }: MonthlyHeatmapProps) {
  const years = monthlyHeatmap?.years ?? [];
  const months = monthlyHeatmap?.months ?? [];
  const returns = monthlyHeatmap?.returns ?? [];
  const hasData = years.length > 0 && months.length > 0 && returns.length > 0;

  return (
    <Card title="Monthly Returns Heatmap">
      {isLoading ? (
        <div className="py-6">
          <LoadingSpinner centered size="md" />
          <p className="mt-2 text-center text-sm text-slate-600">Loading monthly returns...</p>
        </div>
      ) : null}

      {!isLoading && !hasData ? (
        <Alert title="Monthly returns unavailable" variant="info">
          Monthly heatmap data is not available for this result.
        </Alert>
      ) : null}

      {!isLoading && hasData ? (
        <div className="overflow-x-auto">
          <table className="min-w-full border-collapse text-center text-xs">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 border border-slate-200 bg-white px-3 py-2 text-left text-slate-600">
                  Year
                </th>
                {months.map((month) => (
                  <th
                    className="border border-slate-200 bg-slate-50 px-3 py-2 font-semibold text-slate-700"
                    key={month}
                    scope="col"
                  >
                    {month}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {years.map((year, yearIndex) => (
                <tr key={year}>
                  <th className="sticky left-0 border border-slate-200 bg-white px-3 py-2 text-left text-sm font-semibold text-slate-800">
                    {year}
                  </th>
                  {months.map((month, monthIndex) => {
                    const value = returns[yearIndex]?.[monthIndex] ?? null;
                    const style = toHeatmapCellStyle(value);

                    return (
                      <td
                        className="border border-slate-200 px-3 py-2 font-medium"
                        key={`${year}-${month}`}
                        style={style}
                      >
                        {value === null
                          ? '-'
                          : formatPercentage(value, {
                              minimumFractionDigits: 1,
                              maximumFractionDigits: 1
                            })}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </Card>
  );
}
