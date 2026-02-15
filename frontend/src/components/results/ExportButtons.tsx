'use client';

import { Button, Card } from '@/components/ui';
import { downloadCSV, exportBacktestResultsCsv } from '@/lib/export';
import { BacktestResultResponse } from '@/types';

interface ExportButtonsProps {
  results: BacktestResultResponse | null;
  jobId: string | null;
}

export function ExportButtons({ results, jobId }: ExportButtonsProps) {
  const isDisabled = !results || !jobId;

  const handleExport = (type: 'equity' | 'drawdown' | 'monthly' | 'metrics' | 'trades') => {
    if (!results || !jobId) {
      return;
    }

    const { filename, csv } = exportBacktestResultsCsv({ ...results, job_id: jobId }, type);
    downloadCSV(csv, filename);
  };

  return (
    <Card title="Export CSV">
      <div className="flex flex-wrap gap-2">
        <Button
          type="button"
          variant="secondary"
          disabled={isDisabled}
          onClick={() => handleExport('equity')}
        >
          Export Equity
        </Button>
        <Button
          type="button"
          variant="secondary"
          disabled={isDisabled}
          onClick={() => handleExport('drawdown')}
        >
          Export Drawdown
        </Button>
        <Button
          type="button"
          variant="secondary"
          disabled={isDisabled}
          onClick={() => handleExport('monthly')}
        >
          Export Monthly
        </Button>
        <Button
          type="button"
          variant="secondary"
          disabled={isDisabled}
          onClick={() => handleExport('metrics')}
        >
          Export Metrics
        </Button>
        <Button
          type="button"
          variant="secondary"
          disabled={isDisabled}
          onClick={() => handleExport('trades')}
        >
          Export Trades
        </Button>
      </div>
    </Card>
  );
}
