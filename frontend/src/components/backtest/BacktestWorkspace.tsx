'use client';

import { useBacktestActions } from '@/hooks';
import { useBacktestStore } from '@/stores';
import { Button, Card } from '@/components/ui';

export function BacktestWorkspace() {
  const { generatedCode, jobId, jobStatus } = useBacktestStore();
  const { resetState } = useBacktestActions();

  return (
    <div className="grid gap-4 md:grid-cols-2">
      <Card title="Configuration & Actions">
        <p className="mb-4 text-sm text-slate-600">
          Task 1 scaffold complete. Config form and execution flow will be added in subsequent tasks.
        </p>
        <div className="flex items-center gap-2">
          <Button type="button">Generate Code</Button>
          <Button type="button" variant="secondary" onClick={resetState}>
            Reset
          </Button>
        </div>
      </Card>

      <Card title="Code & Results">
        <dl className="grid grid-cols-1 gap-2 text-sm text-slate-700">
          <div>
            <dt className="font-medium">Generated code</dt>
            <dd>{generatedCode ? 'Available' : 'Not generated yet'}</dd>
          </div>
          <div>
            <dt className="font-medium">Job ID</dt>
            <dd>{jobId ?? '-'}</dd>
          </div>
          <div>
            <dt className="font-medium">Job Status</dt>
            <dd>{jobStatus ?? '-'}</dd>
          </div>
        </dl>
      </Card>
    </div>
  );
}
