'use client';

import { useEffect, useMemo, useState } from 'react';
import { Alert, Card, LoadingSpinner } from '@/components/ui';
import { JobStatus } from '@/types';

interface StatusBannerProps {
  status: JobStatus | null;
  jobId: string | null;
  isSubmitting?: boolean;
  isPolling?: boolean;
  startTime?: number | null;
  isJobNotFound?: boolean;
  error?: string | null;
  logs?: string | null;
}

function formatElapsedHHMMSS(totalSeconds: number): string {
  const hours = Math.floor(totalSeconds / 3600)
    .toString()
    .padStart(2, '0');
  const minutes = Math.floor((totalSeconds % 3600) / 60)
    .toString()
    .padStart(2, '0');
  const seconds = Math.floor(totalSeconds % 60)
    .toString()
    .padStart(2, '0');

  return `${hours}:${minutes}:${seconds}`;
}

function toFailureTitle(status: JobStatus | null): string {
  if (status === JobStatus.Cancelled) {
    return 'Backtest cancelled';
  }

  if (status === JobStatus.Timeout) {
    return 'Backtest timed out';
  }

  return 'Backtest failed';
}

export function StatusBanner({
  status,
  jobId,
  isSubmitting = false,
  isPolling = false,
  startTime = null,
  isJobNotFound = false,
  error,
  logs
}: StatusBannerProps) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (status !== JobStatus.Running || !startTime) {
      return;
    }

    const intervalId = setInterval(() => {
      setNow(Date.now());
    }, 1_000);

    return () => {
      clearInterval(intervalId);
    };
  }, [startTime, status]);

  const elapsedLabel = useMemo(() => {
    if (!startTime) {
      return null;
    }

    const elapsedSeconds = Math.max(0, Math.floor((now - startTime) / 1_000));
    return formatElapsedHHMMSS(elapsedSeconds);
  }, [now, startTime]);

  const hasFailureState =
    isJobNotFound ||
    status === JobStatus.Failed ||
    status === JobStatus.Cancelled ||
    status === JobStatus.Timeout;

  if (!isSubmitting && !jobId && !status && !error && !isJobNotFound) {
    return null;
  }

  if (isSubmitting) {
    return (
      <Card className="mb-4" title="Execution Status">
        <Alert title="Submitting backtest" variant="info">
          <div className="flex items-center gap-2">
            <LoadingSpinner size="sm" />
            <p>Sending backtest request to execution API.</p>
          </div>
        </Alert>
      </Card>
    );
  }

  if (isJobNotFound) {
    return (
      <Card className="mb-4" title="Execution Status">
        <Alert title="Job not found" variant="error">
          <p>{error || (jobId ? `Job ${jobId} was not found.` : 'Submitted job was not found.')}</p>
        </Alert>
      </Card>
    );
  }

  if (!status && error) {
    return (
      <Card className="mb-4" title="Execution Status">
        <Alert title="Execution request failed" variant="error">
          <p>{error}</p>
        </Alert>
      </Card>
    );
  }

  if (status === JobStatus.Pending) {
    return (
      <Card className="mb-4" title="Execution Status">
        <Alert title="Backtest pending" variant="info">
          <div className="space-y-1">
            <p>Job ID: {jobId ?? '-'}</p>
            <p>Waiting for worker to start execution.</p>
            {isPolling ? (
              <p className="flex items-center gap-2">
                <LoadingSpinner size="sm" />
                Polling status every 2 seconds.
              </p>
            ) : null}
            {error ? <p>{error}</p> : null}
          </div>
        </Alert>
      </Card>
    );
  }

  if (status === JobStatus.Running) {
    return (
      <Card className="mb-4" title="Execution Status">
        <Alert title="Backtest running" variant="warning">
          <div className="space-y-1">
            <p>Job ID: {jobId ?? '-'}</p>
            <p>Elapsed: {elapsedLabel ?? '00:00:00'}</p>
            <p className="flex items-center gap-2">
              <LoadingSpinner size="sm" />
              Executing strategy and collecting results.
            </p>
            {error ? <p>{error}</p> : null}
          </div>
        </Alert>
      </Card>
    );
  }

  if (status === JobStatus.Completed) {
    return (
      <Card className="mb-4" title="Execution Status">
        <Alert title="Backtest completed" variant="success">
          <div className="space-y-1">
            <p>Job ID: {jobId ?? '-'}</p>
            <p>Execution finished successfully.</p>
            {isPolling ? <p>Finalizing result payload...</p> : null}
          </div>
        </Alert>
      </Card>
    );
  }

  if (hasFailureState) {
    return (
      <Card className="mb-4" title="Execution Status">
        <Alert title={toFailureTitle(status)} variant="error">
          <div className="space-y-1">
            <p>Job ID: {jobId ?? '-'}</p>
            <p>{error || 'Execution did not complete successfully.'}</p>
            {logs ? (
              <pre className="max-h-40 overflow-auto rounded border border-red-100 bg-red-100/50 p-2 text-xs text-red-900">
                {logs}
              </pre>
            ) : null}
          </div>
        </Alert>
      </Card>
    );
  }

  return null;
}
