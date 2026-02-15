'use client';

import { useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import { ApiError, getFormattedResult, getJobResult, getJobStatus } from '@/lib/api';
import { useBacktestStore } from '@/stores';
import { JobStatus, JobStatusResponse } from '@/types';

const JOB_POLLING_INTERVAL_MS = 2_000;
const NETWORK_RETRY_ATTEMPTS = 3;

const TERMINAL_STATUSES = new Set<JobStatus>([
  JobStatus.Completed,
  JobStatus.Failed,
  JobStatus.Cancelled,
  JobStatus.Timeout
]);

function isJobNotFoundError(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404;
}

function isRetryableNetworkError(error: unknown): boolean {
  if (!(error instanceof ApiError)) {
    return false;
  }

  return error.code === 'NETWORK_ERROR' || error.code === 'TIMEOUT_ERROR';
}

function toExecutionResultError(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message.trim().length > 0) {
    return error.message;
  }

  return fallback;
}

interface UseJobPollingResult {
  status: JobStatus | null;
  data: JobStatusResponse | undefined;
  isLoading: boolean;
  isPolling: boolean;
  error: Error | null;
  isJobNotFound: boolean;
}

export function useJobPolling(jobId: string | null): UseJobPollingResult {
  const setJobState = useBacktestStore((state) => state.setJobState);
  const setResults = useBacktestStore((state) => state.setResults);
  const setExecutionResult = useBacktestStore((state) => state.setExecutionResult);
  const setJobNotFound = useBacktestStore((state) => state.setJobNotFound);
  const isJobNotFound = useBacktestStore((state) => state.isJobNotFound);

  const fetchedTerminalJobId = useRef<string | null>(null);

  useEffect(() => {
    fetchedTerminalJobId.current = null;
  }, [jobId]);

  const query = useQuery<JobStatusResponse, Error>({
    queryKey: ['jobStatus', jobId],
    queryFn: async () => {
      if (!jobId) {
        throw new Error('Job ID is required to poll status.');
      }

      return getJobStatus(jobId);
    },
    enabled: Boolean(jobId),
    refetchInterval: (queryState) => {
      if (queryState.state.error) {
        return false;
      }

      const status = queryState.state.data?.status;
      if (status && TERMINAL_STATUSES.has(status)) {
        return false;
      }

      return JOB_POLLING_INTERVAL_MS;
    },
    retry: (failureCount, error) => {
      if (isJobNotFoundError(error)) {
        return false;
      }

      if (isRetryableNetworkError(error)) {
        return failureCount < NETWORK_RETRY_ATTEMPTS;
      }

      return false;
    },
    retryDelay: (attemptIndex) => Math.min(500 * 2 ** attemptIndex, 4_000)
  });

  useEffect(() => {
    if (!query.data) {
      return;
    }

    setJobNotFound(false);
    setJobState(query.data.job_id, query.data.status);
  }, [query.data, setJobNotFound, setJobState]);

  useEffect(() => {
    if (!jobId || !query.error) {
      return;
    }

    if (isJobNotFoundError(query.error)) {
      setJobNotFound(true);
      setExecutionResult({
        success: false,
        job_id: jobId,
        status: JobStatus.Failed,
        data: null,
        error: `Job not found: ${jobId}`,
        logs: ''
      });
      return;
    }

    setJobNotFound(false);
    setExecutionResult({
      success: false,
      job_id: jobId,
      status: query.data?.status ?? JobStatus.Failed,
      data: null,
      error: toExecutionResultError(query.error, 'Failed to poll job status.'),
      logs: ''
    });
  }, [jobId, query.data?.status, query.error, setExecutionResult, setJobNotFound]);

  useEffect(() => {
    if (!jobId || !query.data) {
      return;
    }

    const status = query.data.status;
    if (!TERMINAL_STATUSES.has(status)) {
      return;
    }

    if (fetchedTerminalJobId.current === jobId) {
      return;
    }

    fetchedTerminalJobId.current = jobId;

    let cancelled = false;

    const hydrateTerminalResult = async () => {
      try {
        const result = await getJobResult(jobId);
        if (cancelled) {
          return;
        }

        setExecutionResult(result);

        if (result.status === JobStatus.Completed) {
          try {
            const formatted = await getFormattedResult(jobId);
            if (!cancelled) {
              setResults(formatted);
            }
          } catch {
            if (!cancelled) {
              setResults(null);
            }
          }
        } else {
          setResults(null);
        }
      } catch (error) {
        if (cancelled) {
          return;
        }

        setExecutionResult({
          success: false,
          job_id: jobId,
          status,
          data: null,
          error: toExecutionResultError(error, 'Failed to fetch terminal job result.'),
          logs: ''
        });
      }
    };

    void hydrateTerminalResult();

    return () => {
      cancelled = true;
    };
  }, [jobId, query.data, setExecutionResult, setResults]);

  return {
    status: query.data?.status ?? null,
    data: query.data,
    isLoading: query.isLoading,
    isPolling: query.isFetching,
    error: query.error,
    isJobNotFound
  };
}
