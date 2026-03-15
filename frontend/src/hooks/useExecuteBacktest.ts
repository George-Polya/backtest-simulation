'use client';

import { useCallback, useRef, useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { executeBacktest } from '@/lib/api';
import { isDockerError } from '@/lib/utils';
import { useBacktestStore } from '@/stores';
import { ExecuteBacktestRequest, ExecuteBacktestResponse } from '@/types';

const DOCKER_RETRY_DELAY_MS = 2_000;

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function executeWithDockerRetry(
  payload: ExecuteBacktestRequest,
  onRetrying: () => void
): Promise<ExecuteBacktestResponse> {
  try {
    return await executeBacktest({ ...payload, async_mode: true });
  } catch (firstError) {
    if (!isDockerError(firstError)) {
      throw firstError instanceof Error ? firstError : new Error('Failed to execute backtest.');
    }

    onRetrying();
    await delay(DOCKER_RETRY_DELAY_MS);

    try {
      return await executeBacktest({ ...payload, async_mode: true });
    } catch (retryError) {
      throw retryError instanceof Error ? retryError : new Error('Failed to execute backtest.');
    }
  }
}

export function useExecuteBacktest() {
  const setJobState = useBacktestStore((state) => state.setJobState);
  const setExecutionResult = useBacktestStore((state) => state.setExecutionResult);
  const setJobNotFound = useBacktestStore((state) => state.setJobNotFound);
  const setResults = useBacktestStore((state) => state.setResults);
  const [isRetrying, setIsRetrying] = useState(false);
  const isRetryingRef = useRef(false);

  const handleRetrying = useCallback(() => {
    isRetryingRef.current = true;
    setIsRetrying(true);
  }, []);

  const mutation = useMutation<ExecuteBacktestResponse, Error, ExecuteBacktestRequest>({
    mutationFn: async (payload) => {
      isRetryingRef.current = false;
      setIsRetrying(false);

      return executeWithDockerRetry(payload, handleRetrying);
    },
    onMutate: () => {
      setExecutionResult(null);
      setResults(null);
      setJobNotFound(false);
    },
    onSuccess: (response) => {
      isRetryingRef.current = false;
      setIsRetrying(false);
      setJobState(response.job_id, response.status, Date.now());
      setExecutionResult(response.result ?? null);
    },
    onError: () => {
      isRetryingRef.current = false;
      setIsRetrying(false);
    }
  });

  return {
    ...mutation,
    isRetrying
  };
}
