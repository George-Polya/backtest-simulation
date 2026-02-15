'use client';

import { useMutation } from '@tanstack/react-query';
import { executeBacktest } from '@/lib/api';
import { useBacktestStore } from '@/stores';
import { ExecuteBacktestRequest, ExecuteBacktestResponse } from '@/types';

export function useExecuteBacktest() {
  const setJobState = useBacktestStore((state) => state.setJobState);
  const setExecutionResult = useBacktestStore((state) => state.setExecutionResult);
  const setJobNotFound = useBacktestStore((state) => state.setJobNotFound);
  const setResults = useBacktestStore((state) => state.setResults);

  return useMutation<ExecuteBacktestResponse, Error, ExecuteBacktestRequest>({
    mutationFn: async (payload) => {
      try {
        return await executeBacktest({
          ...payload,
          async_mode: true
        });
      } catch (error) {
        if (error instanceof Error) {
          throw error;
        }

        throw new Error('Failed to execute backtest.');
      }
    },
    onMutate: () => {
      setExecutionResult(null);
      setResults(null);
      setJobNotFound(false);
    },
    onSuccess: (response) => {
      setJobState(response.job_id, response.status, Date.now());
      setExecutionResult(response.result ?? null);
    }
  });
}
