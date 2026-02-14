'use client';

import { useBacktestStore } from '@/stores';
import { BacktestRequest } from '@/types';

export function useBacktestActions() {
  const reset = useBacktestStore((state) => state.reset);
  const setRequestConfig = useBacktestStore((state) => state.setRequestConfig);

  return {
    resetState: reset,
    saveConfig: (payload: BacktestRequest) => setRequestConfig(payload)
  };
}
