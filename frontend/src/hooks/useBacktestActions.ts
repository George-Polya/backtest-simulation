'use client';

import { useBacktestStore } from '@/stores';

export function useBacktestActions() {
  const reset = useBacktestStore((state) => state.reset);

  return {
    resetState: reset
  };
}
