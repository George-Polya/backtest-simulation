'use client';

import { create } from 'zustand';
import { BacktestResultResponse, JobStatus } from '@/types';

interface BacktestState {
  generatedCode: string;
  jobId: string | null;
  jobStatus: JobStatus | null;
  results: BacktestResultResponse | null;
  setGeneratedCode: (code: string) => void;
  setJobState: (jobId: string, status: JobStatus) => void;
  setResults: (results: BacktestResultResponse | null) => void;
  reset: () => void;
}

export const useBacktestStore = create<BacktestState>((set) => ({
  generatedCode: '',
  jobId: null,
  jobStatus: null,
  results: null,
  setGeneratedCode: (generatedCode) => set({ generatedCode }),
  setJobState: (jobId, jobStatus) => set({ jobId, jobStatus }),
  setResults: (results) => set({ results }),
  reset: () =>
    set({
      generatedCode: '',
      jobId: null,
      jobStatus: null,
      results: null
    })
}));
