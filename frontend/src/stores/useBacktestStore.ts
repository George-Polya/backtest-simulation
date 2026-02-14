'use client';

import { create } from 'zustand';
import { BacktestResultResponse, JobStatus } from '@/types';

export type WorkspaceTab = 'code' | 'results';

export interface UiToggles {
  isCodeEditorOpen: boolean;
  isResultsOpen: boolean;
  isConfigExpanded: boolean;
  selectedTab: WorkspaceTab;
}

type UiToggleBooleanKey = Exclude<keyof UiToggles, 'selectedTab'>;

interface BacktestState {
  generatedCode: string;
  jobId: string | null;
  jobStatus: JobStatus | null;
  results: BacktestResultResponse | null;
  uiToggles: UiToggles;
  setGeneratedCode: (code: string) => void;
  setJobState: (jobId: string, status: JobStatus) => void;
  setResults: (results: BacktestResultResponse | null) => void;
  setUiToggle: (key: UiToggleBooleanKey, value: boolean) => void;
  setSelectedTab: (tab: WorkspaceTab) => void;
  reset: () => void;
}

const defaultUiToggles: UiToggles = {
  isCodeEditorOpen: true,
  isResultsOpen: true,
  isConfigExpanded: true,
  selectedTab: 'code'
};

export const useBacktestStore = create<BacktestState>((set) => ({
  generatedCode: '',
  jobId: null,
  jobStatus: null,
  results: null,
  uiToggles: { ...defaultUiToggles },
  setGeneratedCode: (generatedCode) => set({ generatedCode }),
  setJobState: (jobId, jobStatus) => set({ jobId, jobStatus }),
  setResults: (results) => set({ results }),
  setUiToggle: (key, value) =>
    set((state) => ({
      uiToggles: {
        ...state.uiToggles,
        [key]: value
      }
    })),
  setSelectedTab: (selectedTab) =>
    set((state) => ({
      uiToggles: {
        ...state.uiToggles,
        selectedTab
      }
    })),
  reset: () =>
    set({
      generatedCode: '',
      jobId: null,
      jobStatus: null,
      results: null,
      uiToggles: { ...defaultUiToggles }
    })
}));
