'use client';

import { create } from 'zustand';
import { BacktestRequest, BacktestResultResponse, GenerationMetadata, JobStatus } from '@/types';

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
  generationMetadata: GenerationMetadata | null;
  jobId: string | null;
  jobStatus: JobStatus | null;
  results: BacktestResultResponse | null;
  requestConfig: BacktestRequest | null;
  uiToggles: UiToggles;
  setGeneratedCode: (code: string) => void;
  setGenerationMetadata: (metadata: GenerationMetadata | null) => void;
  setJobState: (jobId: string, status: JobStatus) => void;
  setResults: (results: BacktestResultResponse | null) => void;
  setRequestConfig: (requestConfig: BacktestRequest | null) => void;
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
  generationMetadata: null,
  jobId: null,
  jobStatus: null,
  results: null,
  requestConfig: null,
  uiToggles: { ...defaultUiToggles },
  setGeneratedCode: (generatedCode) => set({ generatedCode }),
  setGenerationMetadata: (generationMetadata) => set({ generationMetadata }),
  setJobState: (jobId, jobStatus) => set({ jobId, jobStatus }),
  setResults: (results) => set({ results }),
  setRequestConfig: (requestConfig) => set({ requestConfig }),
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
      generationMetadata: null,
      jobId: null,
      jobStatus: null,
      results: null,
      requestConfig: null,
      uiToggles: { ...defaultUiToggles }
    })
}));
