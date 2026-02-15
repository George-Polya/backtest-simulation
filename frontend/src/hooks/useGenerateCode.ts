'use client';

import { useMutation } from '@tanstack/react-query';
import { generateCode } from '@/lib/api';
import { useBacktestStore } from '@/stores';
import { BacktestRequest, GenerateBacktestResponse, GenerationMetadata } from '@/types';

const GENERATION_TIMEOUT_MS = 30_000;
const GENERATION_TIMEOUT_MESSAGE = 'Code generation timed out after 30 seconds.';

function toGenerationMetadata(response: GenerateBacktestResponse): GenerationMetadata {
  return {
    model_info: response.generated_code.model_info,
    strategy_summary: response.generated_code.strategy_summary,
    tickers_found: response.tickers_found,
    generation_time_seconds: response.generation_time_seconds
  };
}

export function useGenerateCode() {
  const setGeneratedCode = useBacktestStore((state) => state.setGeneratedCode);
  const setGenerationMetadata = useBacktestStore((state) => state.setGenerationMetadata);

  return useMutation<GenerateBacktestResponse, Error, BacktestRequest>({
    mutationFn: async (payload) => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), GENERATION_TIMEOUT_MS);

      try {
        return await generateCode(payload, { signal: controller.signal });
      } catch (error) {
        if (controller.signal.aborted) {
          throw new Error(GENERATION_TIMEOUT_MESSAGE);
        }

        if (error instanceof Error) {
          throw error;
        }

        throw new Error('Failed to generate code.');
      } finally {
        clearTimeout(timeoutId);
      }
    },
    onSuccess: (response) => {
      setGeneratedCode(response.generated_code.code);
      setGenerationMetadata(toGenerationMetadata(response));
    }
  });
}

