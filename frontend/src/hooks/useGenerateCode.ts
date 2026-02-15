'use client';

import { useMutation } from '@tanstack/react-query';
import { generateCode } from '@/lib/api';
import { useBacktestStore } from '@/stores';
import { BacktestRequest, GenerateBacktestResponse, GenerationMetadata } from '@/types';

const DEFAULT_GENERATION_TIMEOUT_MS = 120_000;

function getGenerationTimeoutMs(): number {
  const raw = process.env.NEXT_PUBLIC_CODE_GENERATION_TIMEOUT_MS;
  if (!raw) {
    return DEFAULT_GENERATION_TIMEOUT_MS;
  }

  const parsed = Number.parseInt(raw, 10);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return DEFAULT_GENERATION_TIMEOUT_MS;
  }

  return parsed;
}

function buildGenerationTimeoutMessage(timeoutMs: number): string {
  const timeoutSeconds = Math.max(1, Math.ceil(timeoutMs / 1000));
  return `Code generation timed out after ${timeoutSeconds} seconds.`;
}

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
  const generationTimeoutMs = getGenerationTimeoutMs();
  const generationTimeoutMessage = buildGenerationTimeoutMessage(generationTimeoutMs);

  return useMutation<GenerateBacktestResponse, Error, BacktestRequest>({
    mutationFn: async (payload) => {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), generationTimeoutMs);

      try {
        return await generateCode(payload, {
          signal: controller.signal,
          timeoutMs: generationTimeoutMs
        });
      } catch (error) {
        if (controller.signal.aborted) {
          throw new Error(generationTimeoutMessage);
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
