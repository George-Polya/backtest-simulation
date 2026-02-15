import { useBacktestStore } from '@/stores';
import { GenerationMetadata } from '@/types';

function createMetadata(): GenerationMetadata {
  return {
    model_info: {
      provider: 'openrouter',
      model_id: 'anthropic/claude-3.5-sonnet',
      max_tokens: 8000,
      supports_system_prompt: true,
      cost_per_1k_input: 0.003,
      cost_per_1k_output: 0.015
    },
    strategy_summary: 'Momentum strategy using SPY.',
    tickers_found: ['SPY'],
    generation_time_seconds: 1.5
  };
}

describe('useBacktestStore', () => {
  afterEach(() => {
    useBacktestStore.getState().reset();
  });

  it('stores generation metadata and clears it on reset', () => {
    const metadata = createMetadata();

    useBacktestStore.getState().setGenerationMetadata(metadata);
    expect(useBacktestStore.getState().generationMetadata).toEqual(metadata);

    useBacktestStore.getState().reset();
    expect(useBacktestStore.getState().generationMetadata).toBeNull();
  });
});

