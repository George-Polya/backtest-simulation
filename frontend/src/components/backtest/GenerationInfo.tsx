'use client';

import { Alert } from '@/components/ui';
import { GenerationMetadata } from '@/types';

interface GenerationInfoProps {
  metadata: GenerationMetadata | null;
}

function formatGenerationSeconds(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return '-';
  }

  if (seconds < 1) {
    return `${Math.round(seconds * 1000)} ms`;
  }

  return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
}

export function GenerationInfo({ metadata }: GenerationInfoProps) {
  if (!metadata) {
    return (
      <Alert title="Generation metadata unavailable" variant="info">
        Generate code to view model details, summary, and detected tickers.
      </Alert>
    );
  }

  return (
    <section className="space-y-3">
      <div className="rounded-lg border border-[var(--border)] bg-white p-4">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-semibold uppercase tracking-wide text-slate-600">
            Model
          </span>
          <span className="rounded-full bg-brand-50 px-2 py-1 text-xs font-semibold text-brand-700">
            {metadata.model_info.provider} / {metadata.model_info.model_id}
          </span>
        </div>

        <dl className="space-y-3 text-sm">
          <div>
            <dt className="font-medium text-slate-700">Strategy Summary</dt>
            <dd className="mt-1 text-slate-600">{metadata.strategy_summary || '-'}</dd>
          </div>
          <div>
            <dt className="font-medium text-slate-700">Generation Duration</dt>
            <dd className="mt-1 text-slate-600">
              {formatGenerationSeconds(metadata.generation_time_seconds)}
            </dd>
          </div>
          <div>
            <dt className="font-medium text-slate-700">Detected Tickers</dt>
            <dd className="mt-1 flex flex-wrap gap-2">
              {metadata.tickers_found.length === 0 ? (
                <span className="text-slate-500">No tickers found.</span>
              ) : (
                metadata.tickers_found.map((ticker) => (
                  <span
                    key={ticker}
                    className="rounded-full bg-slate-100 px-2 py-1 text-xs font-semibold text-slate-700"
                  >
                    {ticker}
                  </span>
                ))
              )}
            </dd>
          </div>
        </dl>
      </div>
    </section>
  );
}

