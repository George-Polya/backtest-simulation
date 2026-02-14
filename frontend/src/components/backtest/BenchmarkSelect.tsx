'use client';

import { Controller, useFormContext } from 'react-hook-form';
import { BacktestFormValues } from '@/lib/validations';

export interface BenchmarkOption {
  value: string;
  label: string;
}

export const BENCHMARK_OPTIONS: BenchmarkOption[] = [
  { value: 'SPY', label: 'S&P 500 (SPY)' },
  { value: 'QQQ', label: 'NASDAQ-100 (QQQ)' },
  { value: 'DIA', label: 'Dow Jones (DIA)' }
];

export function BenchmarkSelect() {
  const {
    control,
    formState: { errors }
  } = useFormContext<BacktestFormValues>();
  const benchmarkError =
    typeof errors.benchmarks?.message === 'string' ? errors.benchmarks.message : undefined;

  return (
    <Controller
      name="benchmarks"
      control={control}
      render={({ field }) => {
        const selected = field.value || [];

        const toggleBenchmark = (value: string) => {
          if (selected.includes(value)) {
            field.onChange(selected.filter((item: string) => item !== value));
            return;
          }

          field.onChange([...selected, value]);
        };

        return (
          <fieldset className="rounded-lg border border-[var(--border)] bg-white p-3">
            <legend className="px-1 text-sm font-medium text-slate-700">Benchmarks</legend>
            <p className="mb-2 text-xs text-slate-500">Select at least one benchmark.</p>

            <div className="grid gap-2 sm:grid-cols-2">
              {BENCHMARK_OPTIONS.map((option) => {
                const checked = selected.includes(option.value);
                return (
                  <label
                    className={`flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1.5 text-sm transition ${
                      checked
                        ? 'border-brand-300 bg-brand-50 text-brand-800'
                        : 'border-[var(--border)] bg-white text-slate-700 hover:bg-slate-50'
                    }`}
                    key={option.value}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      onBlur={field.onBlur}
                      onChange={() => toggleBenchmark(option.value)}
                    />
                    <span>{option.label}</span>
                  </label>
                );
              })}
            </div>

            <p className="mt-2 text-xs text-slate-500">Selected: {selected.length}</p>
            {benchmarkError ? (
              <p className="mt-1 text-xs text-red-600">{benchmarkError}</p>
            ) : null}
          </fieldset>
        );
      }}
    />
  );
}
