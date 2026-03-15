'use client';

import { Controller, useFormContext } from 'react-hook-form';
import { Toggle } from '@/components/ui';
import { BacktestFormValues } from '@/lib/validations';

export function Toggles() {
  const {
    control,
    formState: { errors }
  } = useFormContext<BacktestFormValues>();
  const llmErrors = errors.llm_settings as { web_search_enabled?: { message?: string } } | undefined;

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Controller
        name="dividend_reinvestment"
        control={control}
        render={({ field }) => (
          <Toggle
            checked={Boolean(field.value)}
            label="Dividend Reinvestment"
            description="Automatically reinvest dividends into the portfolio."
            name={field.name}
            onBlur={field.onBlur}
            onChange={(event) => field.onChange(event.target.checked)}
            error={typeof errors.dividend_reinvestment?.message === 'string' ? errors.dividend_reinvestment.message : undefined}
            ref={field.ref}
          />
        )}
      />

      <Controller
        name="llm_settings.web_search_enabled"
        control={control}
        render={({ field }) => (
          <Toggle
            checked={Boolean(field.value)}
            label="Web Search"
            description="Allow web search to enrich strategy generation context."
            name={field.name}
            onBlur={field.onBlur}
            onChange={(event) => field.onChange(event.target.checked)}
            error={llmErrors?.web_search_enabled?.message}
            ref={field.ref}
          />
        )}
      />
    </div>
  );
}
