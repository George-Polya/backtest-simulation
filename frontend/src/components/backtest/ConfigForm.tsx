'use client';

import { useState } from 'react';
import { FormProvider, useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { Alert, Button } from '@/components/ui';
import { BacktestRequest } from '@/types';
import {
  backtestFormSchema,
  BacktestFormValues,
  createDefaultBacktestFormValues
} from '@/lib/validations';
import { BenchmarkSelect } from './BenchmarkSelect';
import { CapitalInput } from './CapitalInput';
import { ContributionSettings } from './ContributionSettings';
import { DateRangePicker } from './DateRangePicker';
import { FeeSettings } from './FeeSettings';
import { StrategyInput } from './StrategyInput';
import { Toggles } from './Toggles';
import { ValidationSummary } from './ValidationSummary';

interface ConfigFormProps {
  onSubmitConfig?: (payload: BacktestRequest) => void;
}

function toBacktestRequest(values: BacktestFormValues): BacktestRequest {
  return {
    strategy: values.strategy,
    params: {
      start_date: values.start_date,
      end_date: values.end_date,
      initial_capital: values.initial_capital,
      benchmarks: values.benchmarks,
      contribution: values.contribution,
      fees: values.fees,
      dividend_reinvestment: values.dividend_reinvestment,
      llm_settings: {
        provider: values.llm_settings.provider,
        web_search_enabled: values.llm_settings.web_search_enabled
      }
    }
  };
}

export function ConfigForm({ onSubmitConfig }: ConfigFormProps) {
  const [savedAt, setSavedAt] = useState<string | null>(null);

  const methods = useForm<BacktestFormValues>({
    resolver: zodResolver(backtestFormSchema),
    defaultValues: createDefaultBacktestFormValues(),
    mode: 'onSubmit',
    reValidateMode: 'onChange'
  });

  const {
    handleSubmit,
    reset,
    formState: { isSubmitting }
  } = methods;

  const submit = (values: BacktestFormValues) => {
    const payload = toBacktestRequest(values);
    onSubmitConfig?.(payload);
    setSavedAt(
      new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      })
    );
  };

  const handleReset = () => {
    reset(createDefaultBacktestFormValues());
    setSavedAt(null);
  };

  return (
    <FormProvider {...methods}>
      <form className="space-y-4" noValidate onSubmit={handleSubmit(submit)}>
        <ValidationSummary />
        <input type="hidden" {...methods.register('llm_settings.provider')} />
        <StrategyInput />
        <DateRangePicker />
        <CapitalInput />
        <BenchmarkSelect />
        <ContributionSettings />
        <FeeSettings />
        <Toggles />

        <div className="flex flex-wrap gap-2">
          <Button type="submit">{isSubmitting ? 'Saving...' : 'Save Configuration'}</Button>
          <Button onClick={handleReset} type="button" variant="secondary">
            Reset Form
          </Button>
        </div>

        {savedAt ? (
          <Alert title="Configuration saved" variant="success">
            Last saved at {savedAt}.
          </Alert>
        ) : null}
      </form>
    </FormProvider>
  );
}
