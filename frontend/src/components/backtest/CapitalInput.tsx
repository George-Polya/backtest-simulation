'use client';

import { useMemo } from 'react';
import { useFormContext } from 'react-hook-form';
import { Input } from '@/components/ui';
import { BacktestFormValues } from '@/lib/validations';

function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0
  }).format(value);
}

export function CapitalInput() {
  const {
    register,
    watch,
    formState: { errors }
  } = useFormContext<BacktestFormValues>();

  const value = watch('initial_capital');
  const helperText = useMemo(() => {
    const amount = Number.isFinite(value) ? value : 0;
    return `Formatted: ${formatCurrency(amount)}`;
  }, [value]);

  return (
    <Input
      {...register('initial_capital', { valueAsNumber: true })}
      type="number"
      min={1000}
      step={1000}
      label="Initial Capital"
      helperText={helperText}
      error={errors.initial_capital?.message}
    />
  );
}
