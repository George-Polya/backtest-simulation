'use client';

import { useFormContext } from 'react-hook-form';
import { Textarea } from '@/components/ui';
import { BacktestFormValues, STRATEGY_MAX_LENGTH } from '@/lib/validations';

export function StrategyInput() {
  const {
    register,
    watch,
    formState: { errors }
  } = useFormContext<BacktestFormValues>();
  const strategy = watch('strategy') || '';

  return (
    <Textarea
      {...register('strategy')}
      label="Strategy"
      placeholder="Describe your backtest strategy in detail."
      rows={7}
      maxLength={STRATEGY_MAX_LENGTH}
      showCount
      currentLength={strategy.length}
      helperText="Use between 10 and 10000 characters."
      error={errors.strategy?.message}
    />
  );
}
