'use client';

import { useFormContext } from 'react-hook-form';
import { Input } from '@/components/ui';
import { BacktestFormValues } from '@/lib/validations';

export function FeeSettings() {
  const {
    register,
    formState: { errors }
  } = useFormContext<BacktestFormValues>();
  const feeErrors = errors.fees as
    | { trading_fee_percent?: { message?: string }; slippage_percent?: { message?: string } }
    | undefined;

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Input
        {...register('fees.trading_fee_percent', { valueAsNumber: true })}
        type="number"
        min={0}
        max={10}
        step={0.01}
        label="Trading Fee (%)"
        helperText="Allowed range: 0 to 10%."
        error={feeErrors?.trading_fee_percent?.message}
      />
      <Input
        {...register('fees.slippage_percent', { valueAsNumber: true })}
        type="number"
        min={0}
        max={10}
        step={0.01}
        label="Slippage (%)"
        helperText="Allowed range: 0 to 10%."
        error={feeErrors?.slippage_percent?.message}
      />
    </div>
  );
}
