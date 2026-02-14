'use client';

import { useMemo } from 'react';
import { useFormContext } from 'react-hook-form';
import { Input } from '@/components/ui';
import { BacktestFormValues, toIsoDateString } from '@/lib/validations';

export function DateRangePicker() {
  const {
    register,
    watch,
    formState: { errors }
  } = useFormContext<BacktestFormValues>();
  const startDate = watch('start_date');
  const endDate = watch('end_date');

  const rangeHint = useMemo(() => {
    if (!startDate || !endDate) {
      return 'Select a date range between 2001-01-01 and today.';
    }

    if (startDate > endDate) {
      return 'Start date must be on or before end date.';
    }

    return undefined;
  }, [startDate, endDate]);

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Input
        {...register('start_date')}
        type="date"
        label="Start Date"
        min="2001-01-01"
        max={toIsoDateString(new Date())}
        error={errors.start_date?.message}
      />
      <Input
        {...register('end_date')}
        type="date"
        label="End Date"
        min="2001-01-01"
        max={toIsoDateString(new Date())}
        helperText={rangeHint}
        error={errors.end_date?.message}
      />
    </div>
  );
}
