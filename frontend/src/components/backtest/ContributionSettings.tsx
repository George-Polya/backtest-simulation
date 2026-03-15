'use client';

import { useFormContext } from 'react-hook-form';
import { Input, Select, SelectOption } from '@/components/ui';
import { BacktestFormValues } from '@/lib/validations';

const contributionOptions: SelectOption[] = [
  { value: 'monthly', label: 'Monthly' },
  { value: 'quarterly', label: 'Quarterly' },
  { value: 'semiannual', label: 'Semiannual' },
  { value: 'annual', label: 'Annual' }
];

export function ContributionSettings() {
  const {
    register,
    formState: { errors }
  } = useFormContext<BacktestFormValues>();
  const contributionErrors = errors.contribution as
    | { frequency?: { message?: string }; amount?: { message?: string } }
    | undefined;

  return (
    <div className="grid gap-3 sm:grid-cols-2">
      <Select
        {...register('contribution.frequency')}
        label="Contribution Frequency"
        options={contributionOptions}
        error={contributionErrors?.frequency?.message}
      />
      <Input
        {...register('contribution.amount', { valueAsNumber: true })}
        type="number"
        min={0}
        step={100}
        label="Contribution Amount"
        helperText="Set to 0 if no periodic contribution is planned."
        error={contributionErrors?.amount?.message}
      />
    </div>
  );
}
