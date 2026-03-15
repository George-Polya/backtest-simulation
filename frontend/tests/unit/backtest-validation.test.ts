import {
  backtestFormSchema,
  createDefaultBacktestFormValues,
  toIsoDateString
} from '@/lib/validations';

function createValidForm() {
  return {
    ...createDefaultBacktestFormValues(),
    strategy: 'Use monthly momentum with risk controls and benchmark-relative rebalancing.'
  };
}

describe('backtestFormSchema', () => {
  it('accepts a valid form payload', () => {
    const result = backtestFormSchema.safeParse(createValidForm());

    expect(result.success).toBe(true);
    if (result.success) {
      expect(result.data.strategy).toContain('momentum');
      expect(result.data.initial_capital).toBeGreaterThanOrEqual(1000);
    }
  });

  it('rejects strategy shorter than 10 characters', () => {
    const result = backtestFormSchema.safeParse({
      ...createValidForm(),
      strategy: 'too short'
    });

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.flatten().fieldErrors.strategy?.[0]).toContain('at least 10 characters');
    }
  });

  it('rejects start date earlier than 2001-01-01', () => {
    const result = backtestFormSchema.safeParse({
      ...createValidForm(),
      start_date: '2000-12-31'
    });

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.flatten().fieldErrors.start_date?.[0]).toContain('2001-01-01');
    }
  });

  it('rejects end date later than today', () => {
    const tomorrow = new Date(Date.now() + 24 * 60 * 60 * 1000);
    const result = backtestFormSchema.safeParse({
      ...createValidForm(),
      end_date: toIsoDateString(tomorrow)
    });

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.flatten().fieldErrors.end_date?.[0]).toContain('later than today');
    }
  });

  it('rejects start date after end date', () => {
    const result = backtestFormSchema.safeParse({
      ...createValidForm(),
      start_date: '2024-01-10',
      end_date: '2024-01-01'
    });

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.flatten().fieldErrors.end_date?.[0]).toContain('on or after start date');
    }
  });

  it('rejects empty benchmark selection', () => {
    const result = backtestFormSchema.safeParse({
      ...createValidForm(),
      benchmarks: []
    });

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.flatten().fieldErrors.benchmarks?.[0]).toContain('at least one benchmark');
    }
  });

  it('rejects fees outside 0-10% range', () => {
    const result = backtestFormSchema.safeParse({
      ...createValidForm(),
      fees: {
        trading_fee_percent: 12,
        slippage_percent: -1
      }
    });

    expect(result.success).toBe(false);
    if (!result.success) {
      expect(result.error.issues.some((issue) => issue.path.join('.') === 'fees.trading_fee_percent')).toBe(true);
      expect(result.error.issues.some((issue) => issue.path.join('.') === 'fees.slippage_percent')).toBe(true);
    }
  });
});
