import { z } from 'zod';
import { ContributionFrequency } from '@/types';

const MIN_START_DATE_KEY = Date.UTC(2001, 0, 1);

export const STRATEGY_MIN_LENGTH = 10;
export const STRATEGY_MAX_LENGTH = 10000;

function toDateKey(date: Date): number {
  return Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate());
}

export function toIsoDateString(date: Date): string {
  return date.toISOString().slice(0, 10);
}

function todayDateKey(): number {
  return toDateKey(new Date());
}

function dateField(label: string) {
  return z.preprocess(
    (value) => (typeof value === 'string' && value.trim() === '' ? undefined : value),
    z.coerce.date({
      required_error: `${label} is required.`,
      invalid_type_error: `${label} must be a valid date.`
    })
  );
}

function numberField(
  label: string,
  options: {
    min?: number;
    max?: number;
    minMessage?: string;
    maxMessage?: string;
  } = {}
) {
  return z
    .coerce.number({
      invalid_type_error: `${label} must be a number.`
    })
    .refine((value) => Number.isFinite(value), {
      message: `${label} must be a number.`
    })
    .refine((value) => (options.min === undefined ? true : value >= options.min), {
      message: options.minMessage || `${label} must be at least ${options.min}.`
    })
    .refine((value) => (options.max === undefined ? true : value <= options.max), {
      message: options.maxMessage || `${label} must be at most ${options.max}.`
    });
}

const startDateSchema = dateField('Start date')
  .refine((date) => toDateKey(date) >= MIN_START_DATE_KEY, {
    message: 'Start date must be on or after 2001-01-01.'
  })
  .transform((date) => toIsoDateString(date));

const endDateSchema = dateField('End date')
  .refine((date) => toDateKey(date) <= todayDateKey(), {
    message: 'End date cannot be later than today.'
  })
  .transform((date) => toIsoDateString(date));

export const backtestFormSchema = z
  .object({
    strategy: z
      .string({ required_error: 'Strategy is required.' })
      .trim()
      .min(STRATEGY_MIN_LENGTH, `Strategy must be at least ${STRATEGY_MIN_LENGTH} characters.`)
      .max(STRATEGY_MAX_LENGTH, `Strategy cannot exceed ${STRATEGY_MAX_LENGTH} characters.`),
    start_date: startDateSchema,
    end_date: endDateSchema,
    initial_capital: numberField('Initial capital', {
      min: 1000,
      minMessage: 'Initial capital must be at least 1000.'
    }),
    benchmarks: z.array(z.string().trim().min(1)).min(1, 'Select at least one benchmark.'),
    contribution: z.object({
      frequency: z.nativeEnum(ContributionFrequency, {
        errorMap: () => ({ message: 'Contribution frequency is required.' })
      }),
      amount: numberField('Contribution amount', {
        min: 0,
        minMessage: 'Contribution amount must be 0 or greater.'
      })
    }),
    fees: z.object({
      trading_fee_percent: numberField('Trading fee', {
        min: 0,
        max: 10,
        minMessage: 'Trading fee must be between 0 and 10%.',
        maxMessage: 'Trading fee must be between 0 and 10%.'
      }),
      slippage_percent: numberField('Slippage', {
        min: 0,
        max: 10,
        minMessage: 'Slippage must be between 0 and 10%.',
        maxMessage: 'Slippage must be between 0 and 10%.'
      })
    }),
    dividend_reinvestment: z.boolean(),
    llm_settings: z.object({
      web_search_enabled: z.boolean()
    })
  })
  .superRefine((values, ctx) => {
    if (values.start_date > values.end_date) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ['end_date'],
        message: 'End date must be on or after start date.'
      });
    }
  });

export type BacktestFormValues = z.output<typeof backtestFormSchema>;

export function createDefaultBacktestFormValues(today: Date = new Date()): BacktestFormValues {
  return {
    strategy: '',
    start_date: '2020-01-01',
    end_date: toIsoDateString(today),
    initial_capital: 100000,
    benchmarks: ['SPY'],
    contribution: {
      frequency: ContributionFrequency.Monthly,
      amount: 0
    },
    fees: {
      trading_fee_percent: 0,
      slippage_percent: 0
    },
    dividend_reinvestment: true,
    llm_settings: {
      web_search_enabled: false
    }
  };
}
