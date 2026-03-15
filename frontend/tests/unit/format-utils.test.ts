import { formatCurrency, formatNumber, formatPercentage } from '@/lib/utils';

describe('format utils', () => {
  it('supports maximumFractionDigits below defaults without throwing', () => {
    expect(formatPercentage(1.234, { maximumFractionDigits: 1 })).toBe('1.2%');
    expect(formatNumber(1.234, { maximumFractionDigits: 1 })).toBe('1.2');
  });

  it('formats currency with sign and custom digits', () => {
    expect(formatCurrency(1200.25, { signed: true, maximumFractionDigits: 0 })).toBe('+$1,200');
    expect(formatCurrency(-42.8, { maximumFractionDigits: 1 })).toBe('-$42.8');
  });
});
