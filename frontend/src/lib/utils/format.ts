interface NumberFormatOptions {
  minimumFractionDigits?: number;
  maximumFractionDigits?: number;
  signed?: boolean;
}

function resolveFractionDigits(options: NumberFormatOptions): {
  minimumFractionDigits: number;
  maximumFractionDigits: number;
} {
  const hasMin = typeof options.minimumFractionDigits === 'number';
  const hasMax = typeof options.maximumFractionDigits === 'number';

  let minimumFractionDigits = hasMin ? options.minimumFractionDigits ?? 0 : 2;
  let maximumFractionDigits = hasMax ? options.maximumFractionDigits ?? 0 : 2;

  if (!hasMin && hasMax && maximumFractionDigits < minimumFractionDigits) {
    minimumFractionDigits = maximumFractionDigits;
  }

  if (minimumFractionDigits > maximumFractionDigits) {
    maximumFractionDigits = minimumFractionDigits;
  }

  return {
    minimumFractionDigits,
    maximumFractionDigits
  };
}

export function formatNumber(value: number, options: NumberFormatOptions = {}): string {
  const { signed = false } = options;
  const { minimumFractionDigits, maximumFractionDigits } = resolveFractionDigits(options);
  const formatter = new Intl.NumberFormat('en-US', {
    minimumFractionDigits,
    maximumFractionDigits
  });

  if (signed && value > 0) {
    return `+${formatter.format(value)}`;
  }

  return formatter.format(value);
}

export function formatPercentage(value: number, options: NumberFormatOptions = {}): string {
  return `${formatNumber(value, options)}%`;
}

export function formatCurrency(value: number, options: NumberFormatOptions = {}): string {
  const { signed = false } = options;
  const { minimumFractionDigits, maximumFractionDigits } = resolveFractionDigits(options);
  const formatter = new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits,
    maximumFractionDigits
  });

  if (signed && value > 0) {
    return `+${formatter.format(value)}`;
  }

  return formatter.format(value);
}

export function formatDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }

  return new Intl.DateTimeFormat('en-US', {
    year: 'numeric',
    month: 'short',
    day: '2-digit'
  }).format(date);
}

// Legacy formatter used by older tests/components where decimal input is expected.
export function formatPercent(value: number): string {
  return formatPercentage(value * 100);
}
