interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  centered?: boolean;
  className?: string;
}

const sizeMap: Record<NonNullable<LoadingSpinnerProps['size']>, string> = {
  sm: 'h-4 w-4 border-2',
  md: 'h-6 w-6 border-2',
  lg: 'h-10 w-10 border-4'
};

export function LoadingSpinner({ size = 'md', centered = false, className = '' }: LoadingSpinnerProps) {
  const spinner = (
    <span
      className={`inline-block animate-spin rounded-full border-brand-500 border-t-transparent ${sizeMap[size]} ${className}`}
      aria-label="Loading"
      role="status"
    />
  );

  if (centered) {
    return <div className="flex w-full items-center justify-center py-4">{spinner}</div>;
  }

  return spinner;
}
