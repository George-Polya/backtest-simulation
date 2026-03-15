import { PropsWithChildren } from 'react';

type AlertVariant = 'info' | 'success' | 'error' | 'warning';

interface AlertProps {
  variant?: AlertVariant;
  title?: string;
  className?: string;
}

const variantStyles: Record<AlertVariant, string> = {
  info: 'border-blue-200 bg-blue-50 text-blue-900',
  success: 'border-emerald-200 bg-emerald-50 text-emerald-900',
  error: 'border-red-200 bg-red-50 text-red-900',
  warning: 'border-amber-200 bg-amber-50 text-amber-900'
};

const variantIcon: Record<AlertVariant, string> = {
  info: 'i',
  success: 'v',
  error: '!',
  warning: '!'
};

export function Alert({ variant = 'info', title, className = '', children }: PropsWithChildren<AlertProps>) {
  return (
    <div className={`rounded-lg border px-3 py-2 text-sm ${variantStyles[variant]} ${className}`} role="alert">
      <div className="flex items-start gap-2">
        <span className="mt-[2px] inline-flex h-4 w-4 items-center justify-center rounded-full border border-current text-[10px] font-bold">
          {variantIcon[variant]}
        </span>
        <div>
          {title ? <p className="font-semibold">{title}</p> : null}
          {children ? <div className="mt-1">{children}</div> : null}
        </div>
      </div>
    </div>
  );
}
