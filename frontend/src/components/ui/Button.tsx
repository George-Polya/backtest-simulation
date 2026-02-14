import { ButtonHTMLAttributes, PropsWithChildren } from 'react';

type ButtonVariant = 'primary' | 'secondary';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

const variantMap: Record<ButtonVariant, string> = {
  primary: 'bg-brand-500 text-white hover:bg-brand-700',
  secondary: 'bg-white text-slate-800 border border-slate-200 hover:bg-slate-50'
};

export function Button({ variant = 'primary', className = '', children, ...props }: PropsWithChildren<ButtonProps>) {
  return (
    <button
      className={`inline-flex items-center justify-center rounded-lg px-4 py-2 text-sm font-semibold transition ${variantMap[variant]} ${className}`}
      {...props}
    >
      {children}
    </button>
  );
}
