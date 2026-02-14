import { forwardRef, InputHTMLAttributes } from 'react';

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
  helperText?: string;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(function Input(
  { label, error, helperText, className = '', id, ...props },
  ref
) {
  const inputId = id ?? props.name;
  const descriptionId = inputId ? `${inputId}-description` : undefined;

  return (
    <div className="w-full">
      {label ? (
        <label className="mb-1 block text-sm font-medium text-slate-700" htmlFor={inputId}>
          {label}
        </label>
      ) : null}
      <input
        ref={ref}
        id={inputId}
        aria-invalid={Boolean(error)}
        aria-describedby={descriptionId}
        className={`w-full rounded-lg border bg-white px-3 py-2 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:ring-2 ${
          error
            ? 'border-red-400 focus:border-red-500 focus:ring-red-100'
            : 'border-[var(--border)] focus:border-brand-500 focus:ring-brand-100'
        } disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-500 ${className}`}
        {...props}
      />
      {error ? (
        <p className="mt-1 text-xs text-red-600" id={descriptionId}>
          {error}
        </p>
      ) : helperText ? (
        <p className="mt-1 text-xs text-slate-500" id={descriptionId}>
          {helperText}
        </p>
      ) : null}
    </div>
  );
});
