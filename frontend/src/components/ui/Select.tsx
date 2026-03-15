import { forwardRef, SelectHTMLAttributes } from 'react';

export interface SelectOption {
  value: string;
  label: string;
}

export interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  helperText?: string;
  options: SelectOption[];
  placeholder?: string;
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(function Select(
  { label, error, helperText, className = '', id, options, placeholder, ...props },
  ref
) {
  const selectId = id ?? props.name;
  const descriptionId = selectId ? `${selectId}-description` : undefined;

  return (
    <div className="w-full">
      {label ? (
        <label className="mb-1 block text-sm font-medium text-slate-700" htmlFor={selectId}>
          {label}
        </label>
      ) : null}
      <select
        ref={ref}
        id={selectId}
        aria-invalid={Boolean(error)}
        aria-describedby={descriptionId}
        className={`w-full rounded-lg border bg-white px-3 py-2 text-sm text-slate-900 outline-none transition focus:ring-2 ${
          error
            ? 'border-red-400 focus:border-red-500 focus:ring-red-100'
            : 'border-[var(--border)] focus:border-brand-500 focus:ring-brand-100'
        } disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-500 ${className}`}
        {...props}
      >
        {placeholder ? (
          <option value="" disabled>
            {placeholder}
          </option>
        ) : null}
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
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
