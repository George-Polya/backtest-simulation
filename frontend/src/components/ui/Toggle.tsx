import { forwardRef, InputHTMLAttributes } from 'react';

export interface ToggleProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label: string;
  description?: string;
  error?: string;
}

export const Toggle = forwardRef<HTMLInputElement, ToggleProps>(function Toggle(
  { label, description, error, className = '', id, ...props },
  ref
) {
  const inputId = id ?? props.name;
  const descriptionId = inputId ? `${inputId}-description` : undefined;

  return (
    <div className={`rounded-lg border border-[var(--border)] bg-white px-3 py-2 ${className}`}>
      <label className="flex cursor-pointer items-start justify-between gap-3" htmlFor={inputId}>
        <div>
          <p className="text-sm font-medium text-slate-800">{label}</p>
          {description ? (
            <p className="mt-1 text-xs text-slate-500" id={descriptionId}>
              {description}
            </p>
          ) : null}
        </div>
        <span className="relative inline-flex h-6 w-11 shrink-0 items-center">
          <input
            ref={ref}
            id={inputId}
            type="checkbox"
            role="switch"
            aria-invalid={Boolean(error)}
            aria-describedby={descriptionId}
            className="peer sr-only"
            {...props}
          />
          <span className="absolute inset-0 rounded-full bg-slate-300 transition peer-checked:bg-brand-500 peer-focus-visible:ring-2 peer-focus-visible:ring-brand-200" />
          <span className="absolute left-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition peer-checked:translate-x-5" />
        </span>
      </label>
      {error ? <p className="mt-2 text-xs text-red-600">{error}</p> : null}
    </div>
  );
});
