import { forwardRef, TextareaHTMLAttributes } from 'react';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string;
  helperText?: string;
  showCount?: boolean;
  currentLength?: number;
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(function Textarea(
  {
    label,
    error,
    helperText,
    showCount = false,
    currentLength,
    className = '',
    rows = 4,
    id,
    maxLength,
    ...props
  },
  ref
) {
  const textareaId = id ?? props.name;
  const descriptionId = textareaId ? `${textareaId}-description` : undefined;
  const resolvedLength = currentLength ?? (typeof props.value === 'string' ? props.value.length : 0);

  return (
    <div className="w-full">
      {label ? (
        <label className="mb-1 block text-sm font-medium text-slate-700" htmlFor={textareaId}>
          {label}
        </label>
      ) : null}
      <textarea
        ref={ref}
        id={textareaId}
        rows={rows}
        maxLength={maxLength}
        aria-invalid={Boolean(error)}
        aria-describedby={descriptionId}
        className={`w-full rounded-lg border bg-white px-3 py-2 text-sm text-slate-900 outline-none transition placeholder:text-slate-400 focus:ring-2 ${
          error
            ? 'border-red-400 focus:border-red-500 focus:ring-red-100'
            : 'border-[var(--border)] focus:border-brand-500 focus:ring-brand-100'
        } disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-500 ${className}`}
        {...props}
      />
      <div className="mt-1 flex items-center justify-between gap-2" id={descriptionId}>
        <p className={`text-xs ${error ? 'text-red-600' : 'text-slate-500'}`}>
          {error || helperText || '\u00a0'}
        </p>
        {showCount && typeof maxLength === 'number' ? (
          <p className="text-xs text-slate-500">
            {resolvedLength}/{maxLength}
          </p>
        ) : null}
      </div>
    </div>
  );
});
