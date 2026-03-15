'use client';

import { useMemo } from 'react';
import { FieldErrors, useFormContext } from 'react-hook-form';
import { Alert } from '@/components/ui';
import { BacktestFormValues } from '@/lib/validations';

function collectMessages(errors: FieldErrors<BacktestFormValues>): string[] {
  const messages = new Set<string>();
  const visited = new WeakSet<object>();

  function traverse(node: unknown) {
    if (!node || typeof node !== 'object') {
      return;
    }

    if (visited.has(node)) {
      return;
    }
    visited.add(node);

    if (
      (typeof HTMLElement !== 'undefined' && node instanceof HTMLElement) ||
      (typeof HTMLInputElement !== 'undefined' && node instanceof HTMLInputElement) ||
      (typeof HTMLTextAreaElement !== 'undefined' && node instanceof HTMLTextAreaElement)
    ) {
      return;
    }

    const candidate = node as { message?: unknown };
    if (typeof candidate.message === 'string' && candidate.message.trim()) {
      messages.add(candidate.message);
    }

    for (const [key, value] of Object.entries(node)) {
      if (key === 'ref') {
        continue;
      }
      traverse(value);
    }
  }

  traverse(errors);
  return Array.from(messages);
}

export function ValidationSummary() {
  const {
    formState: { errors, submitCount }
  } = useFormContext<BacktestFormValues>();

  const messages = useMemo(() => collectMessages(errors), [errors]);

  if (submitCount === 0 || messages.length === 0) {
    return null;
  }

  return (
    <Alert title="Please fix the following validation errors" variant="error">
      <ul className="list-disc space-y-1 pl-4 text-xs">
        {messages.map((message) => (
          <li key={message}>{message}</li>
        ))}
      </ul>
    </Alert>
  );
}
