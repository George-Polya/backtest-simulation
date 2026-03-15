import { PropsWithChildren } from 'react';

interface CardProps {
  className?: string;
  title?: string;
}

export function Card({ className = '', title, children }: PropsWithChildren<CardProps>) {
  return (
    <section className={`rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 shadow-sm ${className}`}>
      {title ? <h2 className="mb-3 text-base font-semibold text-slate-900">{title}</h2> : null}
      {children}
    </section>
  );
}
