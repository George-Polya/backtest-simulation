'use client';

import Link from 'next/link';

export function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-[var(--border)] bg-white/90 backdrop-blur">
      <div className="flex w-full items-center justify-between px-4 py-3 md:px-8">
        <Link className="flex items-center gap-3" href="/">
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-brand-500 text-sm font-bold text-white">
            BS
          </span>
          <span className="font-[family-name:var(--font-space-grotesk)] text-lg font-bold tracking-tight text-slate-900">
            AI Backtest
          </span>
        </Link>

        <nav className="flex items-center gap-4">
          <a
            href="https://github.com/George-Polya/backtest-simulation"
            target="_blank"
            rel="noopener noreferrer"
            className="text-sm font-medium text-gray-600 hover:text-gray-900"
          >
            GitHub
          </a>
        </nav>
      </div>
    </header>
  );
}
