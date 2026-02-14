'use client';

import Link from 'next/link';
import { useState } from 'react';

const navItems = [
  { href: '#workspace', label: 'Workspace' },
  { href: '#code-panel', label: 'Code' },
  { href: '#results-panel', label: 'Results' }
];

export function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-30 border-b border-[var(--border)] bg-white/90 backdrop-blur">
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-3 md:px-8">
        <Link className="flex items-center gap-3" href="/">
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-brand-500 text-sm font-bold text-white">
            BS
          </span>
          <span className="font-[family-name:var(--font-space-grotesk)] text-lg font-bold tracking-tight text-slate-900">
            Backtest Simulation
          </span>
        </Link>

        <button
          aria-controls="main-nav"
          aria-expanded={isMenuOpen}
          className="rounded-lg border border-[var(--border)] px-3 py-1 text-sm font-medium text-slate-700 md:hidden"
          onClick={() => setIsMenuOpen((current) => !current)}
          type="button"
        >
          Menu
        </button>

        <nav className="hidden items-center gap-4 md:flex" id="main-nav">
          {navItems.map((item) => (
            <a className="text-sm font-medium text-slate-600 transition hover:text-brand-700" href={item.href} key={item.href}>
              {item.label}
            </a>
          ))}
        </nav>
      </div>

      {isMenuOpen ? (
        <nav className="border-t border-[var(--border)] bg-white px-4 py-3 md:hidden" id="main-nav-mobile">
          <ul className="flex flex-col gap-2">
            {navItems.map((item) => (
              <li key={item.href}>
                <a
                  className="block rounded-md px-2 py-1 text-sm font-medium text-slate-700 hover:bg-slate-100"
                  href={item.href}
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item.label}
                </a>
              </li>
            ))}
          </ul>
        </nav>
      ) : null}
    </header>
  );
}
