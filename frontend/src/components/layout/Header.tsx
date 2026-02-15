'use client';

import Link from 'next/link';
import { useAuthStore } from '@/stores';
import { LogoutButton } from '@/components/auth';

export function Header() {
  const { isAuthenticated, user, isLoading } = useAuthStore();

  return (
    <header className="sticky top-0 z-30 border-b border-[var(--border)] bg-white/90 backdrop-blur">
      <div className="flex w-full items-center justify-between px-4 py-3 md:px-8">
        <Link className="flex items-center gap-3" href="/">
          <span className="inline-flex h-9 w-9 items-center justify-center rounded-lg bg-brand-500 text-sm font-bold text-white">
            BS
          </span>
          <span className="font-[family-name:var(--font-space-grotesk)] text-lg font-bold tracking-tight text-slate-900">
            AI Geneated Backtest
          </span>
        </Link>

        <div className="flex items-center gap-4">
          {isLoading ? (
            <span className="text-sm text-gray-500">Loading...</span>
          ) : isAuthenticated && user ? (
            <div className="flex items-center gap-4">
              <span className="text-sm text-gray-600">{user.email}</span>
              <LogoutButton />
            </div>
          ) : (
            <div className="flex items-center gap-4">
              <Link
                href="/login"
                className="text-sm font-medium text-gray-600 hover:text-gray-900"
              >
                Login
              </Link>
              <Link
                href="/register"
                className="text-sm font-medium text-white bg-blue-600 px-3 py-1.5 rounded hover:bg-blue-700"
              >
                Sign Up
              </Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
