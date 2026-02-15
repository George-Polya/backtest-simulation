'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { BacktestWorkspace } from '@/components/backtest';

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000/api/v1';

export default function WorkspacePage() {
  const router = useRouter();
  const [isAuthorized, setIsAuthorized] = useState(false);

  useEffect(() => {
    let isCancelled = false;

    const verifySession = async () => {
      const accessToken = window.localStorage.getItem('accessToken');

      if (!accessToken) {
        router.replace('/login');
        return;
      }

      try {
        const response = await fetch(`${API_BASE_URL}/auth/me`, {
          headers: {
            Authorization: `Bearer ${accessToken}`
          }
        });

        if (!response.ok) {
          window.localStorage.removeItem('accessToken');
          window.localStorage.removeItem('refreshToken');
          router.replace('/login');
          return;
        }

        if (!isCancelled) {
          setIsAuthorized(true);
        }
      } catch {
        window.localStorage.removeItem('accessToken');
        window.localStorage.removeItem('refreshToken');
        router.replace('/login');
      }
    };

    void verifySession();

    return () => {
      isCancelled = true;
    };
  }, [router]);

  if (!isAuthorized) {
    return (
      <main className="flex min-h-screen w-full items-center justify-center p-4 md:p-6 lg:p-8">
        <p className="text-sm text-slate-600">Checking authentication...</p>
      </main>
    );
  }

  return (
    <main className="flex min-h-screen w-full flex-col p-4 md:p-6 lg:p-8">
      <BacktestWorkspace />
    </main>
  );
}
