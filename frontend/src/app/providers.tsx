'use client';

import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PropsWithChildren, useEffect, useState } from 'react';
import { useAuthStore } from '@/stores';

export function Providers({ children }: PropsWithChildren) {
  const initializeAuth = useAuthStore((state) => state.initialize);
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            retry: 1,
            staleTime: 30_000
          }
        }
      })
  );

  useEffect(() => {
    void initializeAuth();
  }, [initializeAuth]);

  return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
}
