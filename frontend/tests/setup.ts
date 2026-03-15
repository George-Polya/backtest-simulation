import '@testing-library/jest-dom/vitest';
import { afterAll, afterEach, beforeAll } from 'vitest';
import { vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import { server } from './mocks/server';

process.env.NEXT_PUBLIC_API_BASE_URL ??= 'http://localhost:8000/api/v1';

class ResizeObserverMock {
  observe() {
    return undefined;
  }

  unobserve() {
    return undefined;
  }

  disconnect() {
    return undefined;
  }
}

vi.stubGlobal('ResizeObserver', ResizeObserverMock);

beforeAll(() => {
  server.listen({ onUnhandledRequest: 'error' });
});

afterEach(() => {
  server.resetHandlers();
  cleanup();
});

afterAll(() => {
  server.close();
});
