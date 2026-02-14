import { BacktestWorkspace } from '@/components/backtest';

export default function HomePage() {
  return (
    <main className="mx-auto flex min-h-screen w-full max-w-7xl flex-col p-4 md:p-8">
      <BacktestWorkspace />
    </main>
  );
}
