import { BacktestWorkspace } from '@/components/backtest';

export default function HomePage() {
  return (
    <main className="flex min-h-screen w-full flex-col p-4 md:p-6 lg:p-8">
      <BacktestWorkspace />
    </main>
  );
}
