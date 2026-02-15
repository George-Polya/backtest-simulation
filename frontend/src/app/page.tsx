import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="relative min-h-screen overflow-hidden">
      <div className="pointer-events-none absolute inset-0">
        <div className="absolute -left-20 top-16 h-72 w-72 rounded-full bg-brand-100/70 blur-3xl" />
        <div className="absolute right-0 top-40 h-80 w-80 rounded-full bg-sky-100/60 blur-3xl" />
        <div className="absolute bottom-0 left-1/3 h-64 w-64 rounded-full bg-cyan-100/70 blur-3xl" />
      </div>

      <section className="relative mx-auto flex w-full max-w-6xl flex-col gap-10 px-4 py-14 md:px-8 md:py-20">
        <p className="w-fit rounded-full border border-brand-100 bg-white/90 px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-brand-700">
          AI Backtesting
        </p>

        <div className="grid items-start gap-8 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="space-y-6">
            <h1 className="font-[family-name:var(--font-space-grotesk)] text-4xl font-bold leading-tight text-slate-900 md:text-6xl">
              Turn strategy ideas into testable code in minutes.
            </h1>
            <p className="max-w-2xl text-base leading-relaxed text-slate-600 md:text-lg">
              AI Geneated Backtest transforms plain-language investment ideas into executable strategies, then
              visualizes performance with institutional-grade metrics.
            </p>
            <div className="flex flex-wrap items-center gap-3">
              <Link
                className="rounded-lg bg-brand-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-brand-700"
                href="/workspace"
              >
                Open Workspace
              </Link>
              <Link
                className="rounded-lg border border-[var(--border)] bg-white px-5 py-3 text-sm font-semibold text-slate-700 transition hover:border-brand-100 hover:text-brand-700"
                href="/login"
              >
                Sign In
              </Link>
            </div>
          </div>

          <div className="grid gap-3">
            <article className="rounded-2xl border border-[var(--border)] bg-white/90 p-5 shadow-sm backdrop-blur">
              <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-brand-700">Workflow</h2>
              <p className="mt-3 text-sm leading-relaxed text-slate-600">
                1) Save config 2) Generate code 3) Execute backtest 4) Inspect metrics and export.
              </p>
            </article>
            <article className="rounded-2xl border border-[var(--border)] bg-white/90 p-5 shadow-sm backdrop-blur">
              <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-brand-700">Outputs</h2>
              <p className="mt-3 text-sm leading-relaxed text-slate-600">
                Equity curve, drawdown, monthly heatmap, full trade table, and CSV exports.
              </p>
            </article>
            <article className="rounded-2xl border border-[var(--border)] bg-white/90 p-5 shadow-sm backdrop-blur">
              <h2 className="text-sm font-semibold uppercase tracking-[0.14em] text-brand-700">Execution</h2>
              <p className="mt-3 text-sm leading-relaxed text-slate-600">
                Async jobs with polling and structured status updates for production-like workflow.
              </p>
            </article>
          </div>
        </div>
      </section>
    </main>
  );
}
