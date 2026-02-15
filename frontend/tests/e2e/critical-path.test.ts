import { expect, test } from '@playwright/test';

const jobId = 'job-e2e-100';

test('critical path: input -> generate -> execute -> result -> export', async ({ page }) => {
  let statusCallCount = 0;

  await page.route('**/api/v1/backtest/generate', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        generated_code: {
          code: 'def run_backtest(params):\\n    return {"status": "ok"}',
          strategy_summary: 'E2E momentum strategy summary.',
          model_info: {
            provider: 'openrouter',
            model_id: 'anthropic/claude-3.5-sonnet',
            max_tokens: 8000,
            supports_system_prompt: true,
            cost_per_1k_input: 0.003,
            cost_per_1k_output: 0.015
          },
          tickers: ['SPY', 'QQQ']
        },
        tickers_found: ['SPY', 'QQQ'],
        generation_time_seconds: 1.25
      })
    });
  });

  await page.route('**/api/v1/backtest/execute', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        job_id: jobId,
        status: 'pending',
        message: 'submitted',
        result: null
      })
    });
  });

  await page.route('**/api/v1/backtest/status/**', async (route) => {
    statusCallCount += 1;

    const status =
      statusCallCount === 1
        ? 'pending'
        : statusCallCount === 2
          ? 'running'
          : 'completed';

    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        job_id: jobId,
        status
      })
    });
  });

  await page.route('**/api/v1/backtest/result/**', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        job_id: jobId,
        status: 'completed',
        data: {
          equity_series: [{ date: '2024-01-01', value: 10000 }]
        },
        error: null,
        logs: 'execution done',
        duration_seconds: 2.4
      })
    });
  });

  await page.route('**/api/v1/backtest/**/result', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        job_id: jobId,
        status: 'completed',
        metrics: {
          total_return: 12.4,
          cagr: 7.8,
          max_drawdown: 4.2,
          sharpe_ratio: 1.3,
          sortino_ratio: 1.7,
          calmar_ratio: 0.9,
          volatility: 11.5,
          total_trades: 7,
          winning_trades: 4,
          losing_trades: 3,
          win_rate: 57.1
        },
        equity_curve: {
          strategy: [
            { date: '2024-01-01', value: 10000 },
            { date: '2024-12-31', value: 11240 }
          ],
          benchmark: [
            { date: '2024-01-01', value: 10000 },
            { date: '2024-12-31', value: 10980 }
          ],
          log_scale: false
        },
        drawdown: {
          data: [
            { date: '2024-01-01', value: 0 },
            { date: '2024-08-01', value: -3.1 }
          ]
        },
        monthly_heatmap: {
          years: [2024],
          months: ['Jan', 'Feb'],
          returns: [[1.2, -0.4]]
        },
        trades: [{ date: '2024-06-01', symbol: 'SPY', action: 'BUY', profit: 0 }],
        logs: 'formatted result'
      })
    });
  });

  await page.goto('/');

  await page.getByRole('button', { name: 'Collapse Config' }).click();
  await expect(page.getByRole('button', { name: 'Expand Config' })).toBeVisible();
  await page.getByRole('button', { name: 'Expand Config' }).click();
  await expect(page.getByRole('button', { name: 'Collapse Config' })).toBeVisible();

  const strategyText = 'Momentum strategy with monthly rebalance and volatility filter.';
  const strategyInput = page.locator('#strategy');

  await strategyInput.click();
  await strategyInput.fill(strategyText);
  await expect(strategyInput).toHaveValue(strategyText);
  await page.locator('#start_date').fill('2024-01-01');
  await page.locator('#end_date').fill('2024-12-31');
  await page.getByRole('button', { name: 'Save Configuration' }).click();
  await expect(page.getByRole('button', { name: 'Generate Code' })).toBeEnabled();

  await page.getByRole('button', { name: 'Generate Code' }).click();
  await expect(page.getByText('E2E momentum strategy summary.')).toBeVisible();

  await page.getByRole('button', { name: 'Execute Backtest' }).click();

  await expect(page.getByText('Backtest pending')).toBeVisible();
  await expect(page.getByText('Backtest running')).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText('Backtest completed')).toBeVisible({ timeout: 20_000 });

  await expect(page.getByText('Performance Metrics')).toBeVisible();
  await expect(page.getByRole('button', { name: 'Export Equity' })).toBeEnabled();

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Export Equity' }).click()
  ]);

  expect(download.suggestedFilename()).toContain('backtest_equity_job-e2e-100_');
});
