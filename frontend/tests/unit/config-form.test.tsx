import React from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { ConfigForm } from '@/components/backtest';

describe('ConfigForm', () => {
  it('updates strategy character counter', () => {
    render(<ConfigForm />);

    const strategyInput = screen.getByLabelText('Strategy');
    expect(screen.getByText('0/10000')).toBeInTheDocument();

    fireEvent.change(strategyInput, {
      target: {
        value: 'Trend following strategy with volatility filter'
      }
    });

    expect(screen.getByText('47/10000')).toBeInTheDocument();
  });

  it('shows validation summary on invalid submit', async () => {
    render(<ConfigForm />);

    fireEvent.click(screen.getByRole('button', { name: 'Save Configuration' }));

    await waitFor(() => {
      expect(screen.getByText('Please fix the following validation errors')).toBeInTheDocument();
    });

    expect(screen.getAllByText('Strategy must be at least 10 characters.').length).toBeGreaterThan(0);
  });

  it('submits valid payload', async () => {
    const onSubmitConfig = vi.fn();
    render(<ConfigForm onSubmitConfig={onSubmitConfig} />);

    fireEvent.change(screen.getByLabelText('Strategy'), {
      target: {
        value: 'Use a moving average crossover strategy with periodic contributions.'
      }
    });

    fireEvent.click(screen.getByRole('button', { name: 'Save Configuration' }));

    await waitFor(() => {
      expect(onSubmitConfig).toHaveBeenCalledTimes(1);
    });

    expect(onSubmitConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        strategy: expect.stringContaining('moving average crossover'),
        params: expect.objectContaining({
          initial_capital: 100000,
          benchmarks: expect.arrayContaining(['SPY']),
          llm_settings: expect.objectContaining({
            web_search_enabled: false
          })
        })
      })
    );

    expect(screen.getByText('Configuration saved')).toBeInTheDocument();
  });
});
