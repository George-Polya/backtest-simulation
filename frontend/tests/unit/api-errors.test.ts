import { AxiosError, type InternalAxiosRequestConfig } from 'axios';
import { http, HttpResponse } from 'msw';
import { ApiError, apiClient, executeBacktest, generateCode, getFormattedResult, getJobStatus } from '@/lib/api';
import { API_BASE_URL, server } from '../mocks/server';
import { BackendErrorCode, BacktestRequest, ContributionFrequency } from '@/types';

function createRequest(): BacktestRequest {
  return {
    strategy: 'Buy SPY and hold the position for one year.',
    params: {
      start_date: '2024-01-01',
      end_date: '2024-12-31',
      initial_capital: 10000,
      contribution: {
        frequency: ContributionFrequency.Monthly,
        amount: 0
      },
      fees: {
        trading_fee_percent: 0.1,
        slippage_percent: 0.05
      },
      dividend_reinvestment: true,
      benchmarks: ['SPY'],
      llm_settings: {
        web_search_enabled: false
      }
    }
  };
}

describe('api error handling', () => {
  const originalTimeout = apiClient().defaults.timeout;

  afterEach(() => {
    apiClient().defaults.timeout = originalTimeout;
  });

  it('handles 400 bad request errors', async () => {
    server.use(
      http.post(`${API_BASE_URL}/backtest/generate`, () =>
        HttpResponse.json({ detail: 'Invalid strategy format' }, { status: 400 })
      )
    );

    await expect(generateCode(createRequest())).rejects.toMatchObject({
      name: 'ApiError',
      code: 'HTTP_ERROR',
      status: 400,
      message: 'Invalid strategy format'
    });
  });

  it('handles 401 authentication errors', async () => {
    server.use(
      http.post(`${API_BASE_URL}/backtest/generate`, () =>
        HttpResponse.json({ detail: 'Unauthorized' }, { status: 401 })
      )
    );

    await expect(generateCode(createRequest())).rejects.toMatchObject({
      code: 'HTTP_ERROR',
      status: 401,
      message: 'Unauthorized'
    });
  });

  it('handles 403 authorization errors', async () => {
    server.use(
      http.post(`${API_BASE_URL}/backtest/generate`, () =>
        HttpResponse.json({ detail: 'Forbidden' }, { status: 403 })
      )
    );

    await expect(generateCode(createRequest())).rejects.toMatchObject({
      code: 'HTTP_ERROR',
      status: 403,
      message: 'Forbidden'
    });
  });

  it('handles 404 not found errors', async () => {
    server.use(
      http.get(`${API_BASE_URL}/backtest/status/:jobId`, () =>
        HttpResponse.json({ detail: 'Job not found' }, { status: 404 })
      )
    );

    await expect(getJobStatus('missing-job')).rejects.toMatchObject({
      code: 'HTTP_ERROR',
      status: 404,
      message: 'Job not found'
    });
  });

  it('handles 500 server errors', async () => {
    server.use(
      http.get(`${API_BASE_URL}/backtest/:jobId/result`, () =>
        HttpResponse.json({ detail: 'Internal server error' }, { status: 500 })
      )
    );

    await expect(getFormattedResult('job-500')).rejects.toMatchObject({
      code: 'HTTP_ERROR',
      status: 500,
      message: 'Internal server error'
    });
  });

  it('handles network errors', async () => {
    server.use(
      http.post(`${API_BASE_URL}/backtest/execute`, () => HttpResponse.error())
    );

    await expect(
      executeBacktest({
        code: 'print("hello")',
        params: createRequest().params,
        async_mode: true
      })
    ).rejects.toMatchObject({
      code: 'NETWORK_ERROR'
    });
  });

  it('handles timeout errors', async () => {
    const originalAdapter = apiClient().defaults.adapter;
    apiClient().defaults.adapter = async (config) => {
      throw new AxiosError(
        'timeout of 30ms exceeded',
        'ECONNABORTED',
        config as InternalAxiosRequestConfig
      );
    };

    try {
      await expect(getJobStatus('slow-job')).rejects.toMatchObject({
        code: 'TIMEOUT_ERROR',
        message: 'Request timeout. Please try again.'
      });
    } finally {
      apiClient().defaults.adapter = originalAdapter;
    }
  });

  it('extracts backendErrorCode from structured detail', async () => {
    server.use(
      http.get(`${API_BASE_URL}/backtest/status/:jobId`, () =>
        HttpResponse.json(
          {
            detail: {
              message: 'Job not found: job-xyz',
              error_code: 'JOB_NOT_FOUND'
            }
          },
          { status: 404 }
        )
      )
    );

    try {
      await getJobStatus('job-xyz');
      expect.fail('Expected ApiError');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      const apiError = error as ApiError;
      expect(apiError.code).toBe('HTTP_ERROR');
      expect(apiError.status).toBe(404);
      expect(apiError.backendErrorCode).toBe(BackendErrorCode.JOB_NOT_FOUND);
      expect(apiError.message).toBe('Job not found: job-xyz');
    }
  });

  it('extracts backendErrorCode for Docker errors', async () => {
    server.use(
      http.post(`${API_BASE_URL}/backtest/execute`, () =>
        HttpResponse.json(
          {
            detail: {
              message: "Docker image 'backtest-runner:latest' is not available.",
              error_code: 'DOCKER_IMAGE_NOT_AVAILABLE'
            }
          },
          { status: 500 }
        )
      )
    );

    try {
      await executeBacktest({
        code: 'print("hello")',
        params: createRequest().params,
        async_mode: true
      });
      expect.fail('Expected ApiError');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      const apiError = error as ApiError;
      expect(apiError.backendErrorCode).toBe(BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE);
    }
  });

  it('handles legacy string detail without backendErrorCode', async () => {
    server.use(
      http.get(`${API_BASE_URL}/backtest/status/:jobId`, () =>
        HttpResponse.json({ detail: 'Legacy error message' }, { status: 400 })
      )
    );

    try {
      await getJobStatus('job-legacy');
      expect.fail('Expected ApiError');
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      const apiError = error as ApiError;
      expect(apiError.message).toBe('Legacy error message');
      expect(apiError.backendErrorCode).toBeUndefined();
    }
  });
});
