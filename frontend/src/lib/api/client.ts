import axios, { AxiosError } from 'axios';
import { env } from '@/lib/env';
import {
  BackendErrorCode,
  BacktestRequest,
  BacktestResultResponse,
  ExecuteBacktestRequest,
  ExecuteBacktestResponse,
  ExecutionResult,
  GenerateBacktestResponse,
  JobStatusResponse
} from '@/types';

export type ApiErrorCode = 'HTTP_ERROR' | 'NETWORK_ERROR' | 'TIMEOUT_ERROR' | 'UNKNOWN_ERROR';

export class ApiError extends Error {
  readonly status?: number;
  readonly code: ApiErrorCode;
  readonly backendErrorCode?: BackendErrorCode;
  readonly details?: unknown;

  constructor(
    message: string,
    code: ApiErrorCode,
    status?: number,
    details?: unknown,
    backendErrorCode?: BackendErrorCode
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.details = details;
    this.backendErrorCode = backendErrorCode;
  }
}

const client = axios.create({
  baseURL: env.NEXT_PUBLIC_API_BASE_URL,
  timeout: 30_000,
  headers: {
    'Content-Type': 'application/json'
  }
});

interface ExtractedError {
  message: string;
  backendErrorCode?: BackendErrorCode;
}

function isValidBackendErrorCode(value: unknown): value is BackendErrorCode {
  return typeof value === 'string' && Object.values(BackendErrorCode).includes(value as BackendErrorCode);
}

function extractErrorDetail(data: unknown): ExtractedError | undefined {
  if (typeof data === 'string' && data.trim()) {
    return { message: data };
  }

  if (typeof data === 'object' && data !== null) {
    const obj = data as Record<string, unknown>;
    const detail = obj.detail;

    // Structured detail: { message, error_code }
    if (typeof detail === 'object' && detail !== null) {
      const d = detail as Record<string, unknown>;
      const message = typeof d.message === 'string' ? d.message : undefined;
      const errorCode = isValidBackendErrorCode(d.error_code) ? d.error_code : undefined;

      if (message) {
        return { message, backendErrorCode: errorCode };
      }
    }

    // Legacy string detail
    if (typeof detail === 'string' && detail.trim()) {
      return { message: detail };
    }

    if (typeof obj.message === 'string' && obj.message.trim()) {
      return { message: obj.message };
    }
  }

  return undefined;
}

function toApiError(error: AxiosError): ApiError {
  if (error.code === 'ECONNABORTED') {
    return new ApiError('Request timeout. Please try again.', 'TIMEOUT_ERROR');
  }

  if (error.response) {
    const status = error.response.status;
    const extracted = extractErrorDetail(error.response.data);
    const message = extracted?.message ?? `Request failed with status ${status}.`;

    return new ApiError(message, 'HTTP_ERROR', status, error.response.data, extracted?.backendErrorCode);
  }

  if (error.request) {
    return new ApiError(
      'Network error. Please check your connection and API server.',
      'NETWORK_ERROR'
    );
  }

  return new ApiError(error.message || 'Unexpected API error', 'UNKNOWN_ERROR');
}

client.interceptors.response.use(
  (response) => response,
  async (error: unknown) => {
    if (axios.isAxiosError(error)) {
      return Promise.reject(toApiError(error));
    }

    if (error instanceof Error) {
      return Promise.reject(new ApiError(error.message, 'UNKNOWN_ERROR'));
    }

    return Promise.reject(new ApiError('Unexpected API error', 'UNKNOWN_ERROR'));
  }
);

export interface GenerateCodeOptions {
  signal?: AbortSignal;
  timeoutMs?: number;
}

export async function generateCode(
  payload: BacktestRequest,
  options: GenerateCodeOptions = {}
): Promise<GenerateBacktestResponse> {
  const { data } = await client.post<GenerateBacktestResponse>('/backtest/generate', payload, {
    signal: options.signal,
    timeout: options.timeoutMs
  });
  return data;
}

export async function executeBacktest(
  payload: ExecuteBacktestRequest
): Promise<ExecuteBacktestResponse> {
  const { data } = await client.post<ExecuteBacktestResponse>('/backtest/execute', payload);
  return data;
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const { data } = await client.get<JobStatusResponse>(`/backtest/status/${jobId}`);
  return data;
}

export async function getJobResult(jobId: string): Promise<ExecutionResult> {
  const { data } = await client.get<ExecutionResult>(`/backtest/result/${jobId}`);
  return data;
}

export async function getFormattedResult(jobId: string): Promise<BacktestResultResponse> {
  const { data } = await client.get<BacktestResultResponse>(`/backtest/${jobId}/result`);
  return data;
}

export { client as apiClient };
