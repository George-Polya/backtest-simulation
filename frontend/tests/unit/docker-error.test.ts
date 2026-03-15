import { ApiError } from '@/lib/api/client';
import { isDockerError, isDockerErrorCode } from '@/lib/utils';
import { BackendErrorCode } from '@/types';

describe('isDockerError', () => {
  it('returns true for ApiError with DOCKER_IMAGE_NOT_AVAILABLE error code', () => {
    const error = new ApiError(
      'Docker image not available',
      'HTTP_ERROR',
      500,
      undefined,
      BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE
    );

    expect(isDockerError(error)).toBe(true);
  });

  it('returns true for ApiError with DOCKER_CLIENT_FAILED error code', () => {
    const error = new ApiError(
      'Docker client failed',
      'HTTP_ERROR',
      500,
      undefined,
      BackendErrorCode.DOCKER_CLIENT_FAILED
    );

    expect(isDockerError(error)).toBe(true);
  });

  it('returns false for ApiError with non-Docker backend error code', () => {
    const error = new ApiError(
      'Execution failed',
      'HTTP_ERROR',
      500,
      undefined,
      BackendErrorCode.EXECUTION_FAILED
    );

    expect(isDockerError(error)).toBe(false);
  });

  it('returns false for ApiError without backend error code', () => {
    const error = new ApiError('Some error', 'HTTP_ERROR', 500);

    expect(isDockerError(error)).toBe(false);
  });

  it('returns false for plain Error even with Docker keyword in message', () => {
    expect(isDockerError(new Error('Docker image pull failed'))).toBe(false);
  });

  it('returns false for non-Error values', () => {
    expect(isDockerError(null)).toBe(false);
    expect(isDockerError(undefined)).toBe(false);
    expect(isDockerError(42)).toBe(false);
    expect(isDockerError('')).toBe(false);
    expect(isDockerError('Docker error')).toBe(false);
    expect(isDockerError({ message: 'Docker error' })).toBe(false);
  });

  it('returns false for ApiError with NETWORK_ERROR code', () => {
    const error = new ApiError('Network error', 'NETWORK_ERROR');

    expect(isDockerError(error)).toBe(false);
  });
});

describe('isDockerErrorCode', () => {
  it('returns true for DOCKER_IMAGE_NOT_AVAILABLE', () => {
    expect(isDockerErrorCode(BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE)).toBe(true);
  });

  it('returns true for DOCKER_CLIENT_FAILED', () => {
    expect(isDockerErrorCode(BackendErrorCode.DOCKER_CLIENT_FAILED)).toBe(true);
  });

  it('returns false for non-Docker error codes', () => {
    expect(isDockerErrorCode(BackendErrorCode.EXECUTION_FAILED)).toBe(false);
    expect(isDockerErrorCode(BackendErrorCode.JOB_NOT_FOUND)).toBe(false);
  });

  it('returns false for null and undefined', () => {
    expect(isDockerErrorCode(null)).toBe(false);
    expect(isDockerErrorCode(undefined)).toBe(false);
  });

  it('returns false for empty string', () => {
    expect(isDockerErrorCode('')).toBe(false);
  });
});
