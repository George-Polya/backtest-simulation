import { BackendErrorCode } from '@/types';
import { ApiError } from '@/lib/api/client';

const DOCKER_ERROR_CODES = new Set<BackendErrorCode>([
  BackendErrorCode.DOCKER_IMAGE_NOT_AVAILABLE,
  BackendErrorCode.DOCKER_CLIENT_FAILED
]);

/**
 * Determines whether the given error originates from Docker infrastructure.
 *
 * Uses the structured `backendErrorCode` returned by the API.
 * Docker Desktop's Resource Saver mode can cause the first container request
 * to fail with transient errors that are worth retrying.
 */
export function isDockerError(error: unknown): boolean {
  if (error instanceof ApiError && error.backendErrorCode) {
    return DOCKER_ERROR_CODES.has(error.backendErrorCode);
  }

  return false;
}

/**
 * Determines whether an error_code string represents a Docker error.
 *
 * Useful for checking ExecutionResult.error_code from polling results,
 * where the error is not an ApiError instance but a plain string code.
 */
export function isDockerErrorCode(errorCode: string | null | undefined): boolean {
  if (!errorCode) {
    return false;
  }

  return DOCKER_ERROR_CODES.has(errorCode as BackendErrorCode);
}
