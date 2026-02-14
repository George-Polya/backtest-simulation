import { setupServer } from 'msw/node';

export const API_BASE_URL = 'http://localhost:8000/api/v1';
export const server = setupServer();
