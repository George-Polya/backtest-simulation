import { z } from 'zod';

const clientEnvSchema = z.object({
  NEXT_PUBLIC_API_BASE_URL: z.string().url()
});

const parsedEnv = clientEnvSchema.safeParse({
  NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL
});

if (!parsedEnv.success) {
  throw new Error('Missing or invalid NEXT_PUBLIC_API_BASE_URL environment variable.');
}

export const env = parsedEnv.data;
