import { z } from 'zod';

const clientEnvSchema = z.object({
  NEXT_PUBLIC_API_BASE_URL: z.string().url()
});

function getEnv() {
  const parsedEnv = clientEnvSchema.safeParse({
    NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL
  });

  if (!parsedEnv.success) {
    throw new Error('Missing or invalid NEXT_PUBLIC_API_BASE_URL environment variable.');
  }

  return parsedEnv.data;
}

type EnvType = z.infer<typeof clientEnvSchema>;

export const env: EnvType = new Proxy({} as EnvType, {
  get(_, prop: string) {
    return getEnv()[prop as keyof EnvType];
  }
});
