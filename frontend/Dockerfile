# ============================================
# Stage 1: Dependencies
# ============================================
FROM node:22-alpine AS deps

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

# ============================================
# Stage 2: Build
# ============================================
FROM node:22-alpine AS builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build-time environment variables
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL}
ARG NEXT_PUBLIC_CODE_GENERATION_TIMEOUT_MS=1800000
ENV NEXT_PUBLIC_CODE_GENERATION_TIMEOUT_MS=${NEXT_PUBLIC_CODE_GENERATION_TIMEOUT_MS}

RUN npm run build

# ============================================
# Stage 3: Runtime - Minimal production image
# ============================================
FROM node:22-alpine AS runtime

LABEL org.opencontainers.image.source="https://github.com/George-Polya/backtest-simulation"
LABEL org.opencontainers.image.description="AI-powered Natural Language Backtesting Service - Frontend"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app

ENV NODE_ENV=production \
    PORT=3000 \
    HOSTNAME=0.0.0.0

# Create non-root user
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy standalone output from builder
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

CMD ["node", "server.js"]