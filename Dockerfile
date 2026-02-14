# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM python:3.13-slim AS builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy uv binary from official image (much faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment with uv
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install server dependencies (10-100x faster than pip)
COPY requirements-server.txt .
RUN uv pip install --python /opt/venv/bin/python -r requirements-server.txt

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM python:3.13-slim AS runtime

# Labels for container registry
LABEL org.opencontainers.image.source="https://github.com/OWNER/backtest-simulation"
LABEL org.opencontainers.image.description="AI-powered Natural Language Backtesting Service"
LABEL org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Application settings
    APP_DEBUG=false \
    PORT=8000

# Install runtime dependencies (if needed for specific packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser backend/ ./backend/
COPY --chown=appuser:appuser config.yaml ./

# Create data directories
RUN mkdir -p data/prices && chown -R appuser:appuser data/

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
