# Backtest execution environment with pre-installed dependencies
FROM python:3.13-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy uv binary from official image (much faster than pip)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies for numpy/pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages required for backtesting (10-100x faster than pip)
# backtesting>=0.6.0 adds 'spread' parameter for slippage simulation
RUN uv pip install --system \
    pandas==2.2.3 \
    numpy==2.1.3 \
    backtesting==0.6.5 \
    ta==0.11.0 \
    scipy==1.14.1 \
    matplotlib==3.9.2

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash backtest
USER backtest

WORKDIR /workspace
