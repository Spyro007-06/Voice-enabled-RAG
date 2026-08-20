# =============================================================================
# Phase 6.11 — Production Dockerfile for HH Goa 2026 Multilingual Voice RAG
# =============================================================================
# Multi-stage, security-hardened, non-root container image using Python 3.11-slim.

# Stage 1: Build & Dependency Resolution
FROM python:3.11-slim AS builder

WORKDIR /build

# Configure Python & Build Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for deterministic package installation
COPY --from=ghcr.io/astral-sh/uv:0.6.5 /uv /uvx /bin/

# Copy dependency specifications
COPY requirements.txt .

# Install dependencies into virtualenv
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache -r requirements.txt


# Stage 2: Production Runtime Image
FROM python:3.11-slim AS runtime

LABEL maintainer="Antigravity Voice RAG Team" \
      description="HH Goa 2026 Multilingual Voice RAG Backend Service" \
      version="1.0.0"

WORKDIR /app

# Configure Runtime Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME="/app/data/cache/huggingface" \
    TRANSFORMERS_CACHE="/app/data/cache/huggingface" \
    ENVIRONMENT="production" \
    APP_HOST="0.0.0.0" \
    APP_PORT=8000

# Install runtime dependencies for health checks & audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create dedicated non-root user and group (UID 10001)
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -d /app -s /sbin/nologin -c "Voice RAG App User" appuser

# Copy virtualenv from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source code and runtime assets
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser data/ ./data/

# Create persistent cache and runtime directories with appropriate permissions
RUN mkdir -p /app/data/cache/huggingface /app/data/qdrant && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser:appuser

# Expose API service port
EXPOSE 8000

# Container healthcheck targeting the production health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production ASGI Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--proxy-headers", "--forwarded-allow-ips=*"]
