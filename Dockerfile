# ──────────────────────────────────────────────────────────────
# NexaTrade — Production Dockerfile
# Multi-stage build: builder → runtime
# Final image: python:3.11-slim (~180 MB)
# ──────────────────────────────────────────────────────────────

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies into /build/wheels
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /build/wheels \
       -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for security
RUN groupadd --gid 1001 nexatrade \
    && useradd  --uid 1001 --gid nexatrade \
                --shell /bin/bash \
                --create-home nexatrade

# Runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq5 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install wheels from builder
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir --no-index \
        --find-links /wheels /wheels/*.whl \
    && rm -rf /wheels

# Copy application source
COPY --chown=nexatrade:nexatrade . .

# Create required directories
RUN mkdir -p logs plugins \
    && chown -R nexatrade:nexatrade logs plugins

# Switch to non-root user
USER nexatrade

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Entrypoint
CMD ["python", "main.py"]