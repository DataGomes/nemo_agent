# ============================================================================
# Meta-Agent Container
#
# Multi-stage build:
#   Stage 1 (builder): install Python deps
#   Stage 2 (runtime): slim image with just what's needed
#
# Usage:
#   Development:  docker compose up          (hot reload via volume mount)
#   Production:   docker compose -f docker-compose.yml -f docker-compose.prod.yml up
# ============================================================================

# -- Stage 1: Builder --------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first for layer caching
COPY requirements.txt requirements-server.txt ./
RUN pip install --no-cache-dir --prefix=/install \
    -r requirements.txt \
    -r requirements-server.txt

# -- Stage 2: Runtime --------------------------------------------------------
FROM python:3.12-slim AS runtime

# Non-root user for security
RUN groupadd -r agent && useradd -r -g agent -m agent

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY examples/ /app/
COPY requirements.txt requirements-server.txt /app/

# Create workdir for optimization artifacts
RUN mkdir -p /app/workdir && chown -R agent:agent /app

USER agent

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

# Default: run with uvicorn + hot reload (overridden in prod)
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
