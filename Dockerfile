# Multi-stage Dockerfile for MICAP - Developer Sharing Optimized
FROM python:3.11-slim AS base

# Metadata
LABEL maintainer="Ali Vafaei <beady_stats.38@icloud.com>"
LABEL description="Market Intelligence & Competitor Analysis Platform"
LABEL version="0.1.0"

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies in a single layer
# Note: Java removed for lighter image - add back if Spark integration needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# =================================
# Development stage (recommended for developer sharing)
# =================================
FROM base AS development

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm

# Copy source code and configuration
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml ./

# Install package in development mode
ENV SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0-dev
RUN pip install -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/artifacts

EXPOSE 8000
CMD ["python", "-m", "src.api.main"]

# =================================
# Testing stage
# =================================
FROM development AS testing

# Copy test files
COPY tests/ ./tests/

# Install additional test dependencies if needed
RUN pip install --no-cache-dir pytest-cov pytest-mock

# Run tests (optional - can be disabled in CI)
RUN python -m pytest tests/ --cov=src --cov-report=html || true

# =================================
# Production stage (for future production use)
# =================================
FROM base AS production

# Create non-root user for security
RUN groupadd -r micap && useradd -r -g micap -d /app -s /bin/bash micap

WORKDIR /app

# Copy requirements and install production dependencies only
COPY requirements.txt ./

# Install Python dependencies in a single layer
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_sm && \
    pip cache purge

# Copy source code and essential configuration
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml ./

# Install package in production mode
RUN pip install --no-cache-dir . && \
    pip cache purge

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data /app/artifacts && \
    chown -R micap:micap /app

# Switch to non-root user
USER micap

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Use gunicorn for production with proper worker configuration
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "120", "--keepalive", "2", "--max-requests", "1000", "--max-requests-jitter", "50", "--access-logfile", "-", "--error-logfile", "-", "src.api.main:app"]

# =================================
# Minimal stage (smallest possible image)
# =================================
FROM python:3.11-slim AS minimal

# Metadata
LABEL maintainer="Ali Vafaei <beady_stats.38@icloud.com>"
LABEL description="MICAP API - Minimal Image"
LABEL version="0.1.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# Install minimal runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r micap && useradd -r -g micap -d /app -s /bin/bash micap

WORKDIR /app

# Copy only the installed packages from production stage
COPY --from=production /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=production /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=production --chown=micap:micap /app/src ./src
COPY --from=production --chown=micap:micap /app/config ./config

# Create directories
RUN mkdir -p /app/logs && chown micap:micap /app/logs

# Switch to non-root user
USER micap

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Simplified command for minimal image
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"] 