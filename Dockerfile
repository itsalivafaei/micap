# Multi-stage Dockerfile for MICAP
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    openjdk-11-jdk \
    && rm -rf /var/lib/apt/lists/*

# Set Java environment
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# =================================
# Development stage
# =================================
FROM base as development

WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-dev.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

EXPOSE 8000
CMD ["python", "-m", "src.api.main"]

# =================================
# Testing stage
# =================================
FROM development as testing

# Run tests
RUN python -m pytest tests/ --cov=src --cov-report=html

# =================================
# Production stage
# =================================
FROM base as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash micap

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY src/ ./src/
COPY setup.py ./

# Install package
RUN pip install --no-cache-dir .

# Change ownership to non-root user
RUN chown -R micap:micap /app
USER micap

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "src.api.main:app"] 