# ============================================================================
# ML Enhanced Router with CSP Network Integration - Docker Container
# ============================================================================

# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Build arguments
ARG PYTHON_VERSION=3.11
ARG BUILD_DATE
ARG VCS_REF

# Labels for metadata
LABEL maintainer="ML Router Team" \
      org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0" \
      org.label-schema.description="ML Enhanced Router with CSP Network Integration" \
      org.label-schema.name="ml-router-network"

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY requirements_integrated.txt /tmp/requirements_integrated.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements_integrated.txt

# ============================================================================
# Production Stage
# ============================================================================
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FLASK_APP=ml_router_network.py \
    FLASK_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application user for security
RUN groupadd -r mlrouter && useradd -r -g mlrouter mlrouter

# Create application directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/ml_router_network_data \
             /app/logs \
             /app/config \
             /app/backups \
             /app/models \
             /app/static \
             /app/templates

# Copy application files
COPY . /app/

# Remove development files and clean up
RUN rm -f /app/.env.example \
          /app/docker-compose.yml \
          /app/Dockerfile \
          /app/.gitignore \
          /app/README.md

# Set proper ownership
RUN chown -R mlrouter:mlrouter /app

# Switch to non-root user
USER mlrouter

# Create a minimal configuration if none exists
RUN if [ ! -f /app/.env ]; then \
    echo "DATABASE_URL=sqlite:///ml_router_network.db" > /app/.env && \
    echo "SESSION_SECRET=docker-default-secret-change-in-production" >> /app/.env && \
    echo "CSP_NETWORK_ENABLED=true" >> /app/.env && \
    echo "ML_ROUTER_ENABLED=true" >> /app/.env && \
    echo "LOG_LEVEL=INFO" >> /app/.env; \
    fi

# Expose ports
EXPOSE 5000 30405

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Entry point script
COPY --chown=mlrouter:mlrouter docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Default command
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "ml_router_network.py"]

# ============================================================================
# Development Stage (for development builds)
# ============================================================================
FROM production as development

# Switch back to root for development tools
USER root

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    strace \
    telnet \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    jupyter \
    ipython

# Set development environment
ENV FLASK_ENV=development \
    FLASK_DEBUG=1 \
    LOG_LEVEL=DEBUG

# Switch back to application user
USER mlrouter

# Override command for development
CMD ["python", "ml_router_network.py"]