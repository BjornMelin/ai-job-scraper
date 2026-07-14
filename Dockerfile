# Multi-stage build for optimized layer caching
FROM python:3.14-slim AS base

# Install the health-check client
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install the same uv release used in CI
RUN pip install --no-cache-dir uv==0.11.28

# Stage 2: Dependencies
FROM base AS dependencies

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock README.md ./

# Install dependencies with UV
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --locked --no-dev --no-install-project

# Stage 3: Application
FROM dependencies AS app

# Copy application code
COPY . .

# Install the application into the locked environment
RUN --mount=type=cache,target=/root/.cache/uv \
    UV_LINK_MODE=copy uv sync --locked --no-dev

# Create the non-root runtime user and its only writable application directory
RUN useradd --create-home --uid 1000 appuser && \
    install -d --owner=appuser --group=appuser /app/db

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DB_URL=sqlite:////app/db/jobs.db \
    # Streamlit specific
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["/app/.venv/bin/ai-job-scraper", "--port=8501", "--address=0.0.0.0"]
