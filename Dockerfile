# Multi-stage build for optimized layer caching
FROM python:3.12-slim AS base

# Install system dependencies for headless browser operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Core browser dependencies
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdbus-1-3 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 fonts-liberation libappindicator3-1 xdg-utils \
    # Build dependencies for some Python packages
    gcc g++ \
    # Health check dependency
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install UV package manager
RUN pip install --no-cache-dir uv

# Stage 2: Dependencies
FROM base AS dependencies

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock* README.md ./

# Install dependencies with UV
RUN uv sync --frozen --no-dev

# Install Playwright browsers
RUN uv run python -m playwright install chromium

# Stage 3: Application
FROM dependencies AS app

# Copy application code
COPY . .

# Create directory for database persistence
RUN mkdir -p /app/db

# Set Python environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Streamlit specific
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]