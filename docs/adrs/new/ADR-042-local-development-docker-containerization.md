# ADR-042: Local Development Docker Containerization

## Status
Accepted

## Context

Need simple Docker containerization for local development workflow without production complexity. The application uses Reflex as the UI framework, SQLModel with SQLite for local development, and basic async patterns for background processing.

## Decision

Implement minimal Docker setup with:
- Single Dockerfile for Reflex application
- docker-compose.yml for multi-service development
- Basic environment configuration
- Simple health checks for development

### Architecture Components
- **UI Framework**: Reflex (local development server)
- **Database**: SQLite with SQLModel (local file-based)
- **AI Processing**: Simple local vLLM or API fallback
- **Background Tasks**: Basic async patterns with Reflex
- **Containerization**: Docker + docker-compose for development

## Implementation

### Dockerfile for Local Development

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install uv

# Copy dependency files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies
RUN uv sync

# Copy application code
COPY . .

# Expose Reflex default port
EXPOSE 3000
EXPOSE 8000

# Development command
CMD ["uv", "run", "reflex", "run", "--env", "dev", "--frontend-port", "3000", "--backend-port", "8000"]
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: '3.8'

services:
  ai-job-scraper:
    build: 
      context: .
      dockerfile: Dockerfile
    container_name: ai-job-scraper-dev
    ports:
      - "3000:3000"  # Reflex frontend
      - "8000:8000"  # Reflex backend
    volumes:
      - ./src:/app/src          # Source code hot reload
      - ./data:/app/data        # Local SQLite database
      - ./logs:/app/logs        # Application logs
      - ./.env:/app/.env        # Environment configuration
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=sqlite:///./data/jobs.db
      - LOG_LEVEL=DEBUG
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'

  # Optional: Local AI service (if using vLLM)
  local-ai:
    image: vllm/vllm-openai:latest
    container_name: local-ai-dev
    ports:
      - "8080:8000"
    environment:
      - MODEL=Qwen/Qwen2.5-1.5B-Instruct
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
        limits:
          memory: 8G
    profiles: ["ai"]  # Optional service
    
volumes:
  app_data:
```

### Environment Configuration

```bash
# .env.example
# Copy to .env and customize for your development environment

# Database Configuration
DATABASE_URL=sqlite:///./data/jobs.db

# AI Processing Configuration
AI_PROVIDER=local  # or 'openai' for API fallback
LOCAL_AI_BASE_URL=http://local-ai:8000/v1
OPENAI_API_KEY=your_api_key_here

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true

# Scraping Configuration
MAX_CONCURRENT_JOBS=5
REQUEST_DELAY=1.0
USER_AGENT=Mozilla/5.0 (compatible; AI-Job-Scraper/1.0)

# Reflex Configuration
REFLEX_FRONTEND_PORT=3000
REFLEX_BACKEND_PORT=8000
```

### Health Check Endpoint

```python
# src/health.py
from reflex import State
import asyncio
import sqlite3
from pathlib import Path

class HealthState(State):
    """Health check state for monitoring application status."""
    
    def check_database(self) -> bool:
        """Check database connectivity."""
        try:
            db_path = Path("./data/jobs.db")
            if not db_path.exists():
                return True  # New installation is OK
                
            with sqlite3.connect(db_path) as conn:
                conn.execute("SELECT 1")
                return True
        except Exception:
            return False
    
    def check_ai_service(self) -> bool:
        """Check AI service connectivity."""
        try:
            # Basic check - can be expanded based on AI provider
            return True
        except Exception:
            return False
    
    async def health_check(self) -> dict:
        """Comprehensive health check."""
        return {
            "status": "healthy",
            "database": self.check_database(),
            "ai_service": self.check_ai_service(),
            "environment": "development"
        }
```

### Development Scripts

```bash
#!/bin/bash
# scripts/dev-start.sh
# Start development environment

echo "Starting AI Job Scraper development environment..."

# Create necessary directories
mkdir -p data logs models

# Start services
docker-compose up --build -d

# Show logs
echo "Services starting... View logs with:"
echo "docker-compose logs -f"
```

```bash
#!/bin/bash
# scripts/dev-stop.sh
# Stop development environment

echo "Stopping AI Job Scraper development environment..."
docker-compose down

echo "Environment stopped."
```

```bash
#!/bin/bash
# scripts/dev-reset.sh
# Reset development environment (clean slate)

echo "Resetting development environment..."

# Stop services
docker-compose down -v

# Clean up data (optional)
read -p "Remove database and logs? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf data logs
    echo "Data cleaned."
fi

# Rebuild and start
docker-compose up --build -d

echo "Development environment reset complete."
```

## Consequences

### Positive
- Easy local development setup with `docker-compose up`
- Hot reload for source code changes
- Consistent development environment across team members
- Simple debugging and log access
- No production infrastructure complexity
- Clear separation between development and production concerns

### Negative
- Requires Docker and docker-compose installation
- Additional abstraction layer for development
- Container resource usage on development machines
- Need to manage environment variables and configuration

### Risk Mitigation
- Provide clear setup documentation
- Include example environment configuration
- Offer both containerized and direct development options
- Keep container resource limits reasonable for development

## Validation

### Success Criteria
- [ ] Developer can start environment with single command
- [ ] Reflex application accessible at http://localhost:3000
- [ ] SQLite database persists between container restarts
- [ ] Hot reload works for source code changes
- [ ] Health checks pass consistently
- [ ] Logs are accessible and useful for debugging

### Testing Approach
- Manual testing of development workflow
- Documentation validation with fresh setup
- Performance testing with reasonable development loads
- Cross-platform compatibility (Linux, macOS, Windows with WSL)

## Related ADRs
- **Updates ADR-035**: Final Production Architecture (local development focus)
- **Replaces Archived ADR-042**: vLLM Two-Tier Deployment Strategy (production-focused)
- **Supports ADR-040**: UI Component Architecture (Reflex containerization)

---

*This ADR provides simple, practical Docker containerization for local development without unnecessary production complexity. Developers should be able to get started with basic `docker-compose up` command.*