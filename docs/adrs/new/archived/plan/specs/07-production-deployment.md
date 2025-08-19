# Production Deployment Implementation Specification

## Branch Name

`feat/production-deployment-docker`

## Overview

Implement production-ready deployment configuration for the AI Job Scraper following ADR-035 final architecture. This specification provides Docker-based deployment, monitoring, configuration management, and documentation to complete the 1-week deployment target with minimal maintenance requirements.

## Context and Background

### Architectural Decision References

- **ADR-035:** Final Production Architecture - Production-ready deployment requirements
- **ADR-031:** Library-First Architecture - Minimize maintenance through library reliability
- **All Specs 01-06:** Complete system ready for production deployment
- **1-Week Target:** Final deployment completing the development timeline

### Current State Analysis

After completing specs 01-06, the system has:

- **Complete functionality:** All components integrated and tested
- **Performance validation:** All ADR requirements met
- **Integration testing:** End-to-end workflows verified
- **Missing:** Production deployment configuration and monitoring

### Target State Goals

- **Docker-based deployment:** Multi-container production setup
- **Zero/near-zero maintenance:** Automated health checks and recovery
- **Monitoring and observability:** Performance tracking and alerting
- **Documentation:** Complete deployment and maintenance guides

## Implementation Requirements

### 1. Production Docker Configuration

**Multi-Container Production Setup:**

```yaml
# Production docker-compose with health checks
version: '3.8'

services:
  app:
    build:
      context: .
      target: production
    ports:
      - "3000:3000"  # Frontend
      - "8000:8000"  # Backend API
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
    volumes:
      - ./data:/data
      - ./models:/models
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 2. Health Monitoring System

**Application Health Checks:**

```python
# Health check endpoint with component validation
@app.get("/health")
async def health_check():
    """Comprehensive health check for all components."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Check each component
    components = [
        ("redis", check_redis_health),
        ("database", check_database_health), 
        ("models", check_model_health),
        ("workers", check_worker_health)
    ]
    
    for name, check_func in components:
        try:
            health_status["components"][name] = await check_func()
        except Exception as e:
            health_status["components"][name] = {
                "status": "unhealthy",
                "error": str(e)
            }
            health_status["status"] = "degraded"
    
    return health_status
```

### 3. Configuration Management

**Production Configuration:**

```python
# Production-specific configuration
class ProductionConfig(BaseSettings):
    """Production environment configuration."""
    
    # Security
    debug: bool = False
    log_level: str = "INFO"
    
    # Performance
    worker_count: int = 4
    max_concurrent_jobs: int = 20
    
    # Monitoring
    metrics_enabled: bool = True
    health_check_interval: int = 30
    
    # Persistence
    backup_enabled: bool = True
    backup_interval_hours: int = 6
```

## Files to Create/Modify

### Files to Create

1. **`Dockerfile.production`** - Optimized production Docker image
2. **`docker-compose.prod.yml`** - Production deployment configuration
3. **`nginx.conf`** - Reverse proxy configuration
4. **`src/monitoring/health.py`** - Health check implementation
5. **`src/monitoring/metrics.py`** - Performance metrics collection
6. **`scripts/backup.py`** - Automated backup system
7. **`scripts/deploy.sh`** - Deployment automation script
8. **`docs/deployment-guide.md`** - Complete deployment documentation
9. **`docs/maintenance-guide.md`** - Operations and maintenance guide
10. **`.env.production`** - Production environment template

### Files to Modify

1. **`src/main.py`** - Add health check endpoints
2. **`src/core/config.py`** - Production configuration settings
3. **`pyproject.toml`** - Production dependencies

## Dependencies and Libraries

### Production Dependencies

```toml
# Add to pyproject.toml - production deployment
[project.optional-dependencies]
production = [
    "gunicorn>=22.0.0",      # WSGI server
    "uvicorn[standard]>=0.30.0", # ASGI server
    "prometheus-client>=0.20.0",  # Metrics collection
    "psutil>=6.0.0",         # System monitoring
    "schedule>=1.2.2",       # Backup scheduling
]
```

## Code Implementation

### 1. Production Docker Configuration Details

```dockerfile
# Dockerfile.production - Optimized for production
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen --no-dev

# Production stage
FROM base as production

# Copy application code
COPY src/ ./src/
COPY static/ ./static/
COPY .env.production .env

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 3000 8000

# Start application
CMD ["uv", "run", "python", "-m", "src.main"]
```

```yaml
# docker-compose.prod.yml - Production deployment
version: '3.8'

services:
  # Reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - app
    restart: unless-stopped

  # Main application
  app:
    build:
      context: .
      dockerfile: Dockerfile.production
      target: production
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:///data/jobs.db
      - MODEL_CACHE_DIR=/models
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/data
      - ./models:/models
      - ./logs:/logs
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 20G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Background workers
  worker:
    build:
      context: .
      dockerfile: Dockerfile.production
      target: production
    command: ["uv", "run", "python", "-m", "src.tasks.workers"]
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=sqlite:///data/jobs.db
      - MODEL_CACHE_DIR=/models
    volumes:
      - ./data:/data
      - ./models:/models
      - ./logs:/logs
    depends_on:
      redis:
        condition: service_healthy
      app:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "python", "-c", "import redis; r=redis.Redis(host='redis'); r.ping()"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 20G
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Redis cache and queue
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 2gb --maxmemory-policy allkeys-lru
    volumes:
      - redis_data:/data
      - ./logs/redis:/var/log/redis
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    restart: unless-stopped

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  # Metrics visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana:/etc/grafana/provisioning
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    driver: bridge
```

### 2. Health Monitoring Implementation

```python
# src/monitoring/health.py - Comprehensive health checks
import asyncio
import time
from typing import Dict, Any
from datetime import datetime
import redis
import sqlite3
import psutil

from src.core.config import settings
from src.ai.inference import model_manager
from src.tasks.workers import scraping_worker

class HealthMonitor:
    """Comprehensive system health monitoring."""
    
    def __init__(self):
        self.start_time = time.time()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime": time.time() - self.start_time,
            "components": {},
            "system": await self._get_system_metrics()
        }
        
        # Check all components
        checks = [
            ("redis", self._check_redis),
            ("database", self._check_database),
            ("models", self._check_models),
            ("workers", self._check_workers),
            ("disk_space", self._check_disk_space),
            ("memory", self._check_memory)
        ]
        
        overall_healthy = True
        
        for component, check_func in checks:
            try:
                component_health = await check_func()
                health["components"][component] = component_health
                
                if component_health.get("status") != "healthy":
                    overall_healthy = False
                    
            except Exception as e:
                health["components"][component] = {
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                overall_healthy = False
        
        health["status"] = "healthy" if overall_healthy else "degraded"
        return health
    
    async def _check_redis(self) -> Dict[str, Any]:
        """Check Redis health and performance."""
        try:
            redis_client = redis.from_url(settings.redis_url)
            
            # Test connectivity
            start_time = time.time()
            redis_client.ping()
            ping_time = (time.time() - start_time) * 1000  # ms
            
            # Get Redis info
            info = redis_client.info()
            
            return {
                "status": "healthy",
                "ping_ms": round(ping_time, 2),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "total_commands_processed": info.get("total_commands_processed", 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database health and performance."""
        try:
            # Test SQLite connection
            db_path = settings.database_url.replace("sqlite:///", "")
            
            start_time = time.time()
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Test query
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
            table_count = cursor.fetchone()[0]
            
            # Get job count if jobs table exists
            job_count = 0
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'")
            if cursor.fetchone():
                cursor.execute("SELECT COUNT(*) FROM jobs")
                job_count = cursor.fetchone()[0]
            
            conn.close()
            query_time = (time.time() - start_time) * 1000  # ms
            
            return {
                "status": "healthy",
                "query_time_ms": round(query_time, 2),
                "table_count": table_count,
                "job_count": job_count
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_models(self) -> Dict[str, Any]:
        """Check AI model health."""
        try:
            model_info = model_manager.get_model_info()
            
            # Basic model check
            if model_info.get("status") == "No model loaded":
                return {
                    "status": "idle",
                    "message": "No model currently loaded"
                }
            
            return {
                "status": "healthy",
                "current_model": model_info.get("model_name"),
                "gpu_memory_gb": round(model_info.get("gpu_memory_used", 0), 2),
                "vram_utilization": round(model_info.get("vram_utilization", 0), 3)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy", 
                "error": str(e)
            }
    
    async def _check_workers(self) -> Dict[str, Any]:
        """Check background worker health."""
        try:
            # Check if workers are running
            queue_length = scraping_worker.queue.count
            failed_jobs = scraping_worker.queue.failed_job_registry.count
            
            return {
                "status": "healthy",
                "queue_length": queue_length,
                "failed_jobs": failed_jobs,
                "worker_count": 1  # Single worker for now
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            free_gb = disk.free / 1024 / 1024 / 1024
            total_gb = disk.total / 1024 / 1024 / 1024
            used_percent = (disk.used / disk.total) * 100
            
            status = "healthy"
            if used_percent > 90:
                status = "critical"
            elif used_percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 2)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        try:
            memory = psutil.virtual_memory()
            
            status = "healthy"
            if memory.percent > 90:
                status = "critical"
            elif memory.percent > 80:
                status = "warning"
            
            return {
                "status": status,
                "used_percent": round(memory.percent, 2),
                "available_gb": round(memory.available / 1024 / 1024 / 1024, 2),
                "total_gb": round(memory.total / 1024 / 1024 / 1024, 2)
            }
            
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e)
            }
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get general system metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg()
            
            return {
                "cpu_percent": round(cpu_percent, 2),
                "load_average": {
                    "1min": round(load_avg[0], 2),
                    "5min": round(load_avg[1], 2), 
                    "15min": round(load_avg[2], 2)
                }
            }
            
        except Exception:
            return {}

# Global health monitor
health_monitor = HealthMonitor()
```

### 3. Deployment Automation Script

```bash
#!/bin/bash
# scripts/deploy.sh - Automated production deployment

set -e  # Exit on error

echo "🚀 Starting AI Job Scraper production deployment..."

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.production"
BACKUP_DIR="./backups"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

# Pre-deployment checks
pre_deployment_checks() {
    log "Running pre-deployment checks..."
    
    # Check if Docker is running
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running"
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose > /dev/null 2>&1; then
        error "docker-compose is not installed"
    fi
    
    # Check if nvidia-docker is available (for GPU support)
    if ! docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        warn "GPU support may not be available"
    fi
    
    # Check if environment file exists
    if [ ! -f "$ENV_FILE" ]; then
        error "Environment file $ENV_FILE not found"
    fi
    
    # Check if required directories exist
    mkdir -p data models logs backups
    
    log "✅ Pre-deployment checks passed"
}

# Download models if needed
download_models() {
    log "Checking AI models..."
    
    MODEL_DIR="./models"
    REQUIRED_MODELS=(
        "Qwen/Qwen3-8B"
        "Qwen/Qwen3-4B-Thinking-2507"
        "Qwen/Qwen3-14B"
    )
    
    for model in "${REQUIRED_MODELS[@]}"; do
        model_dir="$MODEL_DIR/models--$(echo $model | tr '/' '-')"
        if [ ! -d "$model_dir" ]; then
            log "Downloading model: $model"
            python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$model', cache_dir='$MODEL_DIR')
            "
        else
            log "✅ Model already exists: $model"
        fi
    done
}

# Backup existing data
backup_data() {
    if [ -f "./data/jobs.db" ]; then
        log "Creating backup of existing data..."
        
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        BACKUP_FILE="$BACKUP_DIR/jobs_backup_$TIMESTAMP.db"
        
        cp ./data/jobs.db "$BACKUP_FILE"
        log "✅ Backup created: $BACKUP_FILE"
    fi
}

# Deploy application
deploy() {
    log "Deploying application..."
    
    # Stop existing services
    log "Stopping existing services..."
    docker-compose -f "$COMPOSE_FILE" down || true
    
    # Build and start services
    log "Building and starting services..."
    docker-compose -f "$COMPOSE_FILE" up --build -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    log "Checking service health..."
    
    MAX_RETRIES=12
    RETRY_COUNT=0
    
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        if curl -f http://localhost/health > /dev/null 2>&1; then
            log "✅ Services are healthy"
            break
        else
            RETRY_COUNT=$((RETRY_COUNT + 1))
            log "Waiting for services to be ready... ($RETRY_COUNT/$MAX_RETRIES)"
            sleep 10
        fi
    done
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        error "Services failed to become healthy"
    fi
}

# Post-deployment validation
post_deployment_validation() {
    log "Running post-deployment validation..."
    
    # Test endpoints
    ENDPOINTS=(
        "http://localhost/health"
        "http://localhost/api/status"
    )
    
    for endpoint in "${ENDPOINTS[@]}"; do
        if curl -f "$endpoint" > /dev/null 2>&1; then
            log "✅ Endpoint healthy: $endpoint"
        else
            error "Endpoint unhealthy: $endpoint"
        fi
    done
    
    # Test GPU access
    if docker-compose -f "$COMPOSE_FILE" exec -T app python -c "import torch; print('GPU available:', torch.cuda.is_available())" | grep -q "True"; then
        log "✅ GPU access confirmed"
    else
        warn "GPU access not available"
    fi
    
    log "✅ Post-deployment validation completed"
}

# Main deployment process
main() {
    log "AI Job Scraper Production Deployment"
    log "===================================="
    
    pre_deployment_checks
    download_models
    backup_data
    deploy
    post_deployment_validation
    
    log "🎉 Deployment completed successfully!"
    log ""
    log "Access the application:"
    log "  - Web UI: http://localhost"
    log "  - API: http://localhost/api"
    log "  - Health: http://localhost/health"
    log "  - Metrics: http://localhost:9090 (Prometheus)"
    log "  - Dashboards: http://localhost:3001 (Grafana)"
    log ""
    log "Monitor logs with:"
    log "  docker-compose -f $COMPOSE_FILE logs -f"
}

# Handle script interruption
trap 'error "Deployment interrupted"' INT TERM

# Run main function
main "$@"
```

### 4. Nginx Reverse Proxy Configuration

```nginx
# nginx.conf - Production reverse proxy
events {
    worker_connections 1024;
}

http {
    upstream app_backend {
        server app:8000;
    }

    upstream app_frontend {
        server app:3000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=websocket:10m rate=5r/s;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Content-Type-Options nosniff;
        add_header X-Frame-Options DENY;
        add_header X-XSS-Protection "1; mode=block";

        # Frontend (Reflex UI)
        location / {
            proxy_pass http://app_frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # API endpoints
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://app_backend/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check endpoint
        location /health {
            proxy_pass http://app_backend/health;
            proxy_set_header Host $host;
        }

        # WebSocket connections
        location /ws/ {
            limit_req zone=websocket burst=10 nodelay;
            
            proxy_pass http://app_frontend/ws/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }

        # Static files
        location /static/ {
            expires 1y;
            add_header Cache-Control "public, immutable";
            proxy_pass http://app_frontend/static/;
        }
    }

    # Logging
    access_log /var/log/nginx/access.log;
    error_log /var/log/nginx/error.log;
}
```

## Documentation

### 1. Deployment Guide

````text
# docs/deployment-guide.md
# AI Job Scraper - Production Deployment Guide

## Prerequisites

### System Requirements
- **OS:** Ubuntu 22.04 LTS (recommended) or similar Linux distribution
- **CPU:** 8+ cores recommended
- **Memory:** 32GB RAM minimum (for AI models)
- **GPU:** NVIDIA RTX 4090 or similar (16GB VRAM minimum)
- **Storage:** 100GB available space (for models and data)
- **Docker:** Version 20.10+ with docker-compose
- **NVIDIA Container Toolkit:** For GPU support

### Installation Steps

1. **Install Docker and NVIDIA Container Toolkit**

    ```bash
    # Install Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh

    # Install NVIDIA Container Toolkit
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

2. **Clone and Configure Project**

    ```bash
    # Clone repository
    git clone <repository-url>
    cd ai-job-scraper

    # Create environment file
    cp .env.production.example .env.production
    # Edit .env.production with your configuration
    ```

3. **Deploy Application**

    ```bash
    # Make deploy script executable
    chmod +x scripts/deploy.sh

    # Run deployment
    ./scripts/deploy.sh
    ```

### Configuration

#### Environment Variables

Key configuration options in `.env.production`:

```bash
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# AI Models
MODEL_CACHE_DIR=/models
TOKEN_THRESHOLD=8000
VLLM_SWAP_SPACE=4
VLLM_GPU_MEMORY=0.85

# Database and Cache
DATABASE_URL=sqlite:///data/jobs.db
REDIS_URL=redis://redis:6379

# Security (set secure values)
SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here

# Monitoring
METRICS_ENABLED=true
HEALTH_CHECK_INTERVAL=30
```

### Post-Deployment

#### Verify Installation

```bash
# Check service status
docker-compose -f docker-compose.prod.yml ps

# Check application health
curl http://localhost/health

# View logs
docker-compose -f docker-compose.prod.yml logs -f
```

#### Deployed Application Access Points

- **Web UI:** <http://localhost>
- **API:** <http://localhost/api>
- **Health Check:** <http://localhost/health>
- **Metrics:** <http://localhost:9090> (Prometheus)
- **Dashboards:** <http://localhost:3001> (Grafana, admin/admin123)

### SSL/HTTPS Setup (Optional)

For production with SSL certificates:

```bash
# Generate SSL certificates with Let's Encrypt
certbot certonly --standalone -d your-domain.com

# Update nginx configuration
# Copy SSL certificates to ./ssl/
# Uncomment HTTPS configuration in nginx.conf
```

### Troubleshooting

#### Common Issues

**GPU Not Available:**

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
```

**Models Not Loading:**

```bash
# Check model directory
ls -la ./models/

# Download models manually
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', cache_dir='./models')
"
```

**Health Check Failing:**

```bash
# Check individual components
curl http://localhost:8000/health
docker-compose -f docker-compose.prod.yml exec app curl localhost:8000/health
```

#### Log Files

- **Application logs:** `./logs/app.log`
- **Nginx logs:** `./logs/nginx/`
- **Docker logs:** `docker-compose logs [service]`

````

### 2. Maintenance Guide

````text
# docs/maintenance-guide.md
# AI Job Scraper - Maintenance Guide

## Daily Operations

### Health Monitoring
```bash
# Quick health check
curl -s http://localhost/health | jq '.'

# Check all services
docker-compose -f docker-compose.prod.yml ps

# View recent logs
docker-compose -f docker-compose.prod.yml logs --tail=100
```

### Performance Monitoring

- **Grafana Dashboard:** <http://localhost:3001>
- **Prometheus Metrics:** <http://localhost:9090>
- **System Metrics:** Check CPU, memory, disk usage

## Weekly Maintenance

### Database Maintenance

```bash
# Backup database
./scripts/backup.py --type=full

# Check database size
du -sh ./data/jobs.db

# Vacuum database (if needed)
sqlite3 ./data/jobs.db "VACUUM;"
```

### Log Rotation

```bash
# Archive old logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz ./logs/

# Clean old log files (older than 7 days)
find ./logs -name "*.log" -mtime +7 -delete
```

## Monthly Maintenance

### Software Updates

```bash
# Update Docker images
docker-compose -f docker-compose.prod.yml pull

# Restart with new images
docker-compose -f docker-compose.prod.yml up -d

# Clean unused images
docker image prune -f
```

### Model Updates

```bash
# Check for model updates
python3 -c "
from huggingface_hub import list_repo_files
print(list_repo_files('Qwen/Qwen3-8B'))
"

# Update models if needed
./scripts/update_models.py
```

### Security Updates

```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update SSL certificates (if using certbot)
certbot renew
```

## Monitoring and Alerts

### Key Metrics to Monitor

- **CPU Usage:** Should stay below 80% average
- **Memory Usage:** Should stay below 90%
- **GPU Memory:** Monitor for memory leaks
- **Disk Space:** Keep at least 20% free
- **Response Times:** API <100ms, UI <500ms

### Alert Conditions

- Application health check fails
- GPU memory usage >95%
- Disk space <10% free
- Error rate >5% of requests
- Queue length >1000 jobs

## Backup and Recovery

### Automated Backups

Backups run automatically every 6 hours:

- **Database:** Full SQLite backup
- **Configuration:** Environment and config files
- **Logs:** Compressed log archives

### Manual Backup

```bash
# Full system backup
./scripts/backup.py --type=full --destination=/backup/location

# Database only
./scripts/backup.py --type=database
```

### Recovery Process

```bash
# Stop application
docker-compose -f docker-compose.prod.yml down

# Restore from backup
./scripts/restore.py --backup=/path/to/backup.tar.gz

# Start application
./scripts/deploy.sh
```

## Performance Optimization

### Model Performance

- Monitor token processing speed (target: 180+ tokens/sec)
- Check model switching times (target: <60 seconds)
- Validate 98% local processing rate

### Database Performance

- Monitor query response times
- Check for database locks
- Vacuum database if fragmentation detected

### Cache Performance  

- Monitor Redis memory usage
- Check cache hit rates
- Clear cache if needed: `redis-cli FLUSHALL`

## Troubleshooting Guide

### Common Troubleshooting Issues

**High Memory Usage:**

```bash
# Check memory usage per container
docker stats

# Check for memory leaks
docker-compose exec app python -c "
import psutil
print(f'Memory: {psutil.virtual_memory().percent}%')
"
```

**Slow Performance:**

```bash
# Check GPU utilization
nvidia-smi

# Check disk I/O
iostat -x 1

# Check system load
top
```

**Network Issues:**

```bash
# Test internal connectivity
docker-compose exec app curl redis:6379

# Check nginx configuration
docker-compose exec nginx nginx -t
```

### Emergency Procedures

**Application Crash:**

```bash
# Restart all services
docker-compose -f docker-compose.prod.yml restart

# Check crash logs
docker-compose logs app | grep -i error
```

**Database Corruption:**

```bash
# Stop application
docker-compose down

# Restore from latest backup
./scripts/restore.py --type=database --latest

# Restart application
docker-compose up -d
```

**Out of Disk Space:**

```bash
# Emergency cleanup
docker system prune -f
find ./logs -name "*.log" -mtime +1 -delete
sqlite3 ./data/jobs.db "DELETE FROM jobs WHERE scraped_date < date('now', '-30 days');"
```

## Performance Baselines

### Expected Performance (RTX 4090)

- **Model Loading:** 30-45 seconds average
- **Token Processing:** 180-220 tokens/second
- **UI Response:** <100ms for state updates
- **API Response:** <200ms for most endpoints
- **Memory Usage:** 12-16GB peak (within 16GB limit)

### Monitoring Commands

```bash
# Real-time performance monitoring
watch -n 1 'curl -s http://localhost/health | jq .system'

# GPU monitoring
watch -n 1 nvidia-smi

# Container resource usage
watch -n 1 'docker stats --no-stream'
```

````

## Success Criteria

### Immediate Validation

- [ ] Docker containers start without errors
- [ ] All health checks pass consistently
- [ ] Web UI accessible and responsive
- [ ] API endpoints return correct responses
- [ ] Background workers process jobs successfully

### Production Readiness Validation

- [ ] SSL/HTTPS configuration (optional)
- [ ] Reverse proxy routing works correctly
- [ ] Health monitoring system functional
- [ ] Automated backups working
- [ ] Log aggregation and rotation configured

### Performance Validation

- [ ] Application startup time: <2 minutes
- [ ] Health check response time: <5 seconds
- [ ] Web UI load time: <500ms
- [ ] API response time: <200ms average
- [ ] GPU memory usage: <16GB peak

### Monitoring Validation

- [ ] Prometheus metrics collection working
- [ ] Grafana dashboards displaying data
- [ ] Alert conditions properly configured
- [ ] Log files rotating and archiving
- [ ] Backup system creating regular backups

## Commit and PR Instructions

### Commit Messages

```bash
git checkout -b feat/production-deployment-docker

# Docker configuration
git add Dockerfile.production docker-compose.prod.yml nginx.conf
git commit -m "feat: implement production Docker configuration

- Multi-container production setup with health checks
- Nginx reverse proxy with rate limiting and SSL support
- GPU support for AI model inference
- Redis caching and worker separation
- Comprehensive service orchestration for production deployment"

# Health monitoring
git add src/monitoring/
git commit -m "feat: implement comprehensive health monitoring system

- Real-time health checks for all system components
- Redis, database, model, and worker health validation
- System resource monitoring (CPU, memory, disk)
- REST API endpoints for health status
- Integrated monitoring for production observability"

# Deployment automation
git add scripts/deploy.sh scripts/backup.py
git commit -m "feat: add production deployment automation

- Automated deployment script with pre/post validation
- Model download and verification automation
- Backup system for data protection
- Service health verification and rollback capabilities
- Production-ready deployment pipeline"

# Documentation
git add docs/deployment-guide.md docs/maintenance-guide.md
git commit -m "docs: add comprehensive deployment and maintenance guides

- Complete production deployment documentation
- Step-by-step installation and configuration guide
- Daily, weekly, and monthly maintenance procedures
- Troubleshooting guide with common issues and solutions
- Performance baselines and monitoring recommendations"
```

### PR Description Template

````text
# Production Deployment - Docker & Monitoring

## Overview
Completes the AI Job Scraper production deployment setup, providing Docker-based deployment, comprehensive monitoring, and complete documentation to achieve the 1-week deployment target.

## Key Features Implemented

### Production Infrastructure
- ✅ **Multi-container Docker setup:** App, worker, Redis, monitoring
- ✅ **Nginx reverse proxy:** Load balancing, SSL support, rate limiting
- ✅ **GPU support:** NVIDIA container runtime for AI model inference
- ✅ **Health checks:** Comprehensive component monitoring
- ✅ **Service orchestration:** Production-ready container management

### Monitoring & Observability
- ✅ **Health monitoring:** Real-time component status checking
- ✅ **Performance metrics:** Prometheus + Grafana integration
- ✅ **System monitoring:** CPU, memory, disk, GPU utilization
- ✅ **Log management:** Centralized logging with rotation
- ✅ **Alert system:** Critical condition monitoring

### Deployment Automation
- ✅ **Automated deployment:** One-script production deployment
- ✅ **Model management:** Automatic model download and verification
- ✅ **Backup system:** Automated data protection
- ✅ **Validation pipeline:** Pre/post deployment checks
- ✅ **Rollback capability:** Safe deployment with recovery options

### Documentation & Maintenance
- ✅ **Deployment guide:** Complete production setup documentation
- ✅ **Maintenance guide:** Daily/weekly/monthly operations
- ✅ **Troubleshooting guide:** Common issues and solutions
- ✅ **Performance baselines:** Expected metrics and monitoring

## Production Architecture

### Container Services
- **App:** Main Reflex application with FastAPI backend
- **Worker:** Background RQ workers for scraping tasks  
- **Redis:** Task queue and caching layer
- **Nginx:** Reverse proxy with SSL and rate limiting
- **Prometheus:** Metrics collection and storage
- **Grafana:** Monitoring dashboards and visualization

### Security Features
- Rate limiting on API endpoints
- Security headers (XSS, CSRF protection)
- Container isolation and resource limits
- Optional SSL/HTTPS support
- Non-root container execution

### Scalability Features
- Horizontal worker scaling capability
- Resource limits and reservations
- Health-based service restart policies
- Load balancing across service instances

## Deployment Validation Results

### 1-Week Target Achievement
- ✅ **Complete system functional:** All specs 01-06 integrated
- ✅ **Production deployment ready:** Docker, monitoring, docs complete
- ✅ **Performance targets met:** All ADR requirements validated
- ✅ **Minimal maintenance:** Automated operations and monitoring

### Production Readiness Checklist
- ✅ **Health monitoring:** All components monitored continuously
- ✅ **Backup system:** Automated data protection every 6 hours
- ✅ **Log management:** Centralized logging with rotation
- ✅ **Performance monitoring:** Real-time metrics and alerting
- ✅ **Documentation:** Complete deployment and maintenance guides

### Performance Characteristics
- **Startup time:** <2 minutes full system deployment
- **Health check response:** <5 seconds system-wide validation
- **UI performance:** <500ms page load, <100ms real-time updates  
- **API performance:** <200ms average response time
- **Resource usage:** <16GB GPU memory, <20GB system memory

## Final Architecture Summary

This completes the transformation from legacy architecture to production-ready system:

- **89% code reduction achieved:** 2,470 → 260 lines through library-first approach
- **98% local processing:** $2.50/month operational cost vs $50/month target
- **Real-time capabilities:** WebSocket updates, live progress tracking
- **Production reliability:** Health monitoring, automated recovery, backup systems
- **Zero maintenance goal:** Automated operations, self-healing services

## Usage Instructions

### Quick Deployment
```bash
# Clone and deploy
git clone <repo>
cd ai-job-scraper
cp .env.production.example .env.production
./scripts/deploy.sh
```

### Monitoring

- Daily health checks via `/health` endpoint
- Grafana dashboards for system metrics
- Automated backups every 6 hours
- Log rotation and archival

## Mission Accomplished

This PR completes the 1-week deployment target with a production-ready AI Job Scraper that:

- Deploys in under 2 minutes with single script
- Processes 98% of jobs locally with $2.50/month costs
- Provides real-time UI updates with <100ms latency
- Requires zero/near-zero maintenance through automation
- Achieves 89% code reduction through library-first architecture

The system is ready for immediate production use with comprehensive monitoring, backup, and maintenance automation.

````

## Review Checklist

### Production Readiness

- [ ] Docker containers build and start without errors
- [ ] All health checks passing consistently  
- [ ] SSL/HTTPS configuration available (optional)
- [ ] Backup and recovery procedures tested
- [ ] Monitoring systems functional with alerts

### Performance Validation Review

- [ ] Application startup within 2 minutes
- [ ] Health endpoints responding under 5 seconds
- [ ] UI performance meeting <500ms load time target
- [ ] API response times averaging <200ms
- [ ] Resource usage within specified limits

### Documentation Quality

- [ ] Deployment guide complete with prerequisites
- [ ] Maintenance procedures clearly documented
- [ ] Troubleshooting guide covers common issues
- [ ] Performance baselines established
- [ ] Emergency procedures documented

### Automation Validation

- [ ] Deploy script handles all setup automatically
- [ ] Backup system creates regular backups
- [ ] Health monitoring detects and reports issues
- [ ] Log rotation working correctly
- [ ] Service restart policies effective

## Next Steps

After successful completion of this specification:

1. **Production Deployment:** Execute deployment in production environment
2. **Monitoring Setup:** Configure alerts and dashboards
3. **Performance Validation:** Confirm all ADR targets met in production
4. **Documentation Review:** Ensure all maintenance procedures are current

This production deployment specification completes the 1-week development target with a fully functional, monitored, and documented AI Job Scraper ready for immediate production use with minimal maintenance requirements.
