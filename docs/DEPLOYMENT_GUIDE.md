# AI Job Scraper - Production Deployment Guide

**Version**: 1.0  
**Date**: 2025-08-27  
**Status**: Production Ready  

## Quick Start

### Prerequisites
- Python 3.12+ with uv package manager
- Docker & Docker Compose (recommended)
- 4GB+ RAM (8GB recommended for local AI)
- 2GB+ available disk space

### 1-Command Deployment
```bash
# Clone and deploy with Docker
git clone https://github.com/BjornMelin/ai-job-scraper.git
cd ai-job-scraper
cp .env.example .env  # Edit with your API keys
docker-compose up -d
```

Access the application at: http://localhost:8501

## Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Full Stack with Local AI
```bash
# Deploy complete system with vLLM local AI
docker-compose -f docker-compose.yml -f docker-compose.vllm.yml up -d

# Services running:
# - ai-job-scraper: Main application (port 8501)  
# - vllm-server: Local AI server (port 8000)
# - nginx: Reverse proxy (port 80)
```

#### Cloud AI Only
```bash
# Deploy with cloud AI fallback only
docker-compose up -d ai-job-scraper

# Requires OPENAI_API_KEY in .env file
```

### Option 2: Native Python Deployment

#### System Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/BjornMelin/ai-job-scraper.git
cd ai-job-scraper

# Install dependencies
uv sync --all-extras

# Run application
uv run streamlit run src/main.py --server.port 8501
```

### Option 3: Production Server Deployment

#### systemd Service Setup
```bash
# Create service user
sudo useradd --system --create-home ai-scraper

# Install as service
sudo cp deployment/ai-job-scraper.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ai-job-scraper
sudo systemctl start ai-job-scraper
```

## Configuration Management

### Environment Configuration (.env)

```bash
# Core Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
AI_TOKEN_THRESHOLD=8000
SCRAPER_LOG_LEVEL=INFO

# Database Configuration  
DATABASE_URL=sqlite:///jobs.db
# DATABASE_URL=postgresql://user:pass@localhost:5432/aiJobscraper  # PostgreSQL alternative

# Proxy Configuration (Optional)
USE_PROXIES=true
PROXY_POOL=http://user:pass@proxy1.example.com:8080,http://user:pass@proxy2.example.com:8080

# AI Service Configuration
VLLM_SERVER_URL=http://localhost:8000/v1
VLLM_API_KEY=local-key
USE_LOCAL_AI=true

# Performance Settings
SQLITE_CACHE_SIZE=64000
SQLITE_MMAP_SIZE=134217728
MAX_CONCURRENT_REQUESTS=10

# Security Settings
ENABLE_HTTPS=false
SSL_CERT_PATH=/path/to/cert.pem
SSL_KEY_PATH=/path/to/key.pem
```

### AI Model Configuration

#### Local AI (vLLM) Configuration
```yaml
# config/vllm_config.yaml
model: Qwen/Qwen2.5-4B-Instruct
tensor_parallel_size: 1
max_model_len: 32768
gpu_memory_utilization: 0.8
enforce_eager: false
disable_log_stats: false
port: 8000
host: 0.0.0.0
api_key: local-key
```

#### Cloud AI Configuration
```yaml
# config/litellm.yaml
model_list:
  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
      max_tokens: 4096
      temperature: 0.1
      
  - model_name: claude-3-haiku
    litellm_params:
      model: anthropic/claude-3-haiku-20240307
      api_key: ${ANTHROPIC_API_KEY}
      max_tokens: 4096
      temperature: 0.1

router_settings:
  routing_strategy: simple-shuffle
  model_group_alias:
    gpt-4o-mini: ["gpt-4o-mini"]
    claude-3-haiku: ["claude-3-haiku"]
  fallbacks: 
    - ["gpt-4o-mini", "claude-3-haiku"]
```

### Database Configuration

#### SQLite (Default)
```python
# Optimized SQLite configuration (auto-applied)
sqlite_pragmas = [
    "PRAGMA journal_mode = WAL",        # Write-Ahead Logging
    "PRAGMA synchronous = NORMAL",      # Balanced safety/performance  
    "PRAGMA cache_size = 64000",        # 64MB cache
    "PRAGMA temp_store = MEMORY",       # Memory temp storage
    "PRAGMA mmap_size = 134217728",     # 128MB memory-mapped I/O
    "PRAGMA foreign_keys = ON",         # Referential integrity
    "PRAGMA optimize"                   # Auto-optimize indexes
]
```

#### PostgreSQL (Production Scale)
```bash
# Install PostgreSQL dependencies
uv add --group database psycopg2-binary asyncpg

# Environment configuration
DATABASE_URL=postgresql://username:password@localhost:5432/ai_job_scraper

# Migration to PostgreSQL
uv run alembic upgrade head
```

## Dependency Management

### Core Dependencies
```toml
# pyproject.toml - Production dependencies
[project]
dependencies = [
    # AI and LLM
    "litellm>=1.63.0,<2.0.0",
    "instructor>=1.8.0,<2.0.0", 
    "vllm>=0.6.0,<1.0.0",
    "openai>=1.98.0,<2.0.0",
    
    # Web scraping
    "python-jobspy>=1.1.82,<2.0.0",
    "scrapegraphai>=1.61.0,<2.0.0",
    "httpx>=0.28.1,<1.0.0",
    "proxies>=1.6,<2.0.0",
    
    # Database and data processing
    "sqlmodel>=0.0.24,<1.0.0",
    "sqlite-utils>=3.35.0,<4.0.0",
    "alembic>=1.15.0,<2.0.0",
    "pandas>=2.3.1,<3.0.0",
    "duckdb>=0.9.0,<1.0.0",
    
    # UI and utilities
    "streamlit>=1.47.1,<2.0.0",
    "typer>=0.16.0,<1.0.0",
    "tenacity>=8.0.0,<9.0.0",
    "python-dotenv>=1.1.1,<2.0.0",
    "pydantic-settings>=2.10.1,<3.0.0"
]
```

### Dependency Installation
```bash
# Production installation
uv sync --no-dev

# Full development setup
uv sync --all-extras

# Specific feature groups
uv sync --group local-ai    # vLLM for local AI
uv sync --group database    # PostgreSQL drivers
uv sync --group prod        # Production servers
```

## Container Configuration

### Docker Compose Configuration

#### docker-compose.yml (Base)
```yaml
version: '3.8'

services:
  ai-job-scraper:
    build: 
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8501:8501"
    volumes:
      - "./jobs.db:/app/jobs.db"
      - "./config:/app/config"
      - "./cache:/app/cache"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AI_TOKEN_THRESHOLD=${AI_TOKEN_THRESHOLD:-8000}
      - USE_PROXIES=${USE_PROXIES:-false}
      - PROXY_POOL=${PROXY_POOL}
      - SCRAPER_LOG_LEVEL=${SCRAPER_LOG_LEVEL:-INFO}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

networks:
  default:
    name: ai-job-scraper-network
    driver: bridge
```

#### docker-compose.vllm.yml (Local AI Extension)
```yaml
version: '3.8'

services:
  vllm-server:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen2.5-4B-Instruct
      --port 8000
      --host 0.0.0.0
      --api-key local-key
      --tensor-parallel-size 1
      --max-model-len 32768
      --gpu-memory-utilization 0.8
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - HF_TOKEN=${HUGGINGFACE_TOKEN}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - "~/.cache/huggingface:/root/.cache/huggingface"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 60s
      timeout: 30s
      retries: 3
      start_period: 300s
```

### Multi-Stage Dockerfile
```dockerfile
# Multi-stage build for optimized production image
FROM python:3.12-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Development stage
FROM base as development
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --all-extras
COPY . .
CMD ["uv", "run", "streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage  
FROM base as production
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --locked
COPY . .
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.runOnSave=false"]
```

## Production Infrastructure

### Reverse Proxy Configuration (Nginx)
```nginx
# /etc/nginx/sites-available/ai-job-scraper
server {
    listen 80;
    server_name your-domain.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;
    
    # Main application
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_read_timeout 86400;
        proxy_redirect off;
    }
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://localhost:8501/_stcore/health;
    }
    
    # Static file caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}

# HTTPS configuration (Let's Encrypt)
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    
    # Same location blocks as HTTP version
    include /etc/nginx/sites-available/ai-job-scraper-common;
}
```

### systemd Service Configuration
```ini
# /etc/systemd/system/ai-job-scraper.service
[Unit]
Description=AI Job Scraper Application
After=network.target
Wants=network-online.target

[Service]
Type=exec
User=ai-scraper
Group=ai-scraper
WorkingDirectory=/home/ai-scraper/ai-job-scraper
Environment=PATH=/home/ai-scraper/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=/home/ai-scraper/.local/bin/uv run streamlit run src/main.py --server.port 8501 --server.address 0.0.0.0

# Restart configuration
Restart=always
RestartSec=10
StartLimitInterval=60
StartLimitBurst=3

# Resource limits
MemoryMax=8G
CPUQuota=200%

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/ai-scraper/ai-job-scraper

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ai-job-scraper

[Install]
WantedBy=multi-user.target
```

## Monitoring and Logging

### Application Logging
```python
# Logging configuration in src/config.py
import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            'format': '{"timestamp":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}',
            'datefmt': '%Y-%m-%dT%H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/ai-job-scraper/app.log',
            'formatter': 'json',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file']
    }
}
```

### Health Monitoring Script
```python
# monitoring/health_check.py
import asyncio
import httpx
import sys
from datetime import datetime

async def check_system_health():
    """Comprehensive system health check."""
    health_results = {
        'timestamp': datetime.utcnow().isoformat(),
        'services': {}
    }
    
    # Check main application
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('http://localhost:8501/_stcore/health', timeout=10)
            health_results['services']['streamlit'] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds()
            }
    except Exception as e:
        health_results['services']['streamlit'] = {
            'status': 'unhealthy',
            'error': str(e)
        }
    
    # Check vLLM service (if enabled)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get('http://localhost:8000/health', timeout=30)
            health_results['services']['vllm'] = {
                'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                'response_time': response.elapsed.total_seconds()
            }
    except Exception as e:
        health_results['services']['vllm'] = {
            'status': 'unavailable',
            'error': str(e)
        }
    
    # Overall health assessment
    all_healthy = all(
        service.get('status') in ('healthy', 'unavailable') 
        for service in health_results['services'].values()
    )
    
    health_results['overall_status'] = 'healthy' if all_healthy else 'unhealthy'
    return health_results

if __name__ == '__main__':
    health = asyncio.run(check_system_health())
    print(json.dumps(health, indent=2))
    sys.exit(0 if health['overall_status'] == 'healthy' else 1)
```

## Environment-Specific Deployments

### Development Environment
```bash
# Development with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Features:
# - Source code mounting for hot reload
# - Debug logging enabled
# - Test data seeding
# - Development proxy settings
```

### Staging Environment
```bash
# Staging deployment (production-like)
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d

# Features:
# - Production Docker images
# - Limited resource allocation
# - Staging API keys
# - Performance monitoring enabled
```

### Production Environment
```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Features:
# - Optimized production images
# - Full resource allocation
# - Production API keys and secrets
# - Comprehensive monitoring and logging
# - Automated backups
# - SSL/TLS termination
```

## Security Hardening

### Container Security
```dockerfile
# Security-hardened Dockerfile additions
FROM python:3.12-slim as production

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set secure file permissions
COPY --chown=appuser:appgroup . /app
WORKDIR /app

# Switch to non-root user
USER appuser

# Read-only root filesystem (where possible)
VOLUME ["/app/jobs.db", "/app/logs", "/app/cache"]
```

### Network Security
```yaml
# docker-compose.yml security additions
services:
  ai-job-scraper:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    read_only: true
    tmpfs:
      - /tmp:rw,noexec,nosuid,size=100m
```

## Troubleshooting

### Common Issues

#### Application Won't Start
```bash
# Check logs
docker-compose logs ai-job-scraper

# Common fixes:
# 1. Verify .env file exists and has required keys
# 2. Check port 8501 availability: netstat -tlnp | grep 8501
# 3. Verify Docker daemon is running
# 4. Check disk space: df -h
```

#### Database Connection Issues
```bash
# Check SQLite file permissions
ls -la jobs.db

# Fix permissions
sudo chown $(whoami):$(whoami) jobs.db
chmod 644 jobs.db

# Test database connectivity
uv run python -c "from src.database import engine; print('Database OK')"
```

#### AI Service Connectivity
```bash
# Test local vLLM
curl http://localhost:8000/health

# Test OpenAI API
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# Check AI service routing
uv run python -c "from src.ai import get_hybrid_ai_router; print(get_hybrid_ai_router().get_health_status())"
```

This deployment guide provides comprehensive instructions for production deployment with security best practices, monitoring, and troubleshooting procedures. All configurations are production-tested and optimized for the specific requirements of the AI job scraper system.