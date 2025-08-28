# AI Job Scraper - Comprehensive Deployment Guide

**Version**: 1.0  
**Date**: 2025-08-27  
**Status**: Production Ready  

## Table of Contents

1. [Quick Start](#quick-start)
2. [Deployment Options](#deployment-options)
3. [Local AI (vLLM) Deployment](#local-ai-vllm-deployment)
4. [Configuration Management](#configuration-management)
5. [Production Infrastructure](#production-infrastructure)
6. [Monitoring and Logging](#monitoring-and-logging)
7. [Security Hardening](#security-hardening)
8. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.12+ with uv package manager
- Docker & Docker Compose (recommended)
- 4GB+ RAM (24GB recommended for local AI with RTX 4090)
- 2GB+ available disk space (10GB+ for local AI models)

### 1-Command Deployment

```bash
# Clone and deploy with Docker
git clone https://github.com/BjornMelin/ai-job-scraper.git
cd ai-job-scraper
cp .env.example .env  # Edit with your API keys
docker-compose up -d
```

Access the application at: <http://localhost:8501>

## Deployment Options

### Option 1: Docker Deployment (Recommended)

#### Cloud AI Only

```bash
# Deploy with cloud AI fallback only
docker-compose up -d ai-job-scraper

# Requires OPENAI_API_KEY in .env file
```

#### Full Stack with Local AI

```bash
# Deploy complete system with vLLM local AI
docker-compose -f docker-compose.yml -f docker-compose.vllm.yml up -d

# Services running:
# - ai-job-scraper: Main application (port 8501)  
# - vllm-server: Local AI server (port 8000)
# - nginx: Reverse proxy (port 80)
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

## Local AI (vLLM) Deployment

The AI Job Scraper supports local AI inference using vLLM for cost-effective, privacy-focused AI processing. This section provides comprehensive vLLM deployment instructions.

### Architecture Overview

```mermaid
graph TD
    A[Client Application] --> B[vLLM Server :8000]
    B --> C[Qwen2.5-4B-Instruct]
    C --> D[NVIDIA RTX 4090 GPU]
    
    B --> E[Health Check :8000/health]
    B --> F[OpenAI API :8000/v1/*]
    B --> G[Metrics :8001/metrics]
    
    H[Prometheus :9090] --> G
    
    subgraph "vLLM Features"
        I[FP8 Quantization]
        J[Prefix Caching]
        K[Continuous Batching]
        L[Structured Output]
        M[Swap Space: 4GB]
    end
    
    C --> I
    C --> J
    C --> K
    C --> L
    C --> M
```

### Hardware Requirements

#### Minimum Requirements (RTX 4090 Optimized)

- **GPU**: NVIDIA RTX 4090 (24GB VRAM) with Ada Lovelace architecture (Compute Capability 8.9)
- **System RAM**: 20GB (16GB for model + 4GB swap space)
- **Storage**: 10GB free space for model caching
- **CUDA**: Version 12.1 or higher
- **Docker**: With NVIDIA runtime support

#### Optimal Configuration

- **GPU Memory Utilization**: 90% (21.6GB out of 24GB)
- **Swap Space**: 4GB CPU memory for overflow handling
- **Host RAM**: 20GB total allocation
- **PCIe**: 16x slot for optimal bandwidth

### vLLM Quick Start

#### 1. Prerequisites Check

```bash
# Verify GPU availability
nvidia-smi

# Check CUDA version (should be 12.1+)
nvcc --version

# Verify Docker with NVIDIA support
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu20.04 nvidia-smi
```

#### 2. vLLM Environment Setup

```bash
# Set API key (recommended)
export VLLM_API_KEY="your-secure-api-key-here"

# Verify project structure
ls -la docker-compose.vllm.yml scripts/start_vllm.sh
```

#### 3. Start vLLM Server

```bash
# Using the startup script (recommended)
./scripts/start_vllm.sh

# Or directly with Docker Compose
docker-compose -f docker-compose.vllm.yml up -d
```

#### 4. Validate vLLM Deployment

```bash
# Run comprehensive validation
python scripts/validate_vllm.py

# Quick health check
curl http://localhost:8000/health
```

### vLLM Configuration

#### Docker Compose Configuration

```yaml
# docker-compose.vllm.yml
version: '3.8'

services:
  vllm-server:
    image: vllm/vllm-openai:latest
    command: >
      --model Qwen/Qwen2.5-4B-Instruct
      --quantization fp8
      --max-model-len 8192
      --gpu-memory-utilization 0.9
      --swap-space 4
      --enable-prefix-caching
      --max-num-seqs 128
      --port 8000
      --host 0.0.0.0
      --api-key local-key
    ports:
      - "8000:8000"
      - "8001:8001"  # Metrics
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

#### Performance Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gpu_memory_utilization` | 0.9 | 90% GPU memory usage |
| `swap_space` | 4 | 4GB CPU memory for overflow |
| `max_model_len` | 8192 | Maximum context window |
| `max_num_seqs` | 128 | Concurrent request limit |
| `enable_prefix_caching` | true | Optimize repeated patterns |
| `enable_chunked_prefill` | true | Improve batching efficiency |

### vLLM API Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Models List

```bash
curl -H "Authorization: Bearer $VLLM_API_KEY" \
     http://localhost:8000/v1/models
```

#### Chat Completion

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-4B-Instruct",
    "messages": [
      {"role": "user", "content": "Extract job info: Software Engineer at TechCorp"}
    ],
    "max_tokens": 100
  }'
```

#### Structured Output

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-4B-Instruct",
    "messages": [
      {"role": "user", "content": "Extract job info: Software Engineer at TechCorp"}
    ],
    "extra_body": {
      "guided_json": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "company": {"type": "string"}
        }
      }
    }
  }'
```

### Integration with LiteLLM

Configure LiteLLM to use the vLLM server as specified in the hybrid AI architecture:

```yaml
# config/litellm.yaml
model_list:
  - model_name: local-qwen
    litellm_params:
      model: Qwen2.5-4B-Instruct
      api_base: http://localhost:8000/v1
      api_key: ${VLLM_API_KEY}
      timeout: 30

  - model_name: gpt-4o-mini
    litellm_params:
      model: openai/gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
      max_tokens: 4096
      temperature: 0.1

router_settings:
  routing_strategy: simple-shuffle
  fallbacks: 
    - ["local-qwen", "gpt-4o-mini"]
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

## Production Infrastructure

### Container Configuration

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

networks:
  default:
    name: ai-job-scraper-network
    driver: bridge
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

# Resource limits
MemoryMax=8G
CPUQuota=200%

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/home/ai-scraper/ai-job-scraper

[Install]
WantedBy=multi-user.target
```

## Monitoring and Logging

### vLLM Monitoring

```bash
# View available metrics
curl http://localhost:8001/metrics

# Start Prometheus monitoring (optional)
docker-compose -f docker-compose.vllm.yml --profile monitoring up -d
```

#### Key vLLM Metrics to Monitor

- **Request Latency**: P95 < 2000ms for typical requests
- **GPU Memory Usage**: Should stay around 90%
- **Swap Space Usage**: Monitor for overflow situations
- **Request Queue Depth**: Track concurrent request handling
- **Error Rate**: Should be < 1%

### Application Logging

```python
# Logging configuration in src/config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'json': {
            'format': '{"timestamp":"%(asctime)s","name":"%(name)s","level":"%(levelname)s","message":"%(message)s"}'
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
    }
}
```

### Health Monitoring Script

```python
# monitoring/health_check.py
import asyncio
import httpx
import json
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

### vLLM Security

```bash
# Generate secure API key for vLLM
export VLLM_API_KEY=$(openssl rand -hex 32)

# Store in .env file
echo "VLLM_API_KEY=$VLLM_API_KEY" >> .env
```

## Troubleshooting

### Common Application Issues

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

### vLLM-Specific Troubleshooting

#### GPU Not Available

**Error**: `RuntimeError: No GPU available`

**Solutions**:

```bash
# Check NVIDIA Docker runtime
docker info | grep nvidia

# Install nvidia-container-toolkit if missing
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

#### Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:

- Reduce `gpu_memory_utilization` to 0.8
- Increase `swap_space` to 8GB
- Reduce `max_num_seqs` to 64

#### Model Download Issues

**Error**: `Failed to download model`

**Solutions**:

```bash
# Pre-download model
docker run --rm -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-4B-Instruct --download-dir /root/.cache/huggingface
```

#### vLLM Performance Issues

```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Check system resources
htop

# Monitor request latency
python scripts/validate_vllm.py --timeout 120
```

### AI Service Connectivity

```bash
# Test local vLLM
curl http://localhost:8000/health

# Test OpenAI API
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models

# Check AI service routing
uv run python -c "from src.ai import get_hybrid_ai_router; print(get_hybrid_ai_router().get_health_status())"
```

### Log Analysis

#### vLLM Successful Startup

```text
INFO: Started server process
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Initializing an LLM engine with config:
INFO: Loading model weights took 2.34 GB GPU memory
```

#### vLLM Error Patterns

```text
ERROR: CUDA out of memory
ERROR: Failed to allocate memory
ERROR: Failed to load model
ERROR: Model not found in cache
```

## Performance Optimization

### vLLM Memory Optimization

```yaml
# Adjust for different memory constraints
command: >
  --gpu-memory-utilization 0.85  # Reduce from 0.9 if needed
  --swap-space 8                 # Increase for large requests
  --max-num-seqs 64              # Reduce for memory-constrained environments
```

### vLLM Request Optimization

```yaml
# Enable all caching and optimization
command: >
  --enable-prefix-caching
  --enable-chunked-prefill
  --max-num-batched-tokens 4096
  --scheduler-delay-factor 0.0
```

## Environment-Specific Deployments

### Development Environment

```bash
# Development with hot reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Production Environment

```bash
# Production deployment with vLLM
docker-compose -f docker-compose.yml -f docker-compose.vllm.yml -f docker-compose.prod.yml up -d
```

---

This comprehensive deployment guide provides complete instructions for deploying the AI Job Scraper application across all environments, with specialized focus on local AI deployment using vLLM for optimal cost-effectiveness and privacy.

For additional support, refer to the [Operations Manual](./operations-manual.md) or check the [Performance Benchmarks](./performance-benchmarks.md) for expected system performance metrics.
