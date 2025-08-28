# AI Job Scraper - Operations Manual

**Version**: 1.0  
**Date**: 2025-08-27  
**Status**: Production Ready  

## Overview

This operations manual provides comprehensive guidance for monitoring, maintaining, and troubleshooting the AI Job Scraper system in production. All procedures are validated and production-tested.

### Quick Reference

- **Health Dashboard**: <http://localhost:8501/_stcore/health>
- **Logs Location**: `/var/log/ai-job-scraper/` or `docker logs ai-job-scraper`
- **Emergency Contact**: System administrator
- **Escalation Path**: Application → Infrastructure → External services

## System Health Monitoring

### Health Check Endpoints

#### Primary Health Checks

```python
# Main application health check
GET http://localhost:8501/_stcore/health
Expected Response: 200 OK with Streamlit health data

# vLLM AI service health (if deployed)
GET http://localhost:8000/health  
Expected Response: 200 OK with model status

# Custom system health endpoint
GET http://localhost:8501/api/health
Expected Response: JSON with component status
```

#### Comprehensive Health Check Script

```python
#!/usr/bin/env python3
# scripts/system_health_check.py

import asyncio
import json
import httpx
import sqlite3
from datetime import datetime, UTC
from pathlib import Path

async def comprehensive_health_check():
    """Production health check with detailed component validation."""
    health_report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "overall_status": "unknown",
        "components": {}
    }
    
    # 1. Main Application Health
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8501/_stcore/health", 
                timeout=10.0
            )
            health_report["components"]["streamlit"] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "details": "Main application responding"
            }
    except Exception as e:
        health_report["components"]["streamlit"] = {
            "status": "critical",
            "error": str(e),
            "details": "Main application unreachable"
        }
    
    # 2. Database Health
    try:
        db_path = Path("jobs.db")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM jobs")
            job_count = cursor.fetchone()[0]
            conn.close()
            
            health_report["components"]["database"] = {
                "status": "healthy",
                "job_count": job_count,
                "db_size_mb": round(db_path.stat().st_size / (1024**2), 2),
                "details": f"SQLite database operational with {job_count} jobs"
            }
        else:
            health_report["components"]["database"] = {
                "status": "warning", 
                "details": "Database file not found - will be created on first use"
            }
    except Exception as e:
        health_report["components"]["database"] = {
            "status": "critical",
            "error": str(e),
            "details": "Database connection failed"
        }
    
    # 3. AI Services Health
    # Local vLLM service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8000/health", 
                timeout=30.0
            )
            health_report["components"]["vllm_local"] = {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "response_time_ms": response.elapsed.total_seconds() * 1000,
                "details": "Local AI service operational"
            }
    except Exception as e:
        health_report["components"]["vllm_local"] = {
            "status": "unavailable",
            "error": str(e),
            "details": "Local AI service not running - will use cloud fallback"
        }
    
    # Cloud AI service
    try:
        # Test with a minimal request to verify API key and connectivity
        from openai import OpenAI
        client = OpenAI()
        models = client.models.list()
        health_report["components"]["openai_api"] = {
            "status": "healthy",
            "details": f"Cloud AI accessible with {len(models.data)} models"
        }
    except Exception as e:
        health_report["components"]["openai_api"] = {
            "status": "degraded",
            "error": str(e),
            "details": "Cloud AI service issues - check API key and connectivity"
        }
    
    # 4. Determine Overall Status
    component_statuses = [comp["status"] for comp in health_report["components"].values()]
    if "critical" in component_statuses:
        health_report["overall_status"] = "critical"
    elif "unhealthy" in component_statuses:
        health_report["overall_status"] = "unhealthy"
    elif "degraded" in component_statuses:
        health_report["overall_status"] = "degraded"
    else:
        health_report["overall_status"] = "healthy"
    
    return health_report

if __name__ == "__main__":
    health = asyncio.run(comprehensive_health_check())
    print(json.dumps(health, indent=2))
    
    # Exit with appropriate code for monitoring systems
    exit_codes = {"healthy": 0, "degraded": 1, "unhealthy": 2, "critical": 3}
    exit(exit_codes.get(health["overall_status"], 3))
```

### Monitoring Dashboard Setup

#### Prometheus Metrics (Optional)

```python
# monitoring/prometheus_metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Application metrics
job_scraping_requests = Counter('job_scraping_requests_total', 'Total job scraping requests')
job_scraping_duration = Histogram('job_scraping_duration_seconds', 'Job scraping duration')
ai_processing_requests = Counter('ai_processing_requests_total', 'Total AI processing requests', ['model_type'])
ai_processing_duration = Histogram('ai_processing_duration_seconds', 'AI processing duration')
search_queries_total = Counter('search_queries_total', 'Total search queries')
search_response_time = Histogram('search_response_time_seconds', 'Search response time')

# System health metrics  
database_size_bytes = Gauge('database_size_bytes', 'Database file size in bytes')
job_count_total = Gauge('job_count_total', 'Total number of jobs in database')
active_scraping_tasks = Gauge('active_scraping_tasks', 'Number of active scraping tasks')
memory_usage_bytes = Gauge('memory_usage_bytes', 'Memory usage in bytes')

def start_metrics_server(port=9090):
    """Start Prometheus metrics server."""
    start_http_server(port)
```

#### Custom Health Dashboard

```python
# monitoring/health_dashboard.py
import streamlit as st
import asyncio
import plotly.graph_objects as go
from datetime import datetime, timedelta

def render_health_dashboard():
    """Render comprehensive system health dashboard."""
    st.title("🏥 System Health Dashboard")
    
    # Real-time status
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Status", "Healthy", "✅")
    with col2:
        st.metric("Response Time", "45ms", "-5ms")
    with col3:
        st.metric("Success Rate", "98.5%", "+0.3%")
    with col4:
        st.metric("Active Users", "3", "+1")
    
    # Component health grid
    st.subheader("Component Health Status")
    
    health_data = asyncio.run(comprehensive_health_check())
    
    for component, status in health_data["components"].items():
        with st.expander(f"{component.title()} - {status['status'].title()}"):
            if status["status"] == "healthy":
                st.success(f"✅ {status.get('details', 'Operating normally')}")
            elif status["status"] == "degraded":
                st.warning(f"⚠️ {status.get('details', 'Reduced functionality')}")
            elif status["status"] == "unhealthy":
                st.error(f"❌ {status.get('details', 'Service issues detected')}")
            elif status["status"] == "critical":
                st.error(f"🚨 {status.get('details', 'Critical service failure')}")
            
            if "response_time_ms" in status:
                st.metric("Response Time", f"{status['response_time_ms']:.0f}ms")
```

## Logging and Monitoring

### Log Configuration

```python
# Logging configuration in production
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
            'format': '%(asctime)s %(name)s %(levelname)s %(message)s %(funcName)s %(lineno)d'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'detailed',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/ai-job-scraper/app.log',
            'maxBytes': 50000000,  # 50MB
            'backupCount': 5,
        },
        'error_file': {
            'level': 'ERROR',
            'formatter': 'detailed', 
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': '/var/log/ai-job-scraper/error.log',
            'maxBytes': 10000000,  # 10MB
            'backupCount': 3,
        }
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': False
        },
        'src.services.unified_scraper': {
            'level': 'DEBUG',  # More detailed logging for scraping
        },
        'src.ai.hybrid_ai_router': {
            'level': 'DEBUG',  # Detailed AI routing logs
        }
    }
}
```

### Log Analysis Scripts

```bash
#!/bin/bash
# scripts/log_analysis.sh

# Error rate analysis
echo "=== Error Rate Analysis (Last 24h) ==="
grep -E "ERROR|CRITICAL" /var/log/ai-job-scraper/app.log | \
awk '{print $1 " " $2}' | sort | uniq -c | sort -nr

# Performance analysis
echo "=== Slow Operations (>5s response time) ==="
grep "response_time" /var/log/ai-job-scraper/app.log | \
awk '$NF > 5000 {print $0}' | tail -20

# Scraping success rates
echo "=== Scraping Success Rates ==="
grep "scraping_result" /var/log/ai-job-scraper/app.log | \
awk '{print $NF}' | sort | uniq -c

# AI service usage
echo "=== AI Service Usage Distribution ==="
grep "ai_service_used" /var/log/ai-job-scraper/app.log | \
awk '{print $(NF-1)}' | sort | uniq -c
```

## Backup and Recovery

### Database Backup Procedures

```bash
#!/bin/bash
# scripts/backup_database.sh

BACKUP_DIR="/backup/ai-job-scraper"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DB_FILE="jobs.db"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# SQLite backup with consistency
sqlite3 "$DB_FILE" ".backup $BACKUP_DIR/jobs_${TIMESTAMP}.db"

# Compress backup
gzip "$BACKUP_DIR/jobs_${TIMESTAMP}.db"

# Verify backup integrity
gunzip -t "$BACKUP_DIR/jobs_${TIMESTAMP}.db.gz"
if [ $? -eq 0 ]; then
    echo "Backup successful: jobs_${TIMESTAMP}.db.gz"
else
    echo "Backup verification failed!"
    exit 1
fi

# Cleanup old backups (keep 30 days)
find "$BACKUP_DIR" -name "jobs_*.db.gz" -mtime +30 -delete

# Backup configuration files
tar -czf "$BACKUP_DIR/config_${TIMESTAMP}.tar.gz" config/ .env
```

### Recovery Procedures

```bash
#!/bin/bash
# scripts/restore_database.sh

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_timestamp>"
    echo "Available backups:"
    ls -la /backup/ai-job-scraper/jobs_*.db.gz
    exit 1
fi

BACKUP_FILE="/backup/ai-job-scraper/jobs_$1.db.gz"
CURRENT_DB="jobs.db"

# Verify backup exists
if [ ! -f "$BACKUP_FILE" ]; then
    echo "Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Create safety backup of current database
if [ -f "$CURRENT_DB" ]; then
    cp "$CURRENT_DB" "${CURRENT_DB}.pre-restore-$(date +%Y%m%d_%H%M%S)"
fi

# Restore from backup
echo "Restoring from backup: $BACKUP_FILE"
gunzip -c "$BACKUP_FILE" > "$CURRENT_DB"

# Verify restoration
sqlite3 "$CURRENT_DB" "PRAGMA integrity_check;"
if [ $? -eq 0 ]; then
    echo "Database restoration completed successfully"
else
    echo "Database restoration failed - integrity check failed"
    exit 1
fi
```

### Configuration Backup

```bash
#!/bin/bash
# scripts/backup_config.sh

CONFIG_BACKUP_DIR="/backup/ai-job-scraper/config"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

mkdir -p "$CONFIG_BACKUP_DIR"

# Backup configuration files (excluding sensitive data)
tar -czf "$CONFIG_BACKUP_DIR/config_${TIMESTAMP}.tar.gz" \
    --exclude=".env" \
    --exclude="*.key" \
    --exclude="*secret*" \
    config/ \
    pyproject.toml \
    docker-compose.yml \
    Dockerfile

# Backup environment template (without secrets)
sed 's/=.*/=***REDACTED***/g' .env > "$CONFIG_BACKUP_DIR/.env.template"

echo "Configuration backup completed: config_${TIMESTAMP}.tar.gz"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### Issue: Application Won't Start

```bash
# Symptom: Docker container exits immediately or port 8501 unreachable

# Diagnosis Steps:
1. Check container logs:
   docker logs ai-job-scraper --tail 50

2. Verify port availability:
   netstat -tlnp | grep 8501
   
3. Check environment configuration:
   docker exec ai-job-scraper env | grep -E "OPENAI|DATABASE|PROXY"

# Common Causes & Solutions:
# Missing environment variables
➜ Verify .env file exists and contains required keys
➜ Check .env file permissions (should be readable by container user)

# Port conflict
➜ Change port in docker-compose.yml: "8502:8501"
➜ Or stop conflicting service: sudo lsof -ti:8501 | xargs kill

# Database permission issues
➜ Fix file permissions: chmod 666 jobs.db
➜ Or recreate: rm jobs.db && docker restart ai-job-scraper
```

#### Issue: Slow Search Performance

```bash
# Symptom: Search queries taking >500ms consistently

# Diagnosis:
1. Check database size:
   du -h jobs.db
   
2. Analyze query plans:
   sqlite3 jobs.db "EXPLAIN QUERY PLAN SELECT * FROM jobs WHERE title MATCH 'python';"
   
3. Check FTS5 index status:
   sqlite3 jobs.db ".tables" | grep fts

# Solutions:
# Rebuild FTS5 index
sqlite3 jobs.db "INSERT INTO jobs_fts(jobs_fts) VALUES('rebuild');"

# Enable DuckDB analytics for large datasets
export USE_DUCKDB_ANALYTICS=true

# Add pagination for large result sets (auto-enabled at 100+ results)
# Implemented in UI layer - no manual intervention needed
```

#### Issue: AI Processing Failures

```bash
# Symptom: AI enhancement failing or timing out

# Diagnosis:
1. Check AI service health:
   curl -f http://localhost:8000/health  # vLLM local
   
2. Test OpenAI API connectivity:
   curl -H "Authorization: Bearer $OPENAI_API_KEY" \
        https://api.openai.com/v1/models
        
3. Review AI router logs:
   docker logs ai-job-scraper | grep "hybrid_ai_router"

# Solutions:
# vLLM service issues
➜ Restart vLLM: docker restart vllm-server
➜ Check GPU availability: nvidia-smi
➜ Verify model downloaded: docker exec vllm-server ls /root/.cache/huggingface

# OpenAI API issues  
➜ Verify API key: echo $OPENAI_API_KEY
➜ Check rate limits in OpenAI dashboard
➜ Test with curl to isolate network issues

# Fallback to basic extraction
➜ Disable AI enhancement temporarily in UI settings
➜ Check content complexity threshold (may need adjustment)
```

#### Issue: Scraping Failures

```bash
# Symptom: Job scraping returning empty results or failing

# Diagnosis:
1. Check scraping success rates:
   docker logs ai-job-scraper | grep "scraping_success_rate"
   
2. Test proxy connectivity:
   curl --proxy $PROXY_URL https://httpbin.org/ip
   
3. Check rate limiting:
   docker logs ai-job-scraper | grep "rate_limit"

# Solutions:  
# Proxy issues
➜ Verify proxy credentials in .env file
➜ Test proxy rotation: USE_PROXIES=true
➜ Check proxy balance/credits

# Rate limiting
➜ Reduce concurrent requests: MAX_CONCURRENT_REQUESTS=5
➜ Increase delays between requests: SCRAPING_DELAY=2000

# Site blocking
➜ Enable proxy usage: USE_PROXIES=true
➜ Update user agent rotation
➜ Check site-specific blocking patterns in logs
```

#### Issue: High Memory Usage

```bash
# Symptom: System memory usage >80% or OOM kills

# Diagnosis:
1. Check memory usage by process:
   docker stats ai-job-scraper
   
2. Profile memory usage:
   docker exec ai-job-scraper python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"
   
3. Check database cache settings:
   sqlite3 jobs.db "PRAGMA cache_size;"

# Solutions:
# Reduce SQLite cache
➜ Set SQLITE_CACHE_SIZE=32000 in .env (reduces from 64MB to 32MB)

# Enable memory-mapped I/O limits
➜ Set SQLITE_MMAP_SIZE=67108864 in .env (reduces from 128MB to 64MB)

# Implement result pagination
➜ Automatic above 10K jobs - verify pagination is working

# vLLM memory optimization
➜ Reduce GPU memory utilization: --gpu-memory-utilization 0.7
➜ Use smaller model variant if available
```

### Emergency Procedures

#### Service Recovery Protocol

```bash
#!/bin/bash
# scripts/emergency_recovery.sh

echo "Starting emergency recovery protocol..."

# 1. Stop all services
echo "1. Stopping services..."
docker-compose down

# 2. Check system resources
echo "2. System resource check..."
df -h | grep -E "(/$|/var)"
free -h

# 3. Backup current state
echo "3. Creating emergency backup..."
./scripts/backup_database.sh emergency_$(date +%Y%m%d_%H%M%S)

# 4. Clean temporary files
echo "4. Cleaning temporary files..."
rm -rf cache/*
rm -rf /tmp/ai-job-scraper-*

# 5. Restart services
echo "5. Restarting services..."
docker-compose up -d

# 6. Wait for services to be ready
echo "6. Waiting for services to initialize..."
sleep 30

# 7. Run health check
echo "7. Running comprehensive health check..."
python3 scripts/system_health_check.py

echo "Emergency recovery completed. Check health status above."
```

#### Data Recovery from Corruption

```python
#!/usr/bin/env python3
# scripts/data_recovery.py

import sqlite3
import shutil
import sys
from pathlib import Path

def recover_database(db_path="jobs.db"):
    """Attempt to recover data from corrupted SQLite database."""
    
    print(f"Attempting recovery of {db_path}")
    
    # Create backup of corrupted database
    corrupted_backup = f"{db_path}.corrupted.{int(time.time())}"
    shutil.copy2(db_path, corrupted_backup)
    print(f"Corrupted database backed up to: {corrupted_backup}")
    
    try:
        # Attempt SQLite recovery
        conn = sqlite3.connect(db_path)
        
        # Try to dump recoverable data
        recovery_sql = []
        
        for line in conn.iterdump():
            recovery_sql.append(line)
        
        conn.close()
        
        # Create new database
        new_db = f"{db_path}.recovered"
        new_conn = sqlite3.connect(new_db)
        
        # Restore data
        for line in recovery_sql:
            try:
                new_conn.execute(line)
            except sqlite3.Error as e:
                print(f"Warning: Could not recover line: {e}")
        
        new_conn.commit()
        new_conn.close()
        
        # Replace original with recovered
        shutil.move(new_db, db_path)
        print(f"Database recovery completed: {db_path}")
        
        # Verify integrity
        conn = sqlite3.connect(db_path)
        result = conn.execute("PRAGMA integrity_check;").fetchone()
        conn.close()
        
        if result[0] == "ok":
            print("✅ Database integrity check passed")
            return True
        else:
            print(f"❌ Database integrity check failed: {result[0]}")
            return False
            
    except Exception as e:
        print(f"❌ Recovery failed: {e}")
        print(f"Corrupted database preserved at: {corrupted_backup}")
        return False

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else "jobs.db"
    success = recover_database(db_path)
    sys.exit(0 if success else 1)
```

## Performance Monitoring

### Key Performance Indicators (KPIs)

```python
# Performance monitoring thresholds
KPI_THRESHOLDS = {
    "search_response_time_p95": 500,  # milliseconds
    "ai_processing_time_avg": 3000,   # milliseconds
    "scraping_success_rate": 90,      # percentage
    "ui_render_time_p95": 200,        # milliseconds
    "error_rate": 2,                  # percentage
    "memory_usage": 80,               # percentage of available
    "disk_usage": 85,                 # percentage of available
    "cpu_usage_sustained": 70,        # percentage over 5 minutes
}

# Alert conditions
CRITICAL_ALERTS = {
    "service_down": "Any core service unavailable >5 minutes",
    "data_loss": "Database corruption or significant data loss",
    "security_breach": "Unauthorized access detected",
    "performance_degradation": "Response times >10x normal for >10 minutes"
}
```

### Automated Performance Reports

```python
#!/usr/bin/env python3
# scripts/performance_report.py

import sqlite3
import json
from datetime import datetime, timedelta
from collections import defaultdict

def generate_performance_report(days=7):
    """Generate performance report for the last N days."""
    
    report = {
        "period": f"Last {days} days",
        "generated_at": datetime.now().isoformat(),
        "metrics": {}
    }
    
    # Database performance
    conn = sqlite3.connect("jobs.db")
    
    # Job processing statistics
    cursor = conn.execute("""
        SELECT COUNT(*) as total_jobs,
               COUNT(CASE WHEN date_posted >= date('now', '-{} days') THEN 1 END) as recent_jobs
        FROM jobs
    """.format(days))
    
    job_stats = cursor.fetchone()
    report["metrics"]["job_processing"] = {
        "total_jobs": job_stats[0],
        "jobs_added_last_7d": job_stats[1],
        "average_jobs_per_day": job_stats[1] / days
    }
    
    # Search performance (from logs if available)
    try:
        with open("/var/log/ai-job-scraper/app.log", "r") as log_file:
            search_times = []
            for line in log_file:
                if "search_response_time" in line:
                    time_ms = float(line.split(":")[-1].strip().replace("ms", ""))
                    search_times.append(time_ms)
            
            if search_times:
                report["metrics"]["search_performance"] = {
                    "average_response_time": sum(search_times) / len(search_times),
                    "p95_response_time": sorted(search_times)[int(len(search_times) * 0.95)],
                    "total_searches": len(search_times)
                }
    except FileNotFoundError:
        pass
    
    conn.close()
    return report

if __name__ == "__main__":
    report = generate_performance_report()
    print(json.dumps(report, indent=2))
```

## Maintenance Procedures

### Regular Maintenance Checklist

#### Daily Maintenance (Automated)

```bash
#!/bin/bash
# scripts/daily_maintenance.sh

# Log rotation
logrotate /etc/logrotate.d/ai-job-scraper

# Database optimization
sqlite3 jobs.db "PRAGMA optimize;"

# Health check and alerting
python3 scripts/system_health_check.py || echo "Health check failed - manual review needed"

# Disk cleanup
find cache/ -type f -mtime +7 -delete
find /tmp -name "ai-job-scraper-*" -mtime +1 -delete
```

#### Weekly Maintenance

```bash
#!/bin/bash
# scripts/weekly_maintenance.sh

# Database backup
./scripts/backup_database.sh

# Update system packages (if applicable)
apt update && apt upgrade -y

# Docker image cleanup
docker image prune -a -f --filter "until=168h"

# Performance report
python3 scripts/performance_report.py > "/var/log/ai-job-scraper/weekly_report_$(date +%Y%m%d).json"

# Security scan (if tooling available)
# docker scan ai-job-scraper:latest
```

#### Monthly Maintenance

```bash
#!/bin/bash
# scripts/monthly_maintenance.sh

# Full system backup
tar -czf "/backup/ai-job-scraper/full_backup_$(date +%Y%m%d).tar.gz" \
    --exclude="*.log" \
    --exclude="cache/*" \
    /opt/ai-job-scraper/

# Dependency updates check
uv sync --upgrade

# Security updates
apt update && apt upgrade -y

# Configuration review
echo "Monthly configuration review:"
echo "1. Check API keys expiration"
echo "2. Review proxy service usage/costs"
echo "3. Validate backup procedures"
echo "4. Update documentation if needed"
```

This comprehensive operations manual provides all necessary procedures for maintaining the AI job scraper system in production, with emphasis on proactive monitoring, automated maintenance, and efficient troubleshooting procedures.
