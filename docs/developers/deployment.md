# 🚀 Deployment Guide: AI Job Scraper

This guide covers production deployment strategies for the AI Job Scraper, from single-server setups to containerized deployments with monitoring and scaling.

## 🎯 Deployment Options Overview

### Quick Comparison

| Option | Best For | Complexity | Cost | Scalability |
|--------|----------|------------|------|-------------|
| **Local Production** | Personal use, small teams | Low | Free | Limited |
| **VPS Deployment** | Small organizations | Medium | $5-20/month | Medium |
| **Docker + Reverse Proxy** | Professional teams | Medium | $10-50/month | High |
| **Cloud Container** | Enterprise | High | $20-100/month | Very High |

## 🖥️ Local Production Setup

### System Requirements

**Minimum Requirements:**

- Ubuntu 20.04+ / CentOS 8+ / macOS 10.15+

- 2GB RAM, 10GB storage

- Python 3.12+

- Internet connection

**Recommended:**

- 4GB RAM, 20GB storage

- SSD storage for better database performance

- Firewall configured

### Installation Steps

1. **Create deployment user**

   ```bash
   # Create dedicated user for security
   sudo useradd -m -s /bin/bash ai-scraper
   sudo usermod -aG sudo ai-scraper
   su - ai-scraper
   ```

2. **Install system dependencies**

   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install -y python3.12 python3.12-venv git nginx
   
   # CentOS/RHEL
   sudo dnf install -y python3.12 python3.12-venv git nginx
   
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   ```

3. **Deploy application**

   ```bash
   # Clone to production directory
   cd /opt
   sudo git clone https://github.com/BjornMelin/ai-job-scraper.git
   sudo chown -R ai-scraper:ai-scraper ai-job-scraper
   cd ai-job-scraper
   
   # Install dependencies
   uv sync
   
   # Set up environment
   sudo cp .env.example .env
   sudo nano .env  # Add your OpenAI API key
   
   # Initialize database
   uv run python seed.py
   ```

4. **Create systemd service**

   ```bash
   sudo tee /etc/systemd/system/ai-job-scraper.service > /dev/null <<EOF
   [Unit]
   Description=AI Job Scraper
   After=network.target
   
   [Service]
   Type=simple
   User=ai-scraper
   WorkingDirectory=/opt/ai-job-scraper
   Environment=PATH=/opt/ai-job-scraper/.venv/bin
   ExecStart=/opt/ai-job-scraper/.venv/bin/streamlit run app.py --server.port=8501 --server.address=127.0.0.1
   Restart=always
   RestartSec=3
   
   [Install]
   WantedBy=multi-user.target
   EOF
   
   # Enable and start service
   sudo systemctl daemon-reload
   sudo systemctl enable ai-job-scraper
   sudo systemctl start ai-job-scraper
   ```

5. **Configure Nginx reverse proxy**

   ```bash
   sudo tee /etc/nginx/sites-available/ai-job-scraper > /dev/null <<EOF
   server {
       listen 80;
       server_name your-domain.com;  # Replace with your domain
       
       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade \$http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host \$host;
           proxy_set_header X-Real-IP \$remote_addr;
           proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto \$scheme;
           proxy_read_timeout 86400;
       }
   }
   EOF
   
   sudo ln -s /etc/nginx/sites-available/ai-job-scraper /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

6. **Set up SSL (recommended)**

   ```bash
   # Install Certbot
   sudo apt install -y certbot python3-certbot-nginx
   
   # Get SSL certificate
   sudo certbot --nginx -d your-domain.com
   
   # Auto-renewal
   sudo crontab -e
   # Add: 0 12 * * * /usr/bin/certbot renew --quiet
   ```

## 🐳 Docker Production Deployment

### Basic Docker Setup

1. **Create production Docker Compose**

   ```yaml
   # docker-compose.prod.yml
   version: "3.8"
   
   services:
     app:
       build:
         context: .
         dockerfile: Dockerfile.prod
       ports:
         - "8501:8501"
       environment:
         - OPENAI_API_KEY=${OPENAI_API_KEY}
         - DB_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/ai_jobs
       volumes:
         - ./cache:/app/cache
         - ./logs:/app/logs
       depends_on:
         - db
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8501/"]
         interval: 30s
         timeout: 10s
         retries: 3
         
     db:
       image: postgres:15-alpine
       environment:
         - POSTGRES_DB=ai_jobs
         - POSTGRES_USER=postgres
         - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
       volumes:
         - postgres_data:/var/lib/postgresql/data
         - ./backups:/backups
       restart: unless-stopped
       healthcheck:
         test: ["CMD-SHELL", "pg_isready -U postgres"]
         interval: 30s
         timeout: 5s
         retries: 5
         
     nginx:
       image: nginx:alpine
       ports:
         - "80:80"
         - "443:443"
       volumes:
         - ./nginx.conf:/etc/nginx/nginx.conf
         - ./ssl:/etc/nginx/ssl
       depends_on:
         - app
       restart: unless-stopped
         
   volumes:
     postgres_data:
   ```

2. **Production Dockerfile**

   ```dockerfile
   # Dockerfile.prod
   FROM python:3.12-slim as builder
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       curl \
       && rm -rf /var/lib/apt/lists/*
   
   WORKDIR /app
   RUN pip install uv
   COPY pyproject.toml ./
   RUN uv sync --frozen
   
   # Production stage
   FROM python:3.12-slim
   
   # Install runtime dependencies
   RUN apt-get update && apt-get install -y \
       libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
       libcups2 libdbus-1-3 libdrm2 libxkbcommon0 \
       libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
       libgbm1 libasound2 fonts-liberation libappindicator3-1 \
       xdg-utils curl \
       && rm -rf /var/lib/apt/lists/*
   
   # Create non-root user
   RUN useradd -m -u 1000 appuser
   
   WORKDIR /app
   COPY --from=builder /app/.venv /app/.venv
   COPY --chown=appuser:appuser . .
   
   # Install Playwright browsers
   RUN /app/.venv/bin/python -m playwright install --with-deps
   
   USER appuser
   EXPOSE 8501
   
   HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
     CMD curl -f http://localhost:8501/ || exit 1
   
   CMD ["/app/.venv/bin/streamlit", "run", "app.py", \
        "--server.port=8501", "--server.address=0.0.0.0"]
   ```

3. **Nginx configuration**

   ```nginx
   # nginx.conf
   events {
       worker_connections 1024;
   }
   
   http {
       upstream app {
           server app:8501;
       }
       
       server {
           listen 80;
           server_name your-domain.com;
           return 301 https://$server_name$request_uri;
       }
       
       server {
           listen 443 ssl http2;
           server_name your-domain.com;
           
           ssl_certificate /etc/nginx/ssl/cert.pem;
           ssl_certificate_key /etc/nginx/ssl/key.pem;
           
           location / {
               proxy_pass http://app;
               proxy_http_version 1.1;
               proxy_set_header Upgrade $http_upgrade;
               proxy_set_header Connection "upgrade";
               proxy_set_header Host $host;
               proxy_set_header X-Real-IP $remote_addr;
               proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
               proxy_set_header X-Forwarded-Proto $scheme;
               proxy_read_timeout 86400;
           }
           
           location /health {
               access_log off;
               return 200 "healthy\n";
           }
       }
   }
   ```

4. **Deploy with Docker Compose**

   ```bash
   # Create environment file
   cat > .env.prod <<EOF
   OPENAI_API_KEY=your_key_here
   POSTGRES_PASSWORD=secure_password_here
   EOF
   
   # Deploy
   docker-compose -f docker-compose.prod.yml --env-file .env.prod up -d
   
   # Check status
   docker-compose -f docker-compose.prod.yml ps
   docker-compose -f docker-compose.prod.yml logs -f
   ```

## ☁️ Cloud Deployment Options

### AWS Deployment with ECS

1. **Create ECS Task Definition**

   ```json
   {
     "family": "ai-job-scraper",
     "networkMode": "awsvpc",
     "requiresCompatibilities": ["FARGATE"],
     "cpu": "512",
     "memory": "1024",
     "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
     "containerDefinitions": [
       {
         "name": "ai-job-scraper",
         "image": "your-account.dkr.ecr.region.amazonaws.com/ai-job-scraper:latest",
         "portMappings": [
           {
             "containerPort": 8501,
             "protocol": "tcp"
           }
         ],
         "environment": [
           {
             "name": "DB_URL",
             "value": "postgresql://user:pass@rds-endpoint:5432/ai_jobs"
           }
         ],
         "secrets": [
           {
             "name": "OPENAI_API_KEY",
             "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-key"
           }
         ],
         "logConfiguration": {
           "logDriver": "awslogs",
           "options": {
             "awslogs-group": "/ecs/ai-job-scraper",
             "awslogs-region": "us-west-2",
             "awslogs-stream-prefix": "ecs"
           }
         }
       }
     ]
   }
   ```

2. **Deployment script**

   ```bash
   #!/bin/bash
   # deploy-aws.sh
   
   # Build and push image
   aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin account.dkr.ecr.us-west-2.amazonaws.com
   docker build -t ai-job-scraper .
   docker tag ai-job-scraper:latest account.dkr.ecr.us-west-2.amazonaws.com/ai-job-scraper:latest
   docker push account.dkr.ecr.us-west-2.amazonaws.com/ai-job-scraper:latest
   
   # Update service
   aws ecs update-service --cluster ai-job-scraper --service ai-job-scraper --force-new-deployment
   ```

### Google Cloud Run Deployment

1. **Create Cloud Run service**

   ```bash
   # Build and deploy
   gcloud builds submit --tag gcr.io/PROJECT_ID/ai-job-scraper
   
   gcloud run deploy ai-job-scraper \
     --image gcr.io/PROJECT_ID/ai-job-scraper \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars DB_URL="postgresql://user:pass@/ai_jobs?host=/cloudsql/PROJECT_ID:REGION:INSTANCE_ID" \
     --add-cloudsql-instances PROJECT_ID:REGION:INSTANCE_ID \
     --memory=1Gi \
     --cpu=1 \
     --timeout=900
   ```

2. **Set secrets**

   ```bash
   # Store OpenAI API key in Secret Manager
   echo -n "your_openai_key" | gcloud secrets create openai-api-key --data-file=-
   
   # Grant access to Cloud Run
   gcloud secrets add-iam-policy-binding openai-api-key \
     --member="serviceAccount:your-service-account@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/secretmanager.secretAccessor"
   ```

### DigitalOcean App Platform

1. **Create app specification**

   ```yaml
   # .do/app.yaml
   name: ai-job-scraper
   services:
   - name: web
     source_dir: /
     github:
       repo: your-username/ai-job-scraper
       branch: main
     run_command: streamlit run app.py --server.port=8080 --server.address=0.0.0.0
     environment_slug: python
     instance_count: 1
     instance_size_slug: basic-xxs
     http_port: 8080
     envs:
     - key: OPENAI_API_KEY
       scope: RUN_TIME
       type: SECRET
       value: your_openai_api_key
     - key: DB_URL
       scope: RUN_TIME
       value: ${db.DATABASE_URL}
   databases:
   - name: db
     engine: PG
     size: basic-xs
   ```

2. **Deploy**

   ```bash
   # Install doctl
   doctl apps create .do/app.yaml
   
   # Monitor deployment
   doctl apps list
   doctl apps logs your-app-id
   ```

## 📊 Monitoring & Observability

### Application Monitoring

1. **Health check endpoint**

   ```python
   # Add to app.py
   import os
   from datetime import datetime
   
   def health_check():
       """Health check for load balancers and monitoring."""
       try:
           # Test database connection
           session = Session()
           session.execute("SELECT 1")
           session.close()
           
           # Check cache directory
           cache_exists = os.path.exists("cache")
           
           return {
               "status": "healthy",
               "timestamp": datetime.now().isoformat(),
               "database": "connected",
               "cache": "available" if cache_exists else "missing"
           }
       except Exception as e:
           return {
               "status": "unhealthy",
               "error": str(e),
               "timestamp": datetime.now().isoformat()
           }
   
   # Expose health endpoint
   if st.sidebar.button("Health Check"):
       health = health_check()
       if health["status"] == "healthy":
           st.sidebar.success("✅ Application healthy")
       else:
           st.sidebar.error(f"❌ Health check failed: {health['error']}")
   ```

2. **Structured logging**

   ```python
   # Enhanced logging setup
   import logging
   import json
   from logging.handlers import RotatingFileHandler
   
   class JsonFormatter(logging.Formatter):
       def format(self, record):
           log_data = {
               "timestamp": self.formatTime(record),
               "level": record.levelname,
               "logger": record.name,
               "message": record.getMessage(),
               "module": record.module,
               "function": record.funcName,
               "line": record.lineno
           }
           
           if hasattr(record, 'extra_data'):
               log_data.update(record.extra_data)
               
           return json.dumps(log_data)
   
   def setup_production_logging():
       """Configure production-ready logging."""
       logger = logging.getLogger()
       logger.setLevel(logging.INFO)
       
       # Console handler with JSON format
       console_handler = logging.StreamHandler()
       console_handler.setFormatter(JsonFormatter())
       logger.addHandler(console_handler)
       
       # File handler with rotation
       file_handler = RotatingFileHandler(
           'logs/app.log',
           maxBytes=10_000_000,
           backupCount=5
       )
       file_handler.setFormatter(JsonFormatter())
       logger.addHandler(file_handler)
   ```

3. **Performance monitoring**

   ```python
   # Add to scraper.py
   import time
   import psutil
   from dataclasses import dataclass
   from typing import Dict, Any
   
   @dataclass
   class PerformanceMetrics:
       duration: float
       memory_usage: float
       cpu_usage: float
       companies_processed: int
       jobs_found: int
       cache_hit_rate: float
       llm_calls: int
       errors: int
   
   def collect_performance_metrics() -> PerformanceMetrics:
       """Collect comprehensive performance metrics."""
       return PerformanceMetrics(
           duration=time.time() - session_stats["start_time"],
           memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,  # MB
           cpu_usage=psutil.Process().cpu_percent(),
           companies_processed=session_stats["companies_processed"],
           jobs_found=session_stats["jobs_found"],
           cache_hit_rate=session_stats["cache_hits"] / max(session_stats["companies_processed"], 1),
           llm_calls=session_stats["llm_calls"],
           errors=session_stats["errors"]
       )
   
   def log_performance_metrics():
       """Log performance metrics in structured format."""
       metrics = collect_performance_metrics()
       logger.info("Performance metrics collected", extra={
           "extra_data": {
               "metrics": metrics.__dict__,
               "metric_type": "performance_summary"
           }
       })
   ```

### External Monitoring Integration

1. **Prometheus metrics**

   ```python
   # pip install prometheus-client
   from prometheus_client import Counter, Histogram, Gauge, start_http_server
   
   # Define metrics
   SCRAPE_COUNTER = Counter('scrapes_total', 'Total scrapes performed', ['company', 'status'])
   SCRAPE_DURATION = Histogram('scrape_duration_seconds', 'Scrape duration', ['company'])
   JOBS_GAUGE = Gauge('jobs_total', 'Total jobs in database')
   CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate percentage')
   
   def record_scrape_metrics(company: str, duration: float, jobs_count: int, success: bool):
       """Record scraping metrics for Prometheus."""
       status = 'success' if success else 'error'
       SCRAPE_COUNTER.labels(company=company, status=status).inc()
       SCRAPE_DURATION.labels(company=company).observe(duration)
       JOBS_GAUGE.set(jobs_count)
   
   # Start metrics server
   start_http_server(8000)  # Metrics available at :8000/metrics
   ```

2. **Sentry error tracking**

   ```python
   # pip install sentry-sdk
   import sentry_sdk
   from sentry_sdk.integrations.logging import LoggingIntegration
   
   sentry_logging = LoggingIntegration(
       level=logging.INFO,
       event_level=logging.ERROR
   )
   
   sentry_sdk.init(
       dsn="your_sentry_dsn_here",
       integrations=[sentry_logging],
       traces_sample_rate=0.1,
       environment="production"
   )
   
   # Usage in code
   try:
       jobs = await extract_jobs(url, company)
   except Exception as e:
       sentry_sdk.capture_exception(e)
       logger.error(f"Scraping failed for {company}: {e}")
   ```

## 🔒 Security Hardening

### Application Security

1. **Environment variable security**

   ```bash
   # Use strong passwords
   openssl rand -base64 32  # For database passwords
   
   # Secure file permissions
   chmod 600 .env
   chown ai-scraper:ai-scraper .env
   
   # Use secrets management
   # AWS: AWS Secrets Manager
   # GCP: Secret Manager
   # Azure: Key Vault
   ```

2. **Network security**

   ```bash
   # Configure firewall (Ubuntu/Debian)
   sudo ufw enable
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   sudo ufw allow 22/tcp   # SSH
   sudo ufw allow 80/tcp   # HTTP
   sudo ufw allow 443/tcp  # HTTPS
   
   # Rate limiting with nginx
   # Add to nginx.conf
   http {
       limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
       
       server {
           location / {
               limit_req zone=api burst=20 nodelay;
               # ... other config
           }
       }
   }
   ```

3. **SSL/TLS configuration**

   ```nginx
   # Strong SSL configuration
   ssl_protocols TLSv1.2 TLSv1.3;
   ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
   ssl_prefer_server_ciphers off;
   ssl_session_cache shared:SSL:10m;
   ssl_session_timeout 10m;
   
   # Security headers
   add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
   add_header X-Content-Type-Options nosniff;
   add_header X-Frame-Options DENY;
   add_header X-XSS-Protection "1; mode=block";
   ```

## 📋 Backup & Recovery

### Database Backups

1. **SQLite backups**

   ```bash
   #!/bin/bash
   # backup-sqlite.sh
   
   BACKUP_DIR="/opt/ai-job-scraper/backups"
   DB_FILE="/opt/ai-job-scraper/jobs.db"
   TIMESTAMP=$(date +%Y%m%d_%H%M%S)
   
   mkdir -p $BACKUP_DIR
   
   # Create backup with timestamp
   cp $DB_FILE $BACKUP_DIR/jobs_backup_$TIMESTAMP.db
   
   # Compress old backups
   find $BACKUP_DIR -name "*.db" -mtime +1 -exec gzip {} \;
   
   # Clean up old backups (keep 30 days)
   find $BACKUP_DIR -name "*.gz" -mtime +30 -delete
   
   echo "Backup completed: jobs_backup_$TIMESTAMP.db"
   ```

2. **PostgreSQL backups**

   ```bash
   #!/bin/bash
   # backup-postgres.sh
   
   DB_HOST="localhost"
   DB_NAME="ai_jobs"
   DB_USER="postgres"
   BACKUP_DIR="/opt/backups"
   TIMESTAMP=$(date +%Y%m%d_%H%M%S)
   
   mkdir -p $BACKUP_DIR
   
   # Create backup
   pg_dump -h $DB_HOST -U $DB_USER -d $DB_NAME > $BACKUP_DIR/ai_jobs_$TIMESTAMP.sql
   
   # Compress and clean up
   gzip $BACKUP_DIR/ai_jobs_$TIMESTAMP.sql
   find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
   ```

3. **Automated backup cron jobs**

   ```bash
   # Add to crontab (crontab -e)
   # Daily database backup at 2 AM
   0 2 * * * /opt/ai-job-scraper/scripts/backup-db.sh
   
   # Weekly full backup at 3 AM Sunday
   0 3 * * 0 /opt/ai-job-scraper/scripts/full-backup.sh
   ```

### Disaster Recovery

1. **Recovery procedures**

   ```bash
   # SQLite recovery
   cp /opt/ai-job-scraper/backups/jobs_backup_20240129_020000.db /opt/ai-job-scraper/jobs.db
   sudo systemctl restart ai-job-scraper
   
   # PostgreSQL recovery
   dropdb ai_jobs
   createdb ai_jobs
   gunzip -c ai_jobs_20240129_020000.sql.gz | psql -d ai_jobs
   ```

2. **Recovery testing**

   ```bash
   #!/bin/bash
   # test-recovery.sh
   
   # Create test environment
   docker run -d --name test-postgres -e POSTGRES_DB=test_ai_jobs -e POSTGRES_PASSWORD=test postgres:15
   
   # Restore backup
   gunzip -c latest_backup.sql.gz | docker exec -i test-postgres psql -U postgres -d test_ai_jobs
   
   # Verify data integrity
   docker exec test-postgres psql -U postgres -d test_ai_jobs -c "SELECT COUNT(*) FROM jobs;"
   
   # Cleanup
   docker stop test-postgres
   docker rm test-postgres
   ```

## 📈 Scaling Considerations

### Horizontal Scaling

1. **Load balancing configuration**

   ```nginx
   upstream ai_job_scraper {
       least_conn;
       server app1:8501 max_fails=3 fail_timeout=30s;
       server app2:8501 max_fails=3 fail_timeout=30s;
       server app3:8501 max_fails=3 fail_timeout=30s;
   }
   
   server {
       location / {
           proxy_pass http://ai_job_scraper;
           # ... other proxy settings
       }
   }
   ```

2. **Session affinity (if needed)**

   ```nginx
   upstream ai_job_scraper {
       ip_hash;  # Route same IP to same server
       server app1:8501;
       server app2:8501;
   }
   ```

### Database Scaling

1. **PostgreSQL read replicas**

   ```python
   # Database connection with read/write split
   from sqlalchemy import create_engine
   
   # Master for writes
   write_engine = create_engine("postgresql://user:pass@master:5432/ai_jobs")
   
   # Replica for reads
   read_engine = create_engine("postgresql://user:pass@replica:5432/ai_jobs")
   
   def get_db_session(read_only=False):
       engine = read_engine if read_only else write_engine
       return sessionmaker(bind=engine)()
   ```

2. **Connection pooling**

   ```python
   from sqlalchemy.pool import QueuePool
   
   engine = create_engine(
       "postgresql://user:pass@host:5432/ai_jobs",
       poolclass=QueuePool,
       pool_size=10,
       max_overflow=20,
       pool_pre_ping=True,
       pool_recycle=3600
   )
   ```

## 🔧 Maintenance & Updates

### Update Procedures

1. **Rolling updates**

   ```bash
   #!/bin/bash
   # rolling-update.sh
   
   # Pull latest code
   git pull origin main
   
   # Update dependencies
   uv sync
   
   # Run database migrations (if any)
   # uv run alembic upgrade head
   
   # Restart service with health checks
   sudo systemctl restart ai-job-scraper
   
   # Wait for health check
   sleep 30
   curl -f http://localhost:8501/ || exit 1
   
   echo "Update completed successfully"
   ```

2. **Blue-green deployment**

   ```bash
   #!/bin/bash
   # blue-green-deploy.sh
   
   CURRENT=$(docker-compose -f docker-compose.prod.yml ps -q app)
   
   # Deploy to green environment
   docker-compose -f docker-compose.green.yml up -d
   
   # Health check green environment
   sleep 30
   curl -f http://green.internal:8501/ || exit 1
   
   # Switch traffic to green
   # Update load balancer configuration
   
   # Stop blue environment
   docker-compose -f docker-compose.prod.yml down
   
   echo "Blue-green deployment completed"
   ```

### Maintenance Tasks

1. **Regular maintenance script**

   ```bash
   #!/bin/bash
   # maintenance.sh
   
   # Database maintenance
   sqlite3 jobs.db "VACUUM;"
   sqlite3 jobs.db "ANALYZE;"
   
   # Clean old cache files
   find cache/ -name "*.json" -mtime +7 -delete
   
   # Clean old logs
   find logs/ -name "*.log*" -mtime +30 -delete
   
   # Update job data
   uv run python scraper.py
   
   echo "Maintenance completed"
   ```

2. **Monitoring and alerting**

   ```bash
   #!/bin/bash
   # health-monitor.sh
   
   HEALTH_URL="http://localhost:8501/"
   
   if ! curl -f $HEALTH_URL > /dev/null 2>&1; then
       echo "Health check failed, restarting service"
       sudo systemctl restart ai-job-scraper
       
       # Send alert (email, Slack, etc.)
       curl -X POST -H 'Content-type: application/json' \
            --data '{"text":"AI Job Scraper health check failed, service restarted"}' \
            $SLACK_WEBHOOK_URL
   fi
   ```

This deployment guide provides comprehensive production deployment strategies. Choose the approach that best fits your infrastructure, team size, and scaling requirements. For development and testing, start with the local production setup, then move to containerized deployments as your needs grow.
