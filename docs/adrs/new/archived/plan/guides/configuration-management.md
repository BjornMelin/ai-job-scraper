# Configuration Management Quick Reference Guide

## Overview

This guide provides quick reference for managing configuration across all components of the AI Job Scraper, following the unified configuration approach from the library-first architecture.

## Configuration Architecture

### Single Source of Truth

All configuration is managed through `src/core/config.py` using Pydantic Settings:

```python
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    """Unified application configuration."""
    
    # Core settings loaded from environment
    app_name: str = "AI Job Scraper"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

### Environment-Based Configuration

Different configurations for different environments:

```text
.env.development    # Local development
.env.production     # Production deployment  
.env.testing        # Test environment
```

## Core Configuration Sections

### 1. Local AI Configuration (ADR-035)

```python
class AIConfig(BaseModel):
    """Local AI model configuration."""
    
    # Model Selection (Corrected References)
    model_primary: str = "Qwen/Qwen3-8B"           # Base model for most tasks
    model_thinking: str = "Qwen/Qwen3-4B-Thinking-2507"  # Instruct variant
    model_maximum: str = "Qwen/Qwen3-14B"          # High-quality for complex jobs
    
    # vLLM Configuration (ADR-031)
    vllm_swap_space: int = 4                       # Automatic memory management
    vllm_gpu_memory: float = 0.85                  # GPU memory utilization
    vllm_trust_remote_code: bool = True            # Trust model code
    vllm_quantization: str = "awq-4bit"            # Memory optimization
    
    # Token Threshold (ADR-034)
    token_threshold: int = 8000                    # 98% local processing threshold
    
    # Model Cache
    model_cache_dir: str = "./models"              # Local model storage
```

**Environment Variables:**

```bash
# .env file
MODEL_PRIMARY="Qwen/Qwen3-8B"
MODEL_THINKING="Qwen/Qwen3-4B-Thinking-2507"
MODEL_MAXIMUM="Qwen/Qwen3-14B"
VLLM_SWAP_SPACE=4
VLLM_GPU_MEMORY=0.85
TOKEN_THRESHOLD=8000
MODEL_CACHE_DIR="./models"
```

### 2. Scraping Configuration (ADR-032)

```python
class ScrapingConfig(BaseModel):
    """Scraping strategy configuration."""
    
    # Strategy Settings
    primary_strategy: str = "crawl4ai"             # 90% of cases
    fallback_strategy: str = "jobspy"              # 10% for job boards
    auto_fallback: bool = True                     # Enable automatic fallback
    
    # Crawl4AI Settings
    crawl4ai_timeout: int = 30                     # Page timeout in seconds
    crawl4ai_anti_bot: bool = True                 # Enable anti-bot detection
    crawl4ai_cache: bool = True                    # Enable smart caching
    crawl4ai_screenshot: bool = False              # Screenshots for debugging
    
    # JobSpy Settings
    jobspy_sites: list[str] = ["linkedin", "indeed", "zip_recruiter"]
    jobspy_max_results: int = 100                  # Max results per search
    jobspy_distance: int = 50                      # Location search radius
    
    # Performance Settings
    concurrent_limit: int = 5                      # Max concurrent operations
    rate_limit: float = 1.0                        # Requests per second
    retry_attempts: int = 3                        # Max retry attempts
```

**Environment Variables:**

```bash
# Scraping configuration
SCRAPING_PRIMARY="crawl4ai"
SCRAPING_FALLBACK="jobspy"
CRAWL4AI_TIMEOUT=30
CRAWL4AI_ANTI_BOT=true
JOBSPY_SITES="linkedin,indeed,zip_recruiter"
SCRAPING_CONCURRENT_LIMIT=5
SCRAPING_RATE_LIMIT=1.0
```

### 3. UI Configuration (Reflex)

```python
class UIConfig(BaseModel):
    """UI framework configuration."""
    
    # Reflex Settings
    frontend_port: int = 3000                      # Frontend port
    backend_port: int = 8000                       # Backend API port
    
    # Theme Settings
    theme: str = "dark"                            # UI theme
    accent_color: str = "blue"                     # Accent color
    
    # Performance Settings
    websocket_timeout: int = 30                    # WebSocket timeout
    realtime_updates: bool = True                  # Enable real-time updates
    jobs_per_page: int = 20                        # Pagination size
    
    # Mobile Settings
    mobile_responsive: bool = True                 # Enable responsive design
    mobile_breakpoint: int = 768                   # Mobile breakpoint (px)
```

**Environment Variables:**

```bash
# UI configuration
REFLEX_FRONTEND_PORT=3000
REFLEX_BACKEND_PORT=8000
UI_THEME="dark"
UI_ACCENT_COLOR="blue"
WEBSOCKET_TIMEOUT=30
JOBS_PER_PAGE=20
```

### 4. Background Processing Configuration

```python
class BackgroundConfig(BaseModel):
    """Background task processing configuration."""
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379"      # Redis connection
    redis_db: int = 0                              # Redis database number
    
    # RQ Settings
    queue_name: str = "scraping"                   # Primary queue name
    high_priority_queue: str = "high"             # High priority queue
    worker_timeout: int = 1800                     # Worker timeout (30 min)
    
    # Tenacity Settings (Error Handling)
    retry_max_attempts: int = 3                    # Maximum retry attempts
    retry_min_wait: int = 1                        # Minimum retry wait (seconds)
    retry_max_wait: int = 10                       # Maximum retry wait (seconds)
    retry_multiplier: float = 2.0                  # Exponential backoff multiplier
    
    # Progress Tracking
    progress_cleanup_hours: int = 24               # Hours to keep progress data
    
    # Scheduling
    daily_refresh_hour: int = 2                    # Daily refresh time (24h format)
    weekly_refresh_day: int = 6                    # Weekly refresh day (0=Monday)
```

**Environment Variables:**

```bash
# Background processing
REDIS_URL="redis://localhost:6379"
QUEUE_NAME="scraping"
WORKER_TIMEOUT=1800
RETRY_MAX_ATTEMPTS=3
RETRY_MIN_WAIT=1
RETRY_MAX_WAIT=10
PROGRESS_CLEANUP_HOURS=24
```

### 5. Database Configuration

```python
class DatabaseConfig(BaseModel):
    """Database configuration."""
    
    # SQLite Settings
    database_url: str = "sqlite:///data/jobs.db"   # Database URL
    echo_sql: bool = False                         # Log SQL queries
    
    # Connection Settings
    pool_size: int = 10                            # Connection pool size
    max_overflow: int = 20                         # Max overflow connections
    pool_timeout: int = 30                         # Connection timeout
    
    # Performance Settings
    pragma_journal_mode: str = "WAL"              # Journal mode for performance
    pragma_synchronous: str = "NORMAL"            # Synchronization mode
    pragma_cache_size: int = -2000                # Cache size (-2MB)
    
    # Backup Settings
    backup_enabled: bool = True                    # Enable automatic backups
    backup_interval_hours: int = 6                 # Backup frequency
    backup_retention_days: int = 30                # Backup retention
```

**Environment Variables:**

```bash
# Database configuration
DATABASE_URL="sqlite:///data/jobs.db"
ECHO_SQL=false
BACKUP_ENABLED=true
BACKUP_INTERVAL_HOURS=6
BACKUP_RETENTION_DAYS=30
```

## Production Configuration

### Environment-Specific Settings

**Development (.env.development):**

```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Relaxed performance settings
TOKEN_THRESHOLD=4000
SCRAPING_CONCURRENT_LIMIT=2
SCRAPING_RATE_LIMIT=0.5

# Local services
REDIS_URL="redis://localhost:6379"
DATABASE_URL="sqlite:///dev_jobs.db"
```

**Production (.env.production):**

```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Optimized performance settings  
TOKEN_THRESHOLD=8000
SCRAPING_CONCURRENT_LIMIT=10
SCRAPING_RATE_LIMIT=2.0

# Production services
REDIS_URL="redis://redis:6379"
DATABASE_URL="sqlite:///data/jobs.db"

# Security settings
SECRET_KEY="your-secure-secret-key"
API_KEY="your-secure-api-key"
```

**Testing (.env.testing):**

```bash
ENVIRONMENT=testing
DEBUG=true
LOG_LEVEL=WARNING

# Fast test settings
TOKEN_THRESHOLD=1000
MOCK_AI_MODELS=true
MOCK_SCRAPING=true

# Test services
REDIS_URL="redis://localhost:6380"
DATABASE_URL="sqlite:///:memory:"
```

## Configuration Loading Patterns

### Basic Configuration Loading

```python
# src/core/config.py
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Main application settings."""
    
    # Load from environment with validation
    environment: str = "development"
    debug: bool = False
    
    # Nested configuration objects
    ai: AIConfig = AIConfig()
    scraping: ScrapingConfig = ScrapingConfig()
    ui: UIConfig = UIConfig()
    background: BackgroundConfig = BackgroundConfig()
    database: DatabaseConfig = DatabaseConfig()
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"  # Support nested config via env vars

# Global settings instance
settings = Settings()
```

### Dynamic Configuration Loading

```python
# Load configuration based on environment
import os

def get_settings():
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        return Settings(_env_file=".env.production")
    elif env == "testing":
        return Settings(_env_file=".env.testing")
    else:
        return Settings(_env_file=".env.development")

settings = get_settings()
```

### Configuration Validation

```python
# Validate configuration on startup
def validate_configuration():
    """Validate configuration is complete and correct."""
    
    errors = []
    
    # Check required model files exist
    for model_config in settings.ai.model_configs.values():
        model_path = Path(settings.ai.model_cache_dir) / model_config["name"]
        if not model_path.exists():
            errors.append(f"Model not found: {model_config['name']}")
    
    # Check Redis connectivity
    try:
        import redis
        redis_client = redis.from_url(settings.background.redis_url)
        redis_client.ping()
    except Exception as e:
        errors.append(f"Redis connection failed: {e}")
    
    # Check database accessibility
    db_path = settings.database.database_url.replace("sqlite:///", "")
    if not Path(db_path).parent.exists():
        errors.append(f"Database directory does not exist: {Path(db_path).parent}")
    
    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))
```

## Component Integration

### Using Configuration in Components

```python
# AI component using configuration
from src.core.config import settings

class ModelManager:
    def __init__(self):
        self.config = settings.ai
        
    def get_model(self, model_type: str):
        model_config = self.config.model_configs[model_type]
        return LLM(
            model=model_config["name"],
            swap_space=self.config.vllm_swap_space,
            gpu_memory_utilization=self.config.vllm_gpu_memory
        )

# Scraping component using configuration
class ScrapingManager:
    def __init__(self):
        self.config = settings.scraping
        
    async def scrape(self, url: str):
        timeout = self.config.crawl4ai_timeout
        anti_bot = self.config.crawl4ai_anti_bot
        
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(
                url=url,
                page_timeout=timeout * 1000,
                anti_bot=anti_bot
            )
```

### Configuration Hot Reloading (Development)

```python
# Development hot reload for configuration changes
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigReloadHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path.endswith('.env'):
            print("🔄 Configuration file changed, reloading...")
            global settings
            settings = Settings(_env_file=event.src_path)

# Enable in development mode
if settings.environment == "development":
    observer = Observer()
    observer.schedule(ConfigReloadHandler(), path='.', recursive=False)
    observer.start()
```

## Security Best Practices

### Environment Variable Security

```bash
# Use secure defaults
SECRET_KEY="${SECRET_KEY:-$(openssl rand -hex 32)}"
API_KEY="${API_KEY:-$(openssl rand -hex 16)}"

# Never commit sensitive values
echo ".env*" >> .gitignore

# Use environment-specific files
.env.development     # Not committed
.env.production      # Not committed  
.env.example         # Committed template
```

### Configuration Encryption

```python
# Encrypt sensitive configuration values
from cryptography.fernet import Fernet

class SecureSettings(BaseSettings):
    """Settings with encryption support."""
    
    _encryption_key: Optional[str] = None
    
    def decrypt_value(self, encrypted_value: str) -> str:
        if not self._encryption_key:
            raise ValueError("Encryption key not set")
            
        fernet = Fernet(self._encryption_key.encode())
        return fernet.decrypt(encrypted_value.encode()).decode()
    
    @property
    def api_key(self) -> str:
        encrypted = os.getenv("ENCRYPTED_API_KEY")
        if encrypted:
            return self.decrypt_value(encrypted)
        return os.getenv("API_KEY", "")
```

## Troubleshooting Configuration Issues

### Common Configuration Problems

**Problem:** Environment variables not loading

```bash
# Check environment file exists
ls -la .env*

# Check variable is set
echo $VARIABLE_NAME

# Test loading manually
python -c "from src.core.config import settings; print(settings.debug)"
```

**Problem:** Model paths not found

```python
# Validate model configuration
from src.core.config import settings
print(f"Model cache dir: {settings.ai.model_cache_dir}")
print(f"Models configured: {list(settings.ai.model_configs.keys())}")

# Check actual files
import os
for model_name in settings.ai.model_configs.values():
    path = os.path.join(settings.ai.model_cache_dir, model_name)
    print(f"Model {model_name}: {'✅' if os.path.exists(path) else '❌'}")
```

**Problem:** Redis connection issues

```python
# Test Redis configuration
import redis
from src.core.config import settings

try:
    client = redis.from_url(settings.background.redis_url)
    client.ping()
    print("✅ Redis connection successful")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
```

### Configuration Debugging

```python
# Debug configuration loading
def debug_configuration():
    """Print current configuration for debugging."""
    
    print("=== Configuration Debug ===")
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")
    print()
    
    print("AI Configuration:")
    print(f"  Token threshold: {settings.ai.token_threshold}")
    print(f"  Model cache: {settings.ai.model_cache_dir}")
    print(f"  vLLM swap space: {settings.ai.vllm_swap_space}")
    print()
    
    print("Scraping Configuration:")
    print(f"  Primary strategy: {settings.scraping.primary_strategy}")
    print(f"  Concurrent limit: {settings.scraping.concurrent_limit}")
    print(f"  Rate limit: {settings.scraping.rate_limit}")
    print()
    
    print("Background Configuration:")
    print(f"  Redis URL: {settings.background.redis_url}")
    print(f"  Queue name: {settings.background.queue_name}")
    print(f"  Worker timeout: {settings.background.worker_timeout}")

# Call during startup
if __name__ == "__main__":
    debug_configuration()
```

This configuration management guide ensures all components use consistent, validated configuration while maintaining flexibility across different deployment environments.
