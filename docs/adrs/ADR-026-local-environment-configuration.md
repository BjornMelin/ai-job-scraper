# ADR-026: Local Environment Configuration

## Status

Accepted

## Context

Need straightforward environment setup for local development that supports the Reflex framework, SQLModel with SQLite, and basic AI processing capabilities. Configuration should be simple, clear, and maintainable without production infrastructure complexity.

## Decision

Use simple environment variables and configuration files for:

- Database connection (local SQLite)
- AI service configuration (local vLLM or API fallback)
- Basic logging configuration
- Development vs. production flags
- Reflex-specific settings

### Configuration Approach

- **Environment Variables**: For runtime configuration and secrets
- **Configuration Files**: For structured application settings
- **Default Values**: Sensible defaults for development
- **Type Safety**: Using Pydantic for configuration validation

## Implementation

### Environment Configuration

```python
# src/config.py
from pydantic import BaseSettings, Field
from pathlib import Path
from typing import Optional, Literal
import os

class DatabaseConfig(BaseSettings):
    """Database configuration for local development."""
    
    url: str = Field(
        default="sqlite:///./data/jobs.db",
        description="Database connection URL"
    )
    echo: bool = Field(
        default=False,
        description="Enable SQL query logging"
    )
    pool_size: int = Field(
        default=5,
        description="Connection pool size (SQLite uses single connection)"
    )

class AIConfig(BaseSettings):
    """AI processing configuration."""
    
    provider: Literal["local", "openai"] = Field(
        default="local",
        description="AI service provider"
    )
    local_base_url: str = Field(
        default="http://localhost:8080/v1",
        description="Local AI service base URL"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for fallback"
    )
    model_name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="Model name for AI processing"
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens per request"
    )
    temperature: float = Field(
        default=0.1,
        description="Sampling temperature"
    )

class ScrapingConfig(BaseSettings):
    """Web scraping configuration."""
    
    max_concurrent_jobs: int = Field(
        default=5,
        description="Maximum concurrent scraping jobs"
    )
    request_delay: float = Field(
        default=1.0,
        description="Delay between requests in seconds"
    )
    timeout: int = Field(
        default=30,
        description="Request timeout in seconds"
    )
    user_agent: str = Field(
        default="Mozilla/5.0 (compatible; AI-Job-Scraper/1.0)",
        description="User agent for web requests"
    )
    retry_attempts: int = Field(
        default=3,
        description="Number of retry attempts for failed requests"
    )

class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    file_path: Optional[str] = Field(
        default="./logs/app.log",
        description="Log file path"
    )
    max_size: int = Field(
        default=10485760,  # 10MB
        description="Maximum log file size in bytes"
    )
    backup_count: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )

class ReflexConfig(BaseSettings):
    """Reflex framework configuration."""
    
    frontend_port: int = Field(
        default=3000,
        description="Frontend development server port"
    )
    backend_port: int = Field(
        default=8000,
        description="Backend API server port"
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    hot_reload: bool = Field(
        default=True,
        description="Enable hot reload for development"
    )

class AppConfig(BaseSettings):
    """Main application configuration."""
    
    environment: Literal["development", "production"] = Field(
        default="development",
        description="Application environment"
    )
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    data_dir: str = Field(
        default="./data",
        description="Data directory path"
    )
    logs_dir: str = Field(
        default="./logs",
        description="Logs directory path"
    )
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    reflex: ReflexConfig = Field(default_factory=ReflexConfig)
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"

# Global configuration instance
config = AppConfig()

def setup_directories():
    """Create necessary directories if they don't exist."""
    Path(config.data_dir).mkdir(exist_ok=True)
    Path(config.logs_dir).mkdir(exist_ok=True)
    
    # Create models directory if using local AI
    if config.ai.provider == "local":
        Path("./models").mkdir(exist_ok=True)

def setup_logging():
    """Configure application logging."""
    import logging
    import logging.handlers
    
    # Create logs directory
    log_path = Path(config.logging.file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.handlers.RotatingFileHandler(
                config.logging.file_path,
                maxBytes=config.logging.max_size,
                backupCount=config.logging.backup_count
            )
        ]
    )
    
    # Configure specific loggers
    logging.getLogger("reflex").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if config.database.echo else logging.WARNING
    )

def initialize_app():
    """Initialize application configuration and setup."""
    setup_directories()
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info(f"Application initialized in {config.environment} mode")
    logger.info(f"Database: {config.database.url}")
    logger.info(f"AI Provider: {config.ai.provider}")
    logger.info(f"Reflex ports: frontend={config.reflex.frontend_port}, backend={config.reflex.backend_port}")
```

### Environment File Examples

```bash
# .env - Main environment configuration
# Copy this to .env and customize for your local development

# Application Settings
ENVIRONMENT=development
DEBUG=true

# Database Configuration
DATABASE__URL=sqlite:///./data/jobs.db
DATABASE__ECHO=false

# AI Configuration
AI__PROVIDER=local
AI__LOCAL_BASE_URL=http://localhost:8080/v1
AI__OPENAI_API_KEY=your_openai_api_key_here
AI__MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
AI__MAX_TOKENS=4096
AI__TEMPERATURE=0.1

# Scraping Configuration
SCRAPING__MAX_CONCURRENT_JOBS=5
SCRAPING__REQUEST_DELAY=1.0
SCRAPING__TIMEOUT=30
SCRAPING__RETRY_ATTEMPTS=3

# Logging Configuration
LOGGING__LEVEL=DEBUG
LOGGING__FILE_PATH=./logs/app.log

# Reflex Configuration
REFLEX__FRONTEND_PORT=3000
REFLEX__BACKEND_PORT=8000
REFLEX__DEBUG=true
REFLEX__HOT_RELOAD=true

# Directory Paths
DATA_DIR=./data
LOGS_DIR=./logs
```

```bash
# .env.production - Production environment example
# Minimal configuration for production deployment

ENVIRONMENT=production
DEBUG=false

DATABASE__URL=sqlite:///./data/jobs.db
DATABASE__ECHO=false

AI__PROVIDER=openai
AI__OPENAI_API_KEY=${OPENAI_API_KEY}
AI__MODEL_NAME=gpt-3.5-turbo
AI__TEMPERATURE=0.0

SCRAPING__MAX_CONCURRENT_JOBS=3
SCRAPING__REQUEST_DELAY=2.0

LOGGING__LEVEL=INFO
LOGGING__FILE_PATH=./logs/app.log

REFLEX__DEBUG=false
REFLEX__HOT_RELOAD=false
```

### Configuration Validation

```python
# src/config_validator.py
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

def validate_configuration(config: AppConfig) -> Tuple[bool, List[str]]:
    """Validate application configuration and return issues."""
    issues = []
    
    # Database validation
    if config.database.url.startswith("sqlite:///"):
        db_path = Path(config.database.url.replace("sqlite:///", ""))
        if not db_path.parent.exists():
            issues.append(f"Database directory does not exist: {db_path.parent}")
    
    # AI configuration validation
    if config.ai.provider == "openai" and not config.ai.openai_api_key:
        issues.append("OpenAI API key is required when AI provider is 'openai'")
    
    if config.ai.provider == "local":
        # Check if local AI service is accessible (optional validation)
        logger.info("Local AI provider configured - ensure service is running")
    
    # Directory validation
    for dir_path in [config.data_dir, config.logs_dir]:
        path = Path(dir_path)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create directory {dir_path}: {e}")
    
    # Port validation
    if config.reflex.frontend_port == config.reflex.backend_port:
        issues.append("Frontend and backend ports cannot be the same")
    
    # Scraping configuration validation
    if config.scraping.max_concurrent_jobs < 1:
        issues.append("Max concurrent jobs must be at least 1")
    
    if config.scraping.request_delay < 0:
        issues.append("Request delay cannot be negative")
    
    return len(issues) == 0, issues

def print_configuration_summary(config: AppConfig):
    """Print a summary of current configuration."""
    print("\n=== AI Job Scraper Configuration ===")
    print(f"Environment: {config.environment}")
    print(f"Debug Mode: {config.debug}")
    print(f"Database: {config.database.url}")
    print(f"AI Provider: {config.ai.provider}")
    print(f"Reflex Frontend: http://localhost:{config.reflex.frontend_port}")
    print(f"Reflex Backend: http://localhost:{config.reflex.backend_port}")
    print(f"Data Directory: {config.data_dir}")
    print(f"Logs Directory: {config.logs_dir}")
    print(f"Max Concurrent Jobs: {config.scraping.max_concurrent_jobs}")
    print("=====================================\n")
```

### Configuration Integration with Reflex

```python
# src/app.py
import reflex as rx
from src.config import config, initialize_app, validate_configuration, print_configuration_summary

# Initialize application
initialize_app()

# Validate configuration
is_valid, issues = validate_configuration(config)
if not is_valid:
    for issue in issues:
        print(f"Configuration Error: {issue}")
    exit(1)

# Print configuration summary
if config.debug:
    print_configuration_summary(config)

# Reflex app configuration
app = rx.App(
    state=rx.State,
    frontend_port=config.reflex.frontend_port,
    backend_port=config.reflex.backend_port,
)
```

## Consequences

### Positive

- Clear separation of configuration concerns
- Type safety with Pydantic validation
- Easy environment switching (development/production)
- Centralized configuration management
- Sensible defaults for development
- Environment variable override capability

### Negative

- Additional configuration complexity
- Need to manage multiple environment files
- Potential configuration drift between environments
- Learning curve for nested environment variables

### Risk Mitigation

- Provide clear documentation and examples
- Include configuration validation and helpful error messages
- Use sensible defaults to minimize required configuration
- Template files for different environments

## Validation

### Success Criteria

- [ ] Application starts with default configuration
- [ ] Environment variables properly override defaults
- [ ] Configuration validation catches common errors
- [ ] All configuration options documented with examples
- [ ] Easy switching between development and production configs

### Testing Approach

- Unit tests for configuration validation logic
- Integration tests with different environment configurations
- Manual testing of environment variable overrides
- Documentation testing with fresh setup

## Related ADRs

- **Supports ADR-022**: Local Development Docker Containerization (environment setup)
- **Replaces Archived ADR-026**: Host System Resource Management (production-focused)
- **Updates ADR-017**: Final Production Architecture (local development focus)

---

*This ADR provides straightforward environment configuration for local development while maintaining flexibility for future production deployment.*
