# Foundation Setup Implementation Specification

## Branch Name

`feat/foundation-library-first-setup`

## Overview

Transform the AI Job Scraper project from its current legacy architecture to the library-first foundation defined in ADR-035. This specification restructures dependencies, removes over-engineered components, and establishes the base for the optimized 260-line architecture.

## Context and Background

### Architectural Decision References

- **ADR-031:** Library-First Architecture - 89% code reduction through native library features
- **ADR-035:** Final Production Architecture - Single model constraint, 8000 token threshold
- **ADR-033:** Minimal Implementation Guide - Eliminate custom implementations where libraries suffice

### Current State Analysis

The project currently uses:

- **Streamlit:** Legacy UI framework (to be replaced by Reflex)
- **ScrapeGraphAI:** Complex library with over-engineering (to be replaced by Crawl4AI)
- **LangGraph:** Over-engineered workflow system (to be replaced by simple task queue)
- **Custom implementations:** Memory management, error handling, task orchestration

### Target State Goals

- **Library-first dependencies:** Use native library capabilities over custom code
- **Simplified stack:** vLLM + Reflex + Crawl4AI + RQ + Tenacity
- **Modern patterns:** Latest stable library features only
- **Production readiness:** Docker-compatible, maintainable configuration

## Implementation Requirements

### 1. Dependency Restructuring

**Remove over-engineered dependencies:**

```bash
# Remove complex/obsolete packages
uv remove scrapegraphai langgraph langgraph-checkpoint-sqlite groq langchain-groq openai
```

**Add library-first dependencies:**

```bash
# Core inference and UI
uv add "vllm>=0.6.5" "reflex>=0.6.0"

# Scraping with AI capabilities
uv add "crawl4ai>=0.4.0" "python-jobspy>=1.1.82"

# Background processing and error handling
uv add "redis>=5.0.0" "rq>=1.16.0" "tenacity>=9.0.0"

# Utilities
uv add "tiktoken>=0.8.0" "pydantic>=2.10.0"
```

### 2. Project Structure Reorganization

**Create new library-first structure:**

```text
src/
├── core/
│   ├── __init__.py
│   ├── config.py          # Unified configuration
│   └── models.py          # Pydantic models only
├── ai/
│   ├── __init__.py
│   ├── inference.py       # vLLM management
│   └── threshold.py       # 8000 token logic
├── scraping/
│   ├── __init__.py
│   ├── crawler.py         # Crawl4AI primary
│   └── fallback.py        # JobSpy secondary
├── ui/
│   ├── __init__.py
│   └── app.py             # Reflex application
└── tasks/
    ├── __init__.py
    └── workers.py         # RQ background tasks
```

### 3. Configuration Consolidation

**Replace complex configuration with single file:**

```python
# src/core/config.py
from pydantic import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    """Unified application configuration - library-first approach."""
    
    # Application
    app_name: str = "AI Job Scraper"
    debug: bool = False
    
    # Local AI Configuration (ADR-035)
    model_primary: str = "Qwen/Qwen3-8B"  # Base model for most tasks
    model_thinking: str = "Qwen/Qwen3-4B-Thinking-2507"  # Instruct variant
    model_maximum: str = "Qwen/Qwen3-14B"  # High-quality for complex jobs
    
    # vLLM Configuration (ADR-031)
    vllm_swap_space: int = 4  # Automatic model management
    vllm_gpu_memory: float = 0.85  # Optimal for RTX 4090
    vllm_quantization: str = "awq-4bit"  # Memory optimization
    
    # Hybrid Processing (ADR-034)
    token_threshold: int = 8000  # 98% local processing
    cloud_api_key: str | None = None  # Fallback only
    
    # Database
    database_url: str = "sqlite:///jobs.db"
    redis_url: str = "redis://localhost:6379"
    
    # Scraping Configuration (ADR-032)
    scraping_strategy: Literal["crawl4ai", "jobspy", "auto"] = "auto"
    scraping_timeout: int = 30
    scraping_rate_limit: float = 1.0  # Requests per second
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

## Files to Create/Modify

### Files to Create

1. **`src/core/config.py`** - Unified configuration management
2. **`src/core/models.py`** - Pydantic data models only
3. **`src/ai/__init__.py`** - AI module initialization
4. **`src/scraping/__init__.py`** - Scraping module initialization
5. **`src/ui/__init__.py`** - UI module initialization  
6. **`src/tasks/__init__.py`** - Background tasks module

### Files to Modify

1. **`pyproject.toml`** - Complete dependency restructuring
2. **`.gitignore`** - Add model cache directories
3. **`docker-compose.yml`** - Update for new services
4. **`README.md`** - Update for library-first approach

## Dependencies and Libraries

### Core Dependencies (pyproject.toml updates)

```toml
[project]
name = "ai-job-scraper"
version = "2.0.0"  # Major version for architecture change
description = "Library-first AI job scraper with local processing"
requires-python = ">=3.12"

dependencies = [
    # Local AI Infrastructure
    "vllm>=0.6.5,<1.0.0",              # Local model inference
    "tiktoken>=0.8.0,<1.0.0",          # Token counting
    "torch>=2.0.0",                    # PyTorch for models
    
    # Web Scraping - Library First
    "crawl4ai>=0.4.0,<1.0.0",          # Primary scraper with AI
    "python-jobspy>=1.1.82,<2.0.0",    # Job board fallback
    
    # UI Framework
    "reflex>=0.6.0,<1.0.0",            # Modern UI with WebSockets
    
    # Background Processing
    "redis>=5.0.0,<6.0.0",             # Task queue backend
    "rq>=1.16.0,<2.0.0",               # Simple job queue
    
    # Error Handling & Utilities
    "tenacity>=9.0.0,<10.0.0",         # Retry logic
    "pydantic>=2.10.0,<3.0.0",         # Data validation
    "httpx>=0.28.0,<1.0.0",            # HTTP client
    
    # Data Management
    "sqlmodel>=0.0.24,<1.0.0",         # Database models
    "alembic>=1.13.0,<2.0.0",          # Migrations
    
    # Development
    "python-dotenv>=1.0.0,<2.0.0",     # Environment management
]

# Remove all previous dependencies (groq, langchain, scrapegraphai, streamlit, etc.)
```

## Code Implementation

### 1. Unified Configuration Implementation

```python
# src/core/config.py - Complete implementation
from pydantic import BaseSettings, Field
from typing import Literal, Dict, Any
import os

class ModelConfig:
    """Model configuration following ADR-035 single model constraint."""
    
    MODELS = {
        "primary": {
            "name": "Qwen/Qwen3-8B",
            "type": "base", 
            "use_case": "Most job extractions (2K-8K tokens)",
            "quantization": "awq-4bit"
        },
        "thinking": {
            "name": "Qwen/Qwen3-4B-Thinking-2507", 
            "type": "instruct",
            "use_case": "Simple extractions (<2K tokens)",
            "quantization": None
        },
        "maximum": {
            "name": "Qwen/Qwen3-14B",
            "type": "base",
            "use_case": "Complex extractions (5K-8K tokens)", 
            "quantization": "awq-4bit"
        }
    }

class Settings(BaseSettings):
    """Library-first application configuration."""
    
    # Application Settings
    app_name: str = "AI Job Scraper 2.0"
    debug: bool = Field(default=False, description="Debug mode")
    environment: Literal["development", "production", "testing"] = "development"
    
    # Model Configuration (ADR-035)
    model_cache_dir: str = Field(default="./models", description="Local model cache")
    token_threshold: int = Field(default=8000, description="Local vs cloud threshold")
    
    # vLLM Configuration (ADR-031)
    vllm_swap_space: int = Field(default=4, description="Automatic model swapping")
    vllm_gpu_memory: float = Field(default=0.85, description="GPU memory utilization")
    vllm_trust_remote_code: bool = Field(default=True, description="Trust model code")
    
    # Database
    database_url: str = Field(default="sqlite:///data/jobs.db")
    redis_url: str = Field(default="redis://localhost:6379")
    
    # Scraping (ADR-032) 
    scraping_timeout: int = Field(default=30, description="Per-page timeout")
    scraping_rate_limit: float = Field(default=1.0, description="Requests/second")
    scraping_concurrent: int = Field(default=5, description="Max concurrent")
    
    # API Keys (Optional)
    openai_api_key: str | None = Field(default=None, description="Cloud fallback")
    
    # Deployment
    port: int = Field(default=8000, description="Application port")
    host: str = Field(default="0.0.0.0", description="Bind host")
    
    @property
    def vllm_config(self) -> Dict[str, Any]:
        """vLLM initialization parameters."""
        return {
            "swap_space": self.vllm_swap_space,
            "gpu_memory_utilization": self.vllm_gpu_memory,
            "trust_remote_code": self.vllm_trust_remote_code,
            "quantization": "awq",  # Standard quantization
        }
    
    @property
    def model_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get model configurations with vLLM parameters."""
        return {
            name: {**config, **self.vllm_config}
            for name, config in ModelConfig.MODELS.items()
        }
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
# Global settings instance
settings = Settings()
```

### 2. Data Models (Pydantic Only)

```python
# src/core/models.py - Library-first data models
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    """Job processing status."""
    PENDING = "pending"
    PROCESSING = "processing" 
    COMPLETED = "completed"
    FAILED = "failed"

class ScrapingStrategy(str, Enum):
    """Scraping strategy selection (ADR-032)."""
    CRAWL4AI = "crawl4ai"    # Primary - 90% of cases
    JOBSPY = "jobspy"        # Fallback - job boards only
    AUTO = "auto"            # Automatic selection

class TokenThresholdDecision(BaseModel):
    """8000 token threshold decision model (ADR-034)."""
    content: str
    token_count: int
    threshold: int = 8000
    use_local: bool
    model_selected: Literal["primary", "thinking", "maximum"] | None = None
    reasoning: str
    
    @property
    def processing_location(self) -> str:
        return "local" if self.use_local else "cloud"

class JobPosting(BaseModel):
    """Standardized job posting model."""
    
    # Core Information
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: Optional[str] = Field(default=None, description="Job location")
    
    # Compensation
    salary_min: Optional[int] = Field(default=None, description="Minimum salary")
    salary_max: Optional[int] = Field(default=None, description="Maximum salary") 
    salary_currency: str = Field(default="USD", description="Currency code")
    
    # Job Details
    description: str = Field(description="Full job description")
    requirements: List[str] = Field(default_factory=list, description="Job requirements")
    benefits: List[str] = Field(default_factory=list, description="Job benefits")
    skills: List[str] = Field(default_factory=list, description="Required skills")
    
    # Metadata
    source_url: str = Field(description="Original job posting URL")
    posted_date: Optional[datetime] = Field(default=None, description="Job posting date")
    scraped_date: datetime = Field(default_factory=datetime.now, description="Scraping timestamp")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Processing status")
    
    # Processing Information
    extraction_method: ScrapingStrategy = Field(description="How job was extracted")
    token_decision: Optional[TokenThresholdDecision] = Field(default=None, description="AI processing decision")

class CompanyProfile(BaseModel):
    """Simplified company profile."""
    
    name: str = Field(description="Company name")
    website: str = Field(description="Company website")
    careers_page: Optional[str] = Field(default=None, description="Careers page URL")
    last_scraped: Optional[datetime] = Field(default=None, description="Last scraping time")
    scraping_strategy: ScrapingStrategy = Field(default=ScrapingStrategy.AUTO, description="Preferred strategy")
    success_rate: float = Field(default=0.0, description="Historical success rate")

class ScrapingTask(BaseModel):
    """Background scraping task model."""
    
    task_id: str = Field(description="Unique task identifier") 
    companies: List[str] = Field(description="Companies to scrape")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Task status")
    progress: float = Field(default=0.0, description="Completion percentage")
    jobs_found: int = Field(default=0, description="Jobs extracted")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    started_at: Optional[datetime] = Field(default=None, description="Start time")
    completed_at: Optional[datetime] = Field(default=None, description="Completion time")
```

## Testing Requirements

### 1. Configuration Testing

```python
# tests/test_foundation.py
import pytest
from src.core.config import Settings, ModelConfig

def test_settings_initialization():
    """Test settings load with defaults."""
    settings = Settings()
    
    assert settings.app_name == "AI Job Scraper 2.0"
    assert settings.token_threshold == 8000  # ADR-034
    assert settings.vllm_swap_space == 4     # ADR-031
    assert settings.vllm_gpu_memory == 0.85

def test_model_configurations():
    """Test model config consistency with ADR-035."""
    settings = Settings()
    configs = settings.model_configs
    
    # Verify corrected model names
    assert "Qwen/Qwen3-8B" in configs["primary"]["name"]
    assert "Qwen/Qwen3-4B-Thinking-2507" in configs["thinking"]["name"]
    assert "Qwen/Qwen3-14B" in configs["maximum"]["name"]
    
    # Verify vLLM parameters
    for config in configs.values():
        assert config["swap_space"] == 4
        assert config["gpu_memory_utilization"] == 0.85

def test_library_first_approach():
    """Verify no custom implementations where libraries suffice."""
    settings = Settings()
    
    # Should rely on library defaults, not custom implementations
    assert isinstance(settings.vllm_config, dict)
    assert "swap_space" in settings.vllm_config  # vLLM handles memory
```

### 2. Dependency Validation

```bash
# Validate library-first dependencies are installed
uv pip check

# Verify removed packages are gone
python -c "
try:
    import scrapegraphai
    raise AssertionError('scrapegraphai should be removed')
except ImportError:
    pass

try:
    import langgraph
    raise AssertionError('langgraph should be removed') 
except ImportError:
    pass
"

# Verify new libraries are available
python -c "
import vllm
import reflex
import crawl4ai
import tenacity
print('Library-first dependencies verified')
"
```

## Configuration

### 1. Environment Configuration

```bash
# .env.example - Template for deployment
APP_NAME="AI Job Scraper 2.0"
DEBUG=false
ENVIRONMENT=production

# Model Configuration (ADR-035)
MODEL_CACHE_DIR="./models"
TOKEN_THRESHOLD=8000

# vLLM Settings (ADR-031)
VLLM_SWAP_SPACE=4
VLLM_GPU_MEMORY=0.85
VLLM_TRUST_REMOTE_CODE=true

# Database URLs
DATABASE_URL="sqlite:///data/jobs.db"
REDIS_URL="redis://redis:6379"

# Scraping Limits (ADR-032)
SCRAPING_TIMEOUT=30
SCRAPING_RATE_LIMIT=1.0
SCRAPING_CONCURRENT=5

# Optional Cloud Fallback (2% of cases)
OPENAI_API_KEY=""

# Deployment
PORT=8000
HOST="0.0.0.0"
```

### 2. Docker Configuration Update

```yaml
# docker-compose.yml - Updated for library-first approach
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///data/jobs.db
      - REDIS_URL=redis://redis:6379
      - MODEL_CACHE_DIR=/models
    volumes:
      - ./data:/data
      - ./models:/models  # Local model storage
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"  
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

## Success Criteria

### Immediate Validation

- [ ] All legacy dependencies removed (scrapegraphai, langgraph, streamlit)
- [ ] All library-first dependencies installed and importable
- [ ] Configuration loads with correct ADR-aligned values
- [ ] Project structure follows library-first organization
- [ ] Docker services start without errors

### Architecture Compliance  

- [ ] Settings follow ADR-035 model configurations
- [ ] Token threshold set to 8000 (ADR-034)
- [ ] vLLM swap_space configured for memory management (ADR-031)
- [ ] Scraping strategy defaults align with ADR-032
- [ ] No custom implementations where libraries provide features

### Quality Metrics

- [ ] Code reduction from current state (first step toward 89% target)
- [ ] All configurations centralized in single file
- [ ] Type hints and Pydantic validation throughout
- [ ] Clear separation of concerns (AI, scraping, UI, tasks)

## Commit and PR Instructions

### Commit Messages

```bash
git checkout -b feat/foundation-library-first-setup

# Individual commits
git add pyproject.toml
git commit -m "feat: restructure dependencies for library-first approach

- Remove over-engineered packages (scrapegraphai, langgraph)
- Add library-first stack (vLLM, Reflex, Crawl4AI, RQ, Tenacity)
- Update to modern package versions
- Align with ADR-031 library-first architecture"

git add src/core/
git commit -m "feat: implement unified configuration system

- Single config file following ADR-035 specifications
- Model configurations with corrected Qwen3 references
- vLLM settings with swap_space=4 for memory management
- 8000 token threshold from ADR-034
- Environment-based configuration with sensible defaults"

git add docker-compose.yml .env.example
git commit -m "feat: update deployment configuration

- Docker services for library-first architecture
- Redis integration for RQ background tasks
- Model volume mounting for local AI
- Environment template with ADR-aligned settings"
```

### PR Description Template

```markdown
# Foundation Setup - Library-First Architecture

## Overview
Transforms AI Job Scraper foundation to implement ADR-035 final architecture with 89% code reduction target through library-first approach.

## Changes Made

### Dependencies Restructured
- ❌ Removed: `scrapegraphai`, `langgraph`, `groq`, `langchain-groq` (over-engineered)
- ✅ Added: `vllm`, `reflex`, `crawl4ai`, `rq`, `tenacity` (library-first)
- 🔄 Updated: All dependencies to latest stable versions

### Configuration Consolidated  
- Single `src/core/config.py` replaces multiple config files
- ADR-035 model configurations (corrected Qwen3 references)
- vLLM settings with `swap_space=4` for automatic memory management
- 8000 token threshold for 98% local processing

### Project Structure Modernized
- Library-first module organization
- Clear separation: AI, scraping, UI, background tasks
- Pydantic-only data models (no custom ORM patterns)

## ADR Compliance
- ✅ ADR-031: Library-first architecture foundation
- ✅ ADR-034: 8000 token threshold configuration
- ✅ ADR-035: Final architecture dependency alignment

## Testing
- Configuration loading and validation
- Dependency import verification  
- Docker service startup

## Next Steps
Ready for `02-local-ai-integration.md` implementation.
```

## Review Checklist

### Architecture Review

- [ ] All ADR references are accurate and current
- [ ] Library-first principle applied consistently
- [ ] No custom implementations where libraries provide functionality
- [ ] Configuration follows single-source-of-truth pattern

### Code Quality Review

- [ ] Type hints on all functions and classes
- [ ] Pydantic validation for all data models
- [ ] Environment variables documented and templated
- [ ] Error handling uses library patterns (tenacity)

### Integration Review

- [ ] Dependencies compatible with subsequent specifications
- [ ] Configuration supports all planned features
- [ ] Docker setup ready for multi-service deployment
- [ ] Model configurations align with hardware constraints

## Next Steps

After successful completion of this specification:

1. **Immediate:** Begin `02-local-ai-integration.md` for vLLM and Qwen3 setup
2. **Parallel:** Start researching Reflex UI patterns for `04-reflex-ui-migration.md`
3. **Planning:** Prepare development environment for local AI model downloads

This foundation specification establishes the library-first architecture base required for all subsequent implementations, ensuring the 1-week deployment timeline and 89% code reduction targets are achievable.
