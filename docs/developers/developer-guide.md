# 🛠️ Developer Guide: AI Job Scraper

This guide provides technical details for developers who want to understand, extend, or contribute to the AI Job Scraper codebase.

## 🏗️ Architecture Overview

### System Components

```mermaid
graph TD
    A[User Interface] --> B[Application Layer]
    B --> C[Scraping Engine]
    B --> D[Database Layer]
    C --> E[Extraction Strategies]
    C --> F[Caching System]
    E --> G[LLM Strategy]
    E --> H[CSS Strategy]
    F --> I[File Cache]
    D --> J[SQLite Database]
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style F fill:#fff3e0
```

### Core Modules

| Module | Purpose | Key Components |
|--------|---------|----------------|
| `app.py` | Streamlit UI | Dashboard, filtering, job management |
| `scraper.py` | Web scraping | Async crawling, extraction, caching |
| `models.py` | Data models | SQLAlchemy ORM, Pydantic validation |
| `config.py` | Configuration | Settings management, environment variables |
| `seed.py` | Database setup | Initial company data seeding |

## 📚 Technical Stack

### Core Dependencies

```python

# Web scraping and AI
crawl4ai==0.7.2          # AI-powered web crawler
tenacity==9.1.2          # Retry logic with exponential backoff

# Web framework and UI  
streamlit==1.47.1        # Interactive dashboard framework

# Database and ORM
sqlalchemy==2.0.42       # SQL toolkit and ORM
pandas==2.3.1           # Data manipulation and analysis

# Validation and settings
pydantic==2.11.7         # Data validation using Python type hints
pydantic-settings==2.10.1 # Settings management

# HTTP and utilities
httpx==0.28.1           # Async HTTP client
python-dateutil==2.9.0.post0 # Date parsing utilities
typer==0.16.0           # CLI framework
```

### Development Dependencies

```python

# Testing
pytest>=8.0.0           # Testing framework
pytest-asyncio>=0.23.0  # Async testing support

# Code quality
ruff>=0.8.0             # Fast Python linter and formatter
```

## 🔍 Code Structure Analysis

### Application Entry Points

**Streamlit App (`app.py`)**

```python

# Main dashboard components
def display_jobs(jobs: list[JobSQL], tab_key: str) -> None
def update_status(job_id: int, tab_key: str) -> None  
def update_notes(job_id: int, tab_key: str) -> None

# UI state management via st.session_state
if "filters" not in st.session_state:
    st.session_state.filters = {...}
```

**CLI Scraper (`scraper.py`)**

```python
def main() -> None:
    """Command-line interface entry point."""
    jobs_df = asyncio.run(scrape_all())
    update_db(jobs_df)

if __name__ == "__main__":
    typer.run(main)
```

### Data Models

**SQLAlchemy Models (`models.py`)**

```python
class JobSQL(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    company = Column(String)
    title = Column(String) 
    description = Column(Text)
    link = Column(String, unique=True)  # Unique constraint for deduplication
    # ... user tracking fields
    favorite = Column(Boolean, default=False)
    status = Column(String, default="New")
    notes = Column(Text, default="")
```

**Pydantic Validation (`models.py`)**

```python
class JobPydantic(BaseModel):
    company: str
    title: str
    description: str
    link: str = Field(pattern=r"^https?://")  # URL validation
    location: str | None = "Unknown"
    posted_date: datetime | None = None
```

## 🚀 Scraping Engine Deep Dive

### Extraction Strategy Pattern

The scraping engine uses a strategy pattern with intelligent fallbacks:

```python
async def extract_jobs(url: str, company: str) -> list[dict]:
    # 1. Try cached schema first (90% speed improvement)
    cached_schema = get_cached_schema(company)
    if cached_schema:
        strategy = JsonCssExtractionStrategy(cached_schema)
        # ... attempt extraction
    
    # 2. Fallback to LLM extraction (50% cost reduction via optimization)  
    strategy = LLMExtractionStrategy(
        provider="openai/gpt-4o-mini",
        extraction_schema=SIMPLE_SCHEMA,  # Minimal schema
        instructions=SIMPLE_INSTRUCTIONS, # 50-word instructions
        apply_chunking=True,              # Token efficiency
        chunk_token_threshold=1000,
        overlap_rate=0.02,
    )
    
    # 3. Final fallback to basic CSS
    strategy = JsonCssExtractionStrategy(
        css_selector=".job-listing, .job-item",
        instruction="Extract job details"
    )
```

### Performance Optimization System

**Schema Caching (`scraper.py:91-106`)**

```python
def get_cached_schema(company: str) -> dict | None:
    """Load cached extraction schema for 90% speed improvement."""
    cache_file = CACHE_DIR / f"{company.lower()}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    return None

def save_schema_cache(company: str, schema: dict) -> None:
    """Save successful extraction patterns for reuse."""
    cache_file = CACHE_DIR / f"{company.lower()}.json"
    cache_file.write_text(json.dumps(schema, indent=2))
```

**Rate Limiting (`scraper.py:72-78`)**

```python
COMPANY_DELAYS = {
    "nvidia": 3.0,    # Slower for complex sites
    "meta": 2.0,      # Slower for Meta
    "microsoft": 2.5, # Slower for Microsoft  
    "default": 1.0,   # Default delay
}

# Applied before each extraction
delay = COMPANY_DELAYS.get(company.lower(), COMPANY_DELAYS["default"])
await asyncio.sleep(delay)
```

**Session Metrics (`scraper.py:81-89`)**

```python
session_stats = {
    "start_time": None,
    "companies_processed": 0,
    "jobs_found": 0,
    "cache_hits": 0,        # For cache hit rate calculation
    "llm_calls": 0,         # For cost monitoring
    "errors": 0,            # For reliability tracking
}
```

### Database Operations

**CRUD Operations (`scraper.py:318-383`)**

```python
def update_db(jobs_df: pd.DataFrame) -> None:
    """Full CRUD: Create, Read, Update, Delete with user data preservation."""
    session = Session()
    try:
        existing = {j.link: j for j in session.query(JobSQL).all()}
        
        # Process each scraped job
        for _, row in jobs_df.iterrows():
            job_dict = row.to_dict()
            
            # Validate with Pydantic
            JobPydantic(**job_dict)
            
            # Hash for change detection
            job_hash = hashlib.sha256(job_dict["description"].encode()).hexdigest()
            
            if job_dict["link"] in existing:
                # Update existing job, preserve user fields
                ex = existing[job_dict["link"]]
                if ex.hash != job_hash:  # Only update if content changed
                    ex.title = job_dict["title"]
                    ex.description = job_dict["description"]
                    # ... preserve favorite, status, notes
            else:
                # Create new job
                new_job = JobSQL(**job_dict, hash=job_hash)
                session.add(new_job)
        
        # Delete jobs no longer found
        current_links = set(jobs_df["link"])
        for link in existing.keys():
            if link not in current_links:
                session.delete(existing[link])
                
        session.commit()
    except Exception as e:
        session.rollback()
        raise
```

## 🎨 UI Architecture

### Streamlit State Management

**Session State Pattern (`app.py:324-339`)**

```python

# Initialize persistent UI state
if "filters" not in st.session_state:
    st.session_state.filters = {
        "company": [],
        "keyword": "",
        "date_from": None,
        "date_to": None,
    }

# State persists across reruns and user interactions
st.session_state.view_mode = st.radio("Select View", ["List", "Card"])
```

**Callback Functions (`app.py:86-134`)**

```python
def update_status(job_id: int, tab_key: str) -> None:
    """Database update triggered by UI widget changes."""
    try:
        session = Session()
        job = session.query(JobSQL).filter_by(id=job_id).first()
        job.status = st.session_state[f"status_{job_id}_{tab_key}_{page}"]
        session.commit()
    finally:
        session.close()
    st.rerun()  # Refresh UI
```

### Custom CSS System (`app.py:28-83`)

#### Responsive Design

```css
/* Mobile-first responsive design */
@media (max-width: 768px) {
    .card { 
        margin: 8px 0; 
    }
    [data-testid="column"] { 
        width: 100% !important; 
        flex: 1 1 100% !important; 
        min-width: 100% !important; 
    }
}
```

#### Tech Theme

```css
/* Dark gradient tech theme */
[data-testid="stAppViewContainer"] { 
    background: linear-gradient(to bottom right, #0a192f, #1e3a8a); 
    color: #e2e8f0; 
}

/* Interactive card effects */
.card:hover { 
    box-shadow: 0 6px 8px rgba(59, 130, 246, 0.5); 
}
```

## 🔧 Extension Points

### Adding New Extraction Strategies

#### Custom Strategy Example

```python

# In scraper.py, add to extract_jobs function
class CustomExtractionStrategy:
    """Custom extraction for specific site structures."""
    
    def __init__(self, company_config: dict):
        self.config = company_config
    
    async def extract(self, html_content: str) -> list[dict]:
        # Custom extraction logic here
        return jobs_list

# Usage in extract_jobs
if company.lower() == "custom_company":
    strategy = CustomExtractionStrategy(config)
```

### Adding New Data Fields

#### Database Migration Steps

1. **Update SQLAlchemy model** (`models.py`)

    ```python
    class JobSQL(Base):
        # ... existing fields
        salary_range = Column(String)  # New field
        remote_friendly = Column(Boolean, default=False)
    ```

2. **Update Pydantic model** for validation

    ```python
    class JobPydantic(BaseModel):
        # ... existing fields  
        salary_range: str | None = None
        remote_friendly: bool = False
    ```

3. **Update extraction schema**

    ```python
    EXTENDED_SCHEMA = {
        "jobs": [
            {
                # ... existing fields
                "salary_range": "Salary range if mentioned",
                "remote_friendly": "Whether remote work is allowed"
            }
        ]
    }
    ```

4. **Update UI** (`app.py`) to display new fields

    ```python
    df = pd.DataFrame([
        {
            # ... existing fields
            "Salary": j.salary_range,
            "Remote": j.remote_friendly,
        }
        for j in jobs
    ])
    ```

### Custom Company Configurations

#### Advanced Company Setup (`seed.py`)

```python
ADVANCED_SITES = {
    "custom_company": {
        "url": "https://company.com/careers",
        "extraction_config": {
            "job_selector": ".custom-job-class",
            "title_selector": ".custom-title",
            "rate_limit": 2.0,  # Custom delay
            "requires_js": True,  # For dynamic content
        }
    }
}

def seed_advanced_companies():
    """Seed companies with custom configurations."""
    session = Session()
    for name, config in ADVANCED_SITES.items():
        company = CompanySQL(
            name=name,
            url=config["url"], 
            config=json.dumps(config.get("extraction_config", {}))
        )
        session.add(company)
    session.commit()
```

## 🧪 Testing Framework

### Test Structure

```text
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures
├── test_models.py           # Model validation tests
├── test_scraper.py          # Scraping logic tests  
├── test_app.py              # UI component tests
└── integration/
    ├── test_end_to_end.py   # Full workflow tests
    └── test_database.py     # Database operation tests
```

### Key Testing Patterns

**Async Test Example (`test_scraper.py`)**

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_extract_jobs_with_cache():
    """Test cached extraction path."""
    with patch('scraper.get_cached_schema') as mock_cache:
        mock_cache.return_value = {"jobs": {"selector": ".job"}}
        
        jobs = await extract_jobs("https://example.com", "test_company")
        
        assert len(jobs) > 0
        assert all("company" in job for job in jobs)
```

**Database Test Example (`test_models.py`)**

```python
def test_job_validation():
    """Test Pydantic job validation."""
    valid_job = {
        "company": "Test Corp",
        "title": "AI Engineer", 
        "description": "Build AI systems",
        "link": "https://example.com/job/123"
    }
    
    job = JobPydantic(**valid_job)
    assert job.company == "Test Corp"
    
    # Test invalid URL
    with pytest.raises(ValueError):
        JobPydantic(**{**valid_job, "link": "invalid-url"})
```

### Running Tests

```bash

# Install dev dependencies
uv sync --group dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=. --cov-report=html

# Run specific test file
uv run pytest tests/test_scraper.py -v

# Run async tests only
uv run pytest -m asyncio
```

## 📈 Performance Monitoring

### Metrics Collection

**Performance Tracking (`scraper.py:131-145`)**

```python
def log_session_summary():
    """Performance metrics for optimization."""
    duration = time.time() - session_stats["start_time"]
    cache_rate = session_stats["cache_hits"] / max(session_stats["companies_processed"], 1)
    
    logger.info("📊 Session Summary:")
    logger.info(f"  Duration: {duration:.1f}s")
    logger.info(f"  Cache hit rate: {cache_rate:.1%}")  # Target: >70%
    logger.info(f"  LLM calls: {session_stats['llm_calls']}")  # Target: <3
    logger.info(f"  Jobs found: {session_stats['jobs_found']}")
```

#### Performance Benchmarks

- **Cold start** (no cache): <90s for all companies

- **Warm start** (with cache): <45s for all companies  

- **Cache hit rate**: >70% after first run

- **LLM calls**: <3 per session (cost optimization)

- **Memory usage**: <500MB peak

- **Database operations**: <2s for 1000 jobs

### Optimization Opportunities

#### Cache Optimization

```python

# Current: File-based JSON cache

# Future: Redis cluster for distributed caching

# Future: Compression for large schemas

def get_compressed_cache(company: str) -> dict | None:
    """Compressed cache with TTL support."""
    import gzip
    cache_file = CACHE_DIR / f"{company.lower()}.json.gz"
    if cache_file.exists():
        with gzip.open(cache_file, 'rt') as f:
            return json.load(f)
    return None
```

#### Parallel Processing

```python

# Current: Async processing with company-specific delays

# Future: Dynamic rate limiting based on response times

async def adaptive_extract_jobs(url: str, company: str) -> list[dict]:
    """Extract with adaptive rate limiting."""
    start_time = time.time()
    jobs = await extract_jobs(url, company)
    duration = time.time() - start_time
    
    # Adjust delay based on response time
    if duration > 30:  # Slow response
        COMPANY_DELAYS[company] = min(COMPANY_DELAYS.get(company, 1.0) * 1.5, 5.0)
    elif duration < 10:  # Fast response
        COMPANY_DELAYS[company] = max(COMPANY_DELAYS.get(company, 1.0) * 0.8, 0.5)
        
    return jobs
```

## 🔒 Security Considerations

### Input Validation

**URL Security (`models.py:95`)**

```python

# Pydantic URL validation prevents injection
link: str = Field(pattern=r"^https?://")

# Additional validation in scraper
async def validate_link(link: str) -> str | None:
    """Validate and sanitize URLs."""
    try:
        # Verify URL is accessible and safe
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.head(link, follow_redirects=True)
            return link if resp.status_code == 200 else None
    except Exception:
        return None
```

#### SQL Injection Prevention

```python

# SQLAlchemy ORM prevents SQL injection by default

# All queries use parameterized statements
query = session.query(JobSQL).filter(JobSQL.company.in_(companies))

# NOT: f"SELECT * FROM jobs WHERE company IN ({companies})"
```

### API Key Management

**Environment Variable Security (`config.py`)**

```python
class Settings(BaseSettings):
    openai_api_key: str
    db_url: str = "sqlite:///jobs.db"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True  # Ignore empty env vars
    )

# Usage
settings = Settings()  # Automatically loads from .env
```

#### Development vs Production

```bash

# Development (.env)
OPENAI_API_KEY=sk-dev-key-here
DB_URL=sqlite:///jobs_dev.db

# Production (environment variables)
export OPENAI_API_KEY=sk-prod-key-here
export DB_URL=postgresql://user:pass@prod-db:5432/jobs
```

## 🚀 Deployment Strategies

### Docker Configuration

#### Multi-stage Dockerfile

```dockerfile

# Build stage
FROM python:3.12-slim as builder
WORKDIR /app
RUN pip install uv
COPY pyproject.toml .
RUN uv sync --frozen

# Runtime stage  
FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    # Playwright dependencies for Crawl4AI
    libnss3 libnspr4 libatk1.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .

# Install Playwright browsers
RUN /app/.venv/bin/python -m playwright install --with-deps

EXPOSE 8501
CMD ["/app/.venv/bin/streamlit", "run", "app.py", "--server.port=8501"]
```

#### Production Docker Compose

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DB_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/jobs
    volumes:
      - ./cache:/app/cache  # Persistent cache
      - ./logs:/app/logs    # Application logs
    depends_on:
      - db
      
  db:
    image: postgres:15
    environment:
      - POSTGRES_DB=jobs
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      
volumes:
  postgres_data:
```

### Monitoring & Observability

#### Logging Configuration

```python

# Enhanced logging setup
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure structured logging for production."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            RotatingFileHandler(  # File with rotation
                'logs/scraper.log', 
                maxBytes=10_000_000, 
                backupCount=5
            )
        ]
    )
```

#### Health Check Endpoint

```python

# Add to app.py for production monitoring
def health_check():
    """Simple health check for load balancers."""
    try:
        # Test database connection
        session = Session()
        session.execute("SELECT 1")
        session.close()
        return {"status": "healthy", "timestamp": datetime.now()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# Usage with FastAPI for production
if os.getenv("PRODUCTION"):
    from fastapi import FastAPI
    api = FastAPI()
    
    @api.get("/health")
    async def health():
        return health_check()
```

## 🤝 Contributing Guidelines

### Code Style

#### Ruff Configuration (pyproject.toml:34-112)

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # pyflakes
    "I",    # isort
    "D",    # pydocstyle (Google style)
    "UP",   # pyupgrade
    "S",    # bandit (security)
    "B",    # flake8-bugbear
]

[tool.ruff.lint.pydocstyle]
convention = "google"
```

#### Pre-commit Workflow

```bash

# Format and lint before committing
uv run ruff format .
uv run ruff check . --fix

# Run tests
uv run pytest

# Check types (if using mypy)
uv run mypy scraper.py app.py
```

### Pull Request Process

1. **Fork and branch** from `main`
2. **Implement changes** following KISS/DRY principles
3. **Add tests** for new functionality  
4. **Update documentation** if needed
5. **Run quality checks**:

   ```bash
   uv run ruff format .
   uv run ruff check . --fix
   uv run pytest --cov=.
   ```

6. **Create PR** with clear description
7. **Reference relevant ADRs** for architectural decisions

### Development Environment

#### Recommended VS Code Extensions

- Python

- Ruff (linting/formatting)

- Streamlit (syntax highlighting)

- SQLite Viewer (database inspection)

#### Environment Setup

```bash

# Clone and setup
git clone https://github.com/BjornMelin/ai-job-scraper.git
cd ai-job-scraper

# Install dependencies
uv sync --group dev

# Setup pre-commit hooks (optional)
uv run pre-commit install

# Run in development mode
uv run streamlit run app.py --server.runOnSave=true
```

## 📖 Architectural Decision Records (ADRs)

Key ADRs that shaped the current architecture:

- **[ADR-001](adrs/001-scraping-and-extraction-framework.md)**: Crawl4AI selection for scraping

- **[ADR-003](adrs/003-persistence-and-data-management.md)**: SQLite + SQLAlchemy for data persistence  

- **[ADR-004](adrs/004-user-interface-framework.md)**: Streamlit for rapid UI development

- **[ADR-011](adrs/011-performance-optimization-and-caching.md)**: File-based caching system

## 🔮 Future Enhancements

### Planned Features

#### LLM-Based Filtering (ADR-002)

```python

# Future: Replace regex with LLM relevance scoring
async def llm_filter_relevance(job: dict) -> tuple[bool, float]:
    """LLM-based relevance scoring with confidence."""
    prompt = f"""
    Is this job relevant for AI/ML engineers? Score 0-1.
    Job: {job['title']} - {job['description'][:200]}
    """
    
    # Use cached LLM with cost optimization
    response = await cached_llm_call(prompt)
    score = float(response.strip())
    return score > 0.7, score
```

#### Real-time Notifications

```python

# Future: WebSocket-based real-time updates
class JobNotifier:
    """Real-time job notifications via WebSocket."""
    
    async def notify_new_jobs(self, jobs: list[JobSQL]):
        for job in jobs:
            if await self.matches_user_criteria(job):
                await self.send_notification(job)
```

#### Analytics Dashboard

```python

# Future: Advanced analytics with Plotly
def create_analytics_dashboard():
    """Job market analytics and trends."""
    import plotly.express as px
    
    # Company posting frequency
    company_counts = session.query(JobSQL.company, func.count()).group_by(JobSQL.company)
    fig = px.bar(company_counts, title="Jobs by Company")
    
    # Salary trends (when available)
    # Location analysis  
    # Skill demand tracking
```

### Performance Roadmap

- **Redis integration** for distributed caching

- **Background job processing** with Celery

- **GraphQL API** for flexible data access

- **Elasticsearch** for advanced search capabilities

- **Machine learning** job recommendation system

This developer guide provides the foundation for contributing to and extending the AI Job Scraper. For questions or clarifications, please open an issue on GitHub or refer to the [ADRs](adrs/) for architectural context.
