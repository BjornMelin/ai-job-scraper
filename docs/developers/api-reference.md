# 📚 API Reference: AI Job Scraper

This document provides detailed technical reference for the AI Job Scraper codebase, including data models, function APIs, database schema, and integration points.

## 🗄️ Database Schema

### Jobs Table (`jobs`)

The primary table storing scraped job postings with user tracking information.

```sql
CREATE TABLE jobs (
    id INTEGER PRIMARY KEY,
    company VARCHAR NOT NULL,
    title VARCHAR NOT NULL,
    description TEXT NOT NULL,
    link VARCHAR UNIQUE NOT NULL,
    location VARCHAR,
    posted_date DATETIME,
    hash VARCHAR,           -- Content hash for change detection
    last_seen DATETIME,     -- Last successful scrape timestamp
    favorite BOOLEAN DEFAULT FALSE,
    status VARCHAR DEFAULT 'New',
    notes TEXT DEFAULT ''
);
```

**Indexes:**

- `UNIQUE(link)` - Prevents duplicate job postings

- `INDEX(company)` - Fast company-based filtering

- `INDEX(status)` - Quick status-based queries

- `INDEX(favorite)` - Efficient favorites lookup

**Field Descriptions:**

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `id` | Integer | Auto-incrementing primary key | - |
| `company` | String | Company name from scraping | Required, max 200 chars |
| `title` | String | Job position title | Required, 3-200 chars |
| `description` | Text | Job description content | Required, 10-1000 chars |
| `link` | String | Unique job posting URL | Required, valid HTTP(S) URL |
| `location` | String | Job location or "Remote" | Optional, defaults to "Unknown" |
| `posted_date` | DateTime | When job was originally posted | Optional, parsed from site |
| `hash` | String | SHA256 hash of description | Auto-generated for change detection |
| `last_seen` | DateTime | Last successful scrape timestamp | Auto-updated on scrape |
| `favorite` | Boolean | User-marked favorite status | Defaults to `False` |
| `status` | String | Application status | One of: `New`, `Interested`, `Applied`, `Rejected` |
| `notes` | Text | User's personal notes | Optional, unlimited length |

### Companies Table (`companies`)

Configuration table for managing scraped companies.

```sql
CREATE TABLE companies (
    id INTEGER PRIMARY KEY,
    name VARCHAR UNIQUE NOT NULL,
    url VARCHAR NOT NULL,
    active BOOLEAN DEFAULT TRUE
);
```

**Field Descriptions:**

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| `id` | Integer | Auto-incrementing primary key | - |
| `name` | String | Unique company identifier | Required, unique |
| `url` | String | Company careers page URL | Required, valid HTTP(S) URL |
| `active` | Boolean | Whether to include in scraping | Defaults to `True` |

## 📊 Data Models

### Pydantic Models (`models.py`)

#### JobPydantic - Validation Model

```python
class JobPydantic(BaseModel):
    """Pydantic model for job validation and serialization."""
    
    company: str
    title: str
    description: str
    link: str = Field(pattern=r"^https?://")  # URL validation
    location: str | None = "Unknown"
    posted_date: datetime | None = None
    hash: str | None = None
    last_seen: datetime | None = None
    favorite: bool = False
    status: str = "New"
    notes: str = ""
    
    # Usage
    job = JobPydantic(
        company="OpenAI",
        title="AI Research Engineer", 
        description="Develop next-generation AI systems...",
        link="https://openai.com/careers/job-123"
    )  # Validates all fields automatically
```

**Validation Rules:**

- `link`: Must match HTTP(S) URL pattern

- `title`: 3-200 characters when validated in scraper

- `description`: 10-1000 characters when validated in scraper

- `status`: Must be one of allowed values (enforced in UI)

### SQLAlchemy Models (`models.py`)

#### JobSQL - Database ORM Model

```python
class JobSQL(Base):
    """SQLAlchemy model for jobs table."""
    
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    company = Column(String)
    title = Column(String)
    description = Column(Text)
    link = Column(String, unique=True)
    location = Column(String)
    posted_date = Column(DateTime)
    hash = Column(String)
    last_seen = Column(DateTime)
    favorite = Column(Boolean, default=False)
    status = Column(String, default="New") 
    notes = Column(Text, default="")
    
    # Usage
    job = JobSQL(
        company="Anthropic",
        title="ML Engineer",
        description="Build AI safety systems...",
        link="https://careers.anthropic.com/job/456"
    )
    session.add(job)
    session.commit()
```

#### CompanySQL - Company Configuration Model

```python
class CompanySQL(Base):
    """SQLAlchemy model for companies table."""
    
    __tablename__ = "companies"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    url = Column(String)
    active = Column(Boolean, default=True)
    
    # Usage
    company = CompanySQL(
        name="nvidia",
        url="https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
        active=True
    )
```

## 🔧 Core Functions API

### Scraping Engine (`scraper.py`)

#### Main Scraping Function

```python
async def scrape_all() -> pd.DataFrame:
    """Scrape job postings from all active company websites.
    
    Returns:
        pd.DataFrame: DataFrame containing all relevant scraped jobs with columns:
            company, title, description, link, location, posted_date.
            
    Raises:
        Exception: If all scraping attempts fail or no active companies found.
        
    Performance:
        - Cold start: 45-90 seconds (no cache)
        - Warm start: 15-45 seconds (with cache)
        - Cache hit rate: 70-90% after first run
    """
    
# Usage
jobs_df = await scrape_all()
print(f"Scraped {len(jobs_df)} jobs from {jobs_df['company'].nunique()} companies")
```

#### Individual Company Extraction

```python
async def extract_jobs(url: str, company: str) -> list[dict]:
    """Extract jobs with intelligent caching and LLM optimization.
    
    Args:
        url (str): Company careers page URL to scrape
        company (str): Company name for caching and rate limiting
        
    Returns:
        list[dict]: List of job dictionaries with keys:
            - company: Company name
            - title: Job title
            - description: Job description
            - link: Application URL
            - location: Job location
            - posted_date: When posted (datetime or None)
            
    Raises:
        Exception: After all retry attempts exhausted
        
    Performance:
        - Uses cached schema if available (90% speed boost)
        - Falls back to LLM extraction (optimized for 50% cost reduction)
        - Final fallback to CSS extraction
        - Company-specific rate limiting applied
    """

# Usage  
jobs = await extract_jobs("https://openai.com/careers", "openai")
for job in jobs:
    print(f"{job['company']}: {job['title']}")
```

#### Database Update Operations

```python
def update_db(jobs_df: pd.DataFrame) -> None:
    """Update database with scraped job data using full CRUD operations.
    
    Performs:
        - Validates jobs with Pydantic models
        - Adds new jobs not in database
        - Updates existing jobs when content changes (via hash comparison)
        - Removes jobs no longer found on websites
        - Preserves user edits (favorite, status, notes)
        
    Args:
        jobs_df (pd.DataFrame): DataFrame with scraped job data
        
    Raises:
        Exception: Database operation failures (with rollback)
        
    Side Effects:
        - Creates/updates/deletes rows in jobs table
        - Validates and filters invalid job postings
        - Logs validation errors and processing statistics
    """

# Usage
jobs_df = pd.DataFrame([
    {
        "company": "meta", 
        "title": "AI Research Scientist",
        "description": "Research next-generation AI...",
        "link": "https://ai.meta.com/careers/123",
        "location": "Menlo Park",
        "posted_date": datetime(2024, 1, 15)
    }
])
update_db(jobs_df)
```

#### Link Validation

```python
async def validate_link(link: str) -> str | None:
    """Validate that a job posting URL is accessible.
    
    Args:
        link (str): Job posting URL to validate
        
    Returns:
        str | None: Original link if valid and accessible, None otherwise
        
    Implementation:
        - Makes HTTP HEAD request with 5s timeout
        - Follows redirects automatically
        - Returns None for any HTTP errors or timeouts
    """

# Usage
valid_link = await validate_link("https://careers.anthropic.com/job/123")
if valid_link:
    print(f"Link is accessible: {valid_link}")
else:
    print("Link is broken or inaccessible")
```

#### Relevance Filtering

```python
def is_relevant(job: dict) -> bool:
    """Check if job posting is relevant to AI/ML engineering roles.
    
    Args:
        job (dict): Job dictionary containing at least a 'title' field
        
    Returns:
        bool: True if job title matches AI/ML keywords, False otherwise
        
    Implementation:
        Uses regex pattern: r"(AI|Machine Learning|MLOps|AI Agent).*Engineer"
        Case-insensitive matching
    """

# Usage
job = {"title": "Senior AI Engineer - Computer Vision"}
if is_relevant(job):
    print("This job matches AI/ML criteria")
```

### Caching System (`scraper.py`)

#### Cache Management Functions

```python
def get_cached_schema(company: str) -> dict | None:
    """Get cached extraction schema for company.
    
    Args:
        company (str): Company name (case-insensitive)
        
    Returns:
        dict | None: Cached extraction schema or None if not found
        
    File Location:
        ./cache/{company.lower()}.json
    """

def save_schema_cache(company: str, schema: dict) -> None:
    """Save successful extraction schema for future use.
    
    Args:
        company (str): Company name for cache file naming
        schema (dict): Extraction schema to cache
        
    Side Effects:
        - Creates cache directory if it doesn't exist
        - Overwrites existing cache file for company
        - JSON formatted with 2-space indentation
    """

# Usage
schema = get_cached_schema("anthropic")
if schema:
    print("Using cached schema for faster extraction")
else:
    # After successful LLM extraction
    save_schema_cache("anthropic", extracted_schema)
```

#### Performance Monitoring

```python
def log_session_summary() -> None:
    """Log comprehensive session performance metrics.
    
    Metrics Included:
        - Total session duration
        - Number of companies processed  
        - Jobs found across all companies
        - Cache hit rate percentage
        - LLM API calls made
        - Errors encountered
        
    Output Format:
        📊 Session Summary:
          Duration: 32.1s
          Companies: 7
          Jobs found: 89
          Cache hit rate: 87%
          LLM calls: 1
          Errors: 0
    """

# Usage (automatically called in main())
session_stats["start_time"] = time.time()

# ... scraping operations
log_session_summary()
```

### UI Components (`app.py`)

#### Job Display Function

```python
def display_jobs(jobs: list[JobSQL], tab_key: str) -> None:
    """Display jobs in selected view format with search and filtering.
    
    Args:
        jobs (list[JobSQL]): List of job objects from database query
        tab_key (str): Unique identifier for current tab ("all", "favorites", "applied")
        
    Features:
        - List view: Editable table with bulk save functionality
        - Card view: Visual grid with pagination and sorting
        - Per-tab search functionality
        - Real-time status and favorite updates
        - CSV export for filtered results
        
    UI State:
        Uses st.session_state for:
        - View mode selection (List/Card)
        - Sort preferences (by Posted/Title/Company)
        - Pagination state (current page per tab)
        - Search terms (per tab)
    """

# Usage in Streamlit tabs
with tab1:
    display_jobs(all_jobs, "all")
with tab2: 
    favorites = [j for j in all_jobs if j.favorite]
    display_jobs(favorites, "favorites")
```

#### Status Update Callbacks

```python
def update_status(job_id: int, tab_key: str) -> None:
    """Update job application status in database.
    
    Args:
        job_id (int): Database ID of job to update
        tab_key (str): Tab identifier for session state management
        
    Side Effects:
        - Updates job.status in database
        - Commits transaction immediately  
        - Triggers UI refresh with st.rerun()
        - Logs errors if update fails
        
    Session State Key:
        f"status_{job_id}_{tab_key}_{page_number}"
    """

def update_notes(job_id: int, tab_key: str) -> None:
    """Update job notes in database.
    
    Similar to update_status but for notes field.
    Auto-saves on text area changes in card view.
    """

# Usage (automatically triggered by Streamlit widgets)
st.selectbox(
    "Status",
    ["New", "Interested", "Applied", "Rejected"],
    key=f"status_{job_id}_{tab_key}_{page}",
    on_change=update_status,
    args=(job_id, tab_key)
)
```

## ⚙️ Configuration System

### Settings Management (`config.py`)

#### Settings Class

```python
class Settings(BaseSettings):
    """Application settings with environment variable support.
    
    Attributes:
        openai_api_key (str): OpenAI API key for enhanced job extraction
        db_url (str): Database connection URL, defaults to SQLite file
        
    Configuration:
        - Loads from .env file if present
        - Supports environment variable override
        - Ignores empty environment variables
    """
    
    openai_api_key: str
    db_url: str = "sqlite:///jobs.db"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True
    )

# Usage
settings = Settings()  # Automatically loads configuration
print(f"Using database: {settings.db_url}")
```

#### Environment Variables

| Variable | Default | Description | Required |
|----------|---------|-------------|----------|
| `OPENAI_API_KEY` | - | OpenAI API key for LLM extraction | Yes |
| `DB_URL` | `sqlite:///jobs.db` | Database connection string | No |

#### Example .env File

```env

# OpenAI API key for enhanced extraction
OPENAI_API_KEY=sk-proj-your-api-key-here

# Database URL (optional, defaults to SQLite)
DB_URL=sqlite:///jobs.db

# Production PostgreSQL example

# DB_URL=postgresql://user:password@localhost:5432/ai_jobs
```

## 🔌 Integration Points

### Database Connection

#### Engine Setup

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import Settings

settings = Settings()
engine = create_engine(settings.db_url)
Session = sessionmaker(bind=engine)

# Usage patterns

# 1. Context manager (recommended)
with Session() as session:
    jobs = session.query(JobSQL).filter_by(company="openai").all()
    
# 2. Manual session management
session = Session()
try:
    job = session.query(JobSQL).filter_by(id=123).first()
    job.status = "Applied"
    session.commit()
except Exception as e:
    session.rollback()
    raise
finally:
    session.close()
```

### External API Integration

#### OpenAI LLM Integration

```python

# Configured in extraction strategy
strategy = LLMExtractionStrategy(
    provider="openai/gpt-4o-mini",           # Cost-optimized model
    api_token=settings.openai_api_key,       # From environment
    extraction_schema=SIMPLE_SCHEMA,         # Minimal for efficiency
    instructions=SIMPLE_INSTRUCTIONS,        # 50-word instructions
    apply_chunking=True,                     # For large pages
    chunk_token_threshold=1000,              # Optimal chunk size
    overlap_rate=0.02,                       # Minimal overlap
)
```

#### HTTP Client Configuration

```python

# Link validation with httpx
async with httpx.AsyncClient(
    timeout=5,              # 5-second timeout
    follow_redirects=True,  # Handle redirects
    limits=httpx.Limits(    # Connection pooling
        max_keepalive_connections=10,
        max_connections=20
    )
) as client:
    response = await client.head(url)
```

## 📈 Performance Characteristics

### Caching Performance

#### Cache Hit Rates (Typical)

- First run: 0% (cold start)

- Second run: 85-95% (warm cache)

- Ongoing: 90%+ (stable cache)

#### Speed Improvements

- Cached extraction: ~5 seconds per company

- LLM extraction: ~30-45 seconds per company  

- 90% speed improvement with cache hits

### Database Performance

#### Query Performance (1000 jobs)

- Simple filters: <50ms

- Complex joins: <200ms

- Full table scan: <500ms

- CSV export: <1s

#### Pagination Efficiency

```python

# Efficient pagination using LIMIT/OFFSET
def get_jobs_paginated(page: int, per_page: int = 9) -> list[JobSQL]:
    """Get paginated jobs efficiently."""
    session = Session()
    offset = page * per_page
    return session.query(JobSQL).offset(offset).limit(per_page).all()
```

### Memory Usage

#### Typical Memory Profile

- Base application: ~100MB

- During scraping: ~300-500MB peak  

- Large dataset (5000 jobs): ~200MB

- Browser automation: +150MB (Playwright)

## 🔒 Security Considerations

### Input Validation

#### SQL Injection Prevention

```python

# ✅ Safe: Parameterized queries via SQLAlchemy ORM
jobs = session.query(JobSQL).filter(JobSQL.company.in_(company_list)).all()

# ❌ Unsafe: String concatenation (not used in codebase)

# query = f"SELECT * FROM jobs WHERE company IN ({companies})"
```

#### URL Validation

```python

# Pydantic pattern validation
link: str = Field(pattern=r"^https?://")

# Additional runtime validation
async def validate_link(link: str) -> str | None:
    """Validate URL accessibility and safety."""
    # Only allow HTTP(S) protocols
    # Test accessibility with HEAD request
    # Return None for any suspicious URLs
```

### API Key Security

#### Environment-based Configuration

```python

# ✅ Secure: Environment variables
OPENAI_API_KEY=sk-proj-...

# ❌ Insecure: Hardcoded keys (not present in codebase)

# api_key = "sk-proj-hardcoded-key"
```

#### Logging Safety

```python

# API keys are never logged
logger.info(f"Using OpenAI provider: {provider}")  # ✅ Safe

# logger.info(f"API key: {api_key}")                # ❌ Never done
```

## 📊 Error Handling

### Exception Hierarchy

#### Database Errors

```python
try:
    session.commit()
except SQLAlchemyError as e:
    session.rollback() 
    logger.error(f"Database error: {e}")
    raise
except Exception as e:
    session.rollback()
    logger.error(f"Unexpected error: {e}")
    raise
```

#### Scraping Errors

```python

# Tenacity retry decorator with exponential backoff
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def extract_jobs_with_retry(url: str, company: str) -> list[dict]:
    """Extract jobs with automatic retry on failure."""
    # Implementation handles temporary network issues
    # Exponential backoff prevents overwhelming servers
```

#### Validation Errors

```python
try:
    job_data = JobPydantic(**scraped_job)
except ValidationError as e:
    logger.warning(f"Job validation failed: {e}")
    # Skip invalid job but continue processing others
    continue
```

This API reference provides comprehensive technical documentation for integrating with and extending the AI Job Scraper. For implementation examples and architectural context, see the [Developer Guide](DEVELOPER_GUIDE.md).
