# AI Job Scraper - Implementation Guide

## Overview

This guide provides actionable implementation instructions for the AI Job Scraper. For architectural decisions, see the ADR documents. For configuration, see inference_config.yaml.

**Target**: Deploy working MVP in 1 week with library-first implementation.

## Quick Start

### Prerequisites

- Python 3.11+
- NVIDIA GPU with 16GB+ VRAM (RTX 4090 or similar)
- Docker and docker-compose
- 32GB system RAM recommended

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/ai-job-scraper.git
cd ai-job-scraper

# Install dependencies with uv
uv sync

# Setup environment
cp .env.example .env
# Edit .env with your API keys (if any)

# Start services
docker-compose up -d redis qdrant

# Initialize database
uv run alembic upgrade head

# Download models (one-time)
uv run python scripts/download_models.py

# Run application
uv run python src/main.py
```

## Project Structure

```text
ai-job-scraper/
├── src/
│   ├── main.py                 # FastAPI + Reflex entry point
│   ├── models/                 # SQLModel database models
│   ├── scraping/               # Scraping implementations
│   │   ├── jobspy_scraper.py  # Multi-board scraping
│   │   ├── crawl4ai_scraper.py # AI-powered extraction
│   │   └── controller.py      # Scraping orchestration
│   ├── inference/              # Local LLM integration
│   │   ├── vllm_client.py     # vLLM client wrapper
│   │   └── structured.py      # Outlines structured output
│   ├── ui/                    # Reflex UI components
│   │   ├── state.py           # Global state management
│   │   ├── pages/             # Page components
│   │   └── components/        # Reusable components
│   ├── tasks/                 # Background task definitions
│   └── services/              # Business logic services
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── docker-compose.yml         # Service orchestration
├── pyproject.toml            # Dependencies
└── inference_config.yaml     # LLM configuration
```

## Core Implementation Patterns

### 1. Database Models (SQLModel)

```python
from sqlmodel import SQLModel, Field, Relationship
from datetime import datetime
from typing import Optional

class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    company: str = Field(index=True)
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    description: str
    posted_date: datetime
    url: str = Field(unique=True)
    source: str  # 'linkedin', 'indeed', etc.
    
    # Relationships
    applications: list["Application"] = Relationship(back_populates="job")
    
    # Computed fields
    @property
    def salary_range(self) -> Optional[tuple[float, float]]:
        if self.salary_min and self.salary_max:
            return (self.salary_min, self.salary_max)
        return None
```

### 2. Scraping with Library Features

```python
from python_jobspy import scrape_jobs
from crawl4ai import AsyncWebCrawler
import asyncio

class ScrapingController:
    def __init__(self):
        self.jobspy_config = {
            "site_name": ["linkedin", "indeed", "glassdoor"],
            "results_wanted": 50,
            "hours_old": 24
        }
    
    async def scrape_all_sources(self, search_term: str):
        # Use JobSpy for multi-board scraping
        jobs = await self.scrape_with_jobspy(search_term)
        
        # Use Crawl4AI for company pages
        for job in jobs:
            if job.needs_enrichment:
                await self.enrich_with_crawl4ai(job)
        
        return jobs
    
    async def scrape_with_jobspy(self, search_term: str):
        # JobSpy handles anti-bot, pagination, multiple sites
        return scrape_jobs(
            search_term=search_term,
            **self.jobspy_config
        )
    
    async def enrich_with_crawl4ai(self, job):
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(
                url=job.company_url,
                bypass_cache=False,
                anti_bot=True,
                extract_schema={
                    "type": "object",
                    "properties": {
                        "benefits": {"type": "array"},
                        "tech_stack": {"type": "array"},
                        "company_size": {"type": "string"}
                    }
                }
            )
            job.update(result.extracted_data)
```

### 3. Local LLM with vLLM

```python
from vllm import LLM, SamplingParams
from outlines import models, generate

class InferenceClient:
    def __init__(self):
        # vLLM handles memory management automatically
        self.llm = LLM(
            model="Qwen/Qwen3-4B-Instruct-2507",
            gpu_memory_utilization=0.85,
            swap_space=4,  # Automatic CPU offload
            dtype="half"
        )
        
    def extract_job_info(self, text: str, schema):
        # Outlines guarantees valid JSON
        model = models.VLLM(self.llm)
        generator = generate.json(model, schema)
        return generator(text)
    
    def process_batch(self, texts: list[str]):
        # vLLM optimizes batch processing automatically
        params = SamplingParams(
            temperature=0.1,
            max_tokens=2000
        )
        return self.llm.generate(texts, params)
```

### 4. Reflex UI with Real-time Updates

```python
import reflex as rx
from typing import list

class AppState(rx.State):
    """Global application state"""
    jobs: list[Job] = []
    is_scraping: bool = False
    scraping_progress: int = 0
    
    @rx.event(background=True)
    async def start_scraping(self, search_term: str):
        """Background scraping with real-time updates"""
        async with self:
            self.is_scraping = True
            self.scraping_progress = 0
        
        # Stream updates via WebSocket
        async for job in scrape_jobs_stream(search_term):
            async with self:
                self.jobs.append(job)
                self.scraping_progress += 1
        
        async with self:
            self.is_scraping = False

def job_card(job: Job):
    """Reusable job card component"""
    return rx.card(
        rx.heading(job.title),
        rx.text(job.company),
        rx.badge(f"${job.salary_min}-${job.salary_max}"),
        rx.button(
            "Apply",
            on_click=lambda: AppState.apply_to_job(job.id)
        )
    )

def index():
    """Main page"""
    return rx.container(
        rx.heading("AI Job Scraper"),
        rx.input(
            placeholder="Search jobs...",
            on_submit=AppState.start_scraping
        ),
        rx.cond(
            AppState.is_scraping,
            rx.progress(value=AppState.scraping_progress),
            rx.foreach(AppState.jobs, job_card)
        )
    )

app = rx.App()
app.add_page(index)
```

### 5. Background Tasks with RQ

```python
from rq import Queue
from redis import Redis
import asyncio

redis_conn = Redis(host='localhost', port=6379)
q = Queue(connection=redis_conn)

def enqueue_scraping(search_term: str, location: str = None):
    """Enqueue scraping task"""
    return q.enqueue(
        scrape_jobs_task,
        search_term,
        location,
        job_timeout='30m',
        result_ttl=86400,
        failure_ttl=86400
    )

async def scrape_jobs_task(search_term: str, location: str = None):
    """Background scraping task"""
    controller = ScrapingController()
    jobs = await controller.scrape_all_sources(search_term)
    
    # Process with local LLM if under threshold
    if len(jobs) < 100:  # 8000 token threshold
        inference = InferenceClient()
        for job in jobs:
            job.enriched = inference.extract_job_info(job.description)
    
    # Save to database
    async with get_session() as session:
        session.add_all(jobs)
        await session.commit()
    
    return {"jobs_found": len(jobs)}
```

## Configuration

### Environment Variables (.env)

```bash
# Database
DATABASE_URL=sqlite+aiosqlite:///./data/jobs.db

# Redis
REDIS_URL=redis://localhost:6379

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Models
MODEL_PATH=./models
DEFAULT_MODEL=Qwen/Qwen3-4B-Instruct-2507

# Optional API Keys (for fallback)
OPENAI_API_KEY=sk-...  # Optional cloud fallback
```

### Docker Services

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite+aiosqlite:///./data/jobs.db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - redis
      - qdrant
    volumes:
      - ./data:/app/data
      - ./models:/app/models

volumes:
  redis_data:
  qdrant_data:
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test
uv run pytest tests/test_scraping.py::test_jobspy_integration
```

## Deployment

### Local Development

```bash
# Start with hot reload
uv run reflex run --env dev

# Access at http://localhost:3000
```

### Production

```bash
# Build optimized frontend
uv run reflex export --frontend-only

# Run with gunicorn
uv run gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.main:app

# Or use Docker
docker-compose up --build
```

## Performance Optimization

### Key Metrics

- **Initial page load**: < 2s
- **Job scraping**: 50 jobs/minute
- **LLM inference**: 300+ tokens/second (RTX 4090)
- **Database queries**: < 100ms
- **WebSocket latency**: < 200ms

### Optimization Tips

1. **Use vLLM's built-in features** - Don't implement custom memory management
2. **Leverage Crawl4AI caching** - Set bypass_cache=False for repeated URLs
3. **Batch database operations** - Use bulk_insert_mappings()
4. **Enable Reflex production mode** - reflex run --env prod
5. **Use Redis for session caching** - Reduce database load

## Troubleshooting

### Common Issues

1. **VRAM out of memory**
   - Solution: Reduce gpu_memory_utilization to 0.80
   - Enable swap_space=8 in vLLM config

2. **Slow scraping**
   - Solution: Increase concurrent workers in JobSpy
   - Use Crawl4AI's incremental mode

3. **WebSocket disconnections**
   - Solution: Implement reconnection logic in Reflex
   - Check nginx/proxy timeouts

4. **Model switching delays**
   - Solution: Keep only one model loaded (hardware constraint)
   - Use cloud API for overflow

## Next Steps

1. **Week 1**: Core scraping and database
2. **Week 2**: UI implementation with Reflex
3. **Week 3**: Local LLM integration
4. **Week 4**: Testing and optimization
5. **Week 5**: Deployment and monitoring

## Resources

- [Reflex Documentation](https://reflex.dev/docs/)
- [vLLM Documentation](https://docs.vllm.ai/)
- [JobSpy GitHub](https://github.com/Bunsly/JobSpy)
- [Crawl4AI Documentation](https://crawl4ai.com/mkdocs/)
