# ADR-038: Simple Data Management

## Title

Simple Data Management for Local Development

## Version/Date

2.0 / August 19, 2025

## Status

**Accepted** - Focused on local development simplicity

## Description

Simple data management patterns for local development using SQLModel's built-in capabilities. Provides straightforward data persistence, basic synchronization patterns, and simple real-time updates with Reflex without complex sync engines or production optimization.

## Context

### Local Development Requirements

This data management approach focuses on:

1. **Simple Operations**: Basic CRUD operations using SQLModel
2. **Development Velocity**: Quick data operations without optimization complexity
3. **Real-Time Updates**: Basic Reflex state updates for development UI
4. **Data Persistence**: Simple file-based persistence between sessions
5. **Minimal Complexity**: Avoid production-level synchronization engines

### Integration Patterns

- **SQLModel**: Native upsert operations with `session.merge()`
- **SQLite**: Simple file-based database for development
- **Reflex State**: Direct state updates for UI reactivity
- **Async Patterns**: Basic async operations for non-blocking UI

## Decision

**Use Simple Data Management Patterns** for local development:

### Basic Data Operations

```python
# src/services/data_service.py
from sqlmodel import Session, select
from src.models.database import engine, JobModel, CompanyModel
from typing import List, Dict, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class DataService:
    """Simple data management for local development."""
    
    def __init__(self):
        self.session_factory = lambda: Session(engine)
    
    def save_job(self, job_data: dict) -> JobModel:
        """Save or update job with simple merge operation."""
        with self.session_factory() as session:
            # Check if job exists by URL
            existing = session.exec(
                select(JobModel).where(JobModel.url == job_data["url"])
            ).first()
            
            if existing:
                # Update existing job, preserving user data
                for key, value in job_data.items():
                    if key not in ["is_favorited", "notes", "application_status"]:
                        setattr(existing, key, value)
                
                session.add(existing)
                session.commit()
                session.refresh(existing)
                return existing
            else:
                # Create new job
                job = JobModel(**job_data)
                session.add(job)
                session.commit()
                session.refresh(job)
                return job
    
    def save_jobs_batch(self, jobs_data: List[dict]) -> List[JobModel]:
        """Save multiple jobs efficiently."""
        results = []
        
        with self.session_factory() as session:
            for job_data in jobs_data:
                # Simple individual saves for development
                existing = session.exec(
                    select(JobModel).where(JobModel.url == job_data["url"])
                ).first()
                
                if existing:
                    # Update existing
                    for key, value in job_data.items():
                        if key not in ["is_favorited", "notes", "application_status"]:
                            setattr(existing, key, value)
                    session.add(existing)
                    results.append(existing)
                else:
                    # Create new
                    job = JobModel(**job_data)
                    session.add(job)
                    results.append(job)
            
            session.commit()
            
            # Refresh all
            for job in results:
                session.refresh(job)
        
        return results
    
    def get_or_create_company(self, company_name: str, company_data: dict = None) -> CompanyModel:
        """Get existing company or create new one."""
        with self.session_factory() as session:
            existing = session.exec(
                select(CompanyModel).where(CompanyModel.name == company_name)
            ).first()
            
            if existing:
                return existing
            
            # Create new company
            data = company_data or {"name": company_name}
            company = CompanyModel(**data)
            session.add(company)
            session.commit()
            session.refresh(company)
            return company
    
    def mark_inactive_jobs(self, active_urls: List[str]):
        """Mark jobs as inactive if not in active URLs list."""
        with self.session_factory() as session:
            # Simple update for development
            inactive_jobs = session.exec(
                select(JobModel).where(
                    ~JobModel.url.in_(active_urls),
                    JobModel.is_active == True
                )
            ).all()
            
            for job in inactive_jobs:
                job.is_active = False
                session.add(job)
            
            session.commit()
            return len(inactive_jobs)
    
    def get_job_stats(self) -> dict:
        """Get simple job statistics."""
        with self.session_factory() as session:
            total = len(list(session.exec(select(JobModel))))
            active = len(list(session.exec(
                select(JobModel).where(JobModel.is_active == True)
            )))
            favorited = len(list(session.exec(
                select(JobModel).where(JobModel.is_favorited == True)
            )))
            
            return {
                "total_jobs": total,
                "active_jobs": active,
                "favorited_jobs": favorited,
                "inactive_jobs": total - active
            }

# Global data service instance
data_service = DataService()
```

### Real-Time Updates with Reflex

```python
# src/state/scraping_state.py
import reflex as rx
import asyncio
from src.services.data_service import data_service
from src.services.scraper_service import scraper_service
from typing import List, Dict

class ScrapingState(rx.State):
    """Simple scraping state for real-time updates."""
    
    # Scraping status
    is_scraping: bool = False
    scraping_progress: str = "Ready to scrape"
    jobs_processed: int = 0
    new_jobs_found: int = 0
    updated_jobs: int = 0
    
    # Results
    recent_jobs: List[dict] = []
    scraping_stats: dict = {}
    
    async def start_scraping(self, sources: List[str] = None):
        """Start scraping with real-time updates."""
        if self.is_scraping:
            return
        
        self.is_scraping = True
        self.scraping_progress = "Starting scraping..."
        self.jobs_processed = 0
        self.new_jobs_found = 0
        self.updated_jobs = 0
        yield
        
        try:
            # Simple async scraping with real-time updates
            sources = sources or ["indeed", "linkedin"]
            
            for source in sources:
                self.scraping_progress = f"Scraping {source}..."
                yield
                
                # Scrape jobs from source
                async for job_data in scraper_service.scrape_source(source):
                    # Save job to database
                    job = data_service.save_job(job_data)
                    
                    # Update counters
                    self.jobs_processed += 1
                    if job.id:  # New job
                        self.new_jobs_found += 1
                    else:  # Updated job
                        self.updated_jobs += 1
                    
                    # Add to recent jobs list
                    if len(self.recent_jobs) >= 10:
                        self.recent_jobs.pop(0)
                    
                    self.recent_jobs.append({
                        "title": job.title,
                        "company": job.company,
                        "location": job.location,
                        "source": source
                    })
                    
                    # Update progress
                    self.scraping_progress = f"Processed {self.jobs_processed} jobs from {source}"
                    yield
                    
                    # Small delay for demo purposes
                    await asyncio.sleep(0.1)
            
            # Final update
            self.scraping_progress = f"Completed! Found {self.new_jobs_found} new jobs, updated {self.updated_jobs}"
            self.scraping_stats = data_service.get_job_stats()
            
        except Exception as e:
            self.scraping_progress = f"Error: {str(e)}"
            logger.error(f"Scraping failed: {e}")
        
        finally:
            self.is_scraping = False
            yield
    
    def stop_scraping(self):
        """Stop scraping process."""
        self.is_scraping = False
        self.scraping_progress = "Stopped by user"
    
    def clear_progress(self):
        """Clear scraping progress."""
        self.scraping_progress = "Ready to scrape"
        self.jobs_processed = 0
        self.new_jobs_found = 0
        self.updated_jobs = 0
        self.recent_jobs = []
```

### Simple Async Scraper Service

```python
# src/services/scraper_service.py
import asyncio
import logging
from typing import AsyncGenerator, Dict, List
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime

logger = logging.getLogger(__name__)

class ScraperService:
    """Simple scraper service for local development."""
    
    def __init__(self):
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def scrape_source(self, source: str) -> AsyncGenerator[Dict, None]:
        """Scrape jobs from a source with async generator."""
        if source == "indeed":
            async for job in self._scrape_indeed():
                yield job
        elif source == "linkedin":
            async for job in self._scrape_linkedin():
                yield job
        else:
            logger.warning(f"Unknown source: {source}")
    
    async def _scrape_indeed(self) -> AsyncGenerator[Dict, None]:
        """Simple Indeed scraping (mock for development)."""
        # Mock data for development
        mock_jobs = [
            {
                "title": "Python Developer",
                "company": "Tech Corp",
                "location": "San Francisco, CA",
                "description": "Looking for a Python developer with 3+ years experience...",
                "url": "https://indeed.com/job1",
                "salary_text": "$80,000 - $120,000",
                "scraped_at": datetime.now()
            },
            {
                "title": "Data Scientist",
                "company": "Data Inc",
                "location": "New York, NY",
                "description": "Data science position with machine learning focus...",
                "url": "https://indeed.com/job2",
                "salary_text": "$100,000 - $150,000",
                "scraped_at": datetime.now()
            }
        ]
        
        for job in mock_jobs:
            await asyncio.sleep(0.5)  # Simulate scraping delay
            yield job
    
    async def _scrape_linkedin(self) -> AsyncGenerator[Dict, None]:
        """Simple LinkedIn scraping (mock for development)."""
        # Mock data for development
        mock_jobs = [
            {
                "title": "Full Stack Developer",
                "company": "Startup LLC",
                "location": "Remote",
                "description": "Remote full stack position...",
                "url": "https://linkedin.com/job1",
                "salary_text": "$70,000 - $110,000",
                "scraped_at": datetime.now()
            }
        ]
        
        for job in mock_jobs:
            await asyncio.sleep(0.5)  # Simulate scraping delay
            yield job
    
    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()

# Global scraper service instance
scraper_service = ScraperService()
```

### Integration with Reflex Pages

```python
# src/pages/scraping.py
import reflex as rx
from src.state.scraping_state import ScrapingState

def scraping_controls() -> rx.Component:
    """Simple scraping controls."""
    return rx.vstack(
        rx.heading("Job Scraping"),
        
        # Status display
        rx.text(f"Status: {ScrapingState.scraping_progress}"),
        rx.text(f"Jobs Processed: {ScrapingState.jobs_processed}"),
        rx.text(f"New Jobs: {ScrapingState.new_jobs_found}"),
        rx.text(f"Updated Jobs: {ScrapingState.updated_jobs}"),
        
        # Controls
        rx.hstack(
            rx.button(
                "Start Scraping",
                on_click=ScrapingState.start_scraping,
                disabled=ScrapingState.is_scraping,
                bg="green.500",
                color="white"
            ),
            rx.button(
                "Stop",
                on_click=ScrapingState.stop_scraping,
                disabled=~ScrapingState.is_scraping,
                bg="red.500",
                color="white"
            ),
            rx.button(
                "Clear",
                on_click=ScrapingState.clear_progress,
                disabled=ScrapingState.is_scraping
            )
        ),
        
        # Recent jobs
        rx.cond(
            ScrapingState.recent_jobs,
            rx.vstack(
                rx.heading("Recent Jobs", size="md"),
                rx.foreach(
                    ScrapingState.recent_jobs,
                    lambda job: rx.text(f"{job['title']} at {job['company']}")
                )
            )
        ),
        
        spacing="4",
        align="start"
    )

def page() -> rx.Component:
    """Scraping page."""
    return rx.container(
        scraping_controls(),
        padding="4"
    )
```

## Consequences

### Positive Outcomes

- **Simple Implementation**: Straightforward data operations without complex sync engines
- **Real-Time Updates**: Basic real-time UI updates with Reflex state
- **Development Focus**: Optimized for development workflow and debugging
- **SQLModel Native**: Leverages built-in SQLModel capabilities
- **Async Support**: Non-blocking operations for better UI responsiveness

### Negative Consequences

- **Development Only**: Not optimized for production-scale data operations
- **Limited Optimization**: Basic patterns without production performance tuning
- **Simple Error Handling**: Basic error handling for development needs
- **Mock Data**: Includes mock scrapers for development testing

### Risk Mitigation

- **Clear Documentation**: Simple patterns easy to understand and modify
- **Upgrade Path**: Clear migration to production-level data management
- **Error Logging**: Basic error logging for development debugging
- **State Recovery**: Simple state management with clear reset capabilities

## Development Guidelines

### Data Operations

- Use `data_service.save_job()` for individual job saves
- Use `data_service.save_jobs_batch()` for bulk operations
- Leverage SQLModel's `session.merge()` for upsert operations
- Preserve user data fields during updates

### Real-Time Updates

- Use Reflex state variables for live updates
- Implement async generators for streaming data
- Add small delays for demonstration purposes
- Keep UI responsive with yield statements

### Testing and Development

- Mock scrapers included for development testing
- Simple statistics available via `get_job_stats()`
- Easy data reset capabilities
- Clear progress tracking and logging

## Related ADRs

- **Supports ADR-035**: Local Development Architecture (data management component)
- **Uses ADR-037**: Local Database Setup (SQLModel operations)
- **Replaces Archived ADR-038**: Smart Data Synchronization Engine (production-focused)
- **Integrates ADR-040**: UI Component Architecture (state management)

## Success Criteria

- [ ] Simple job saving and updating works correctly
- [ ] Real-time updates display properly in Reflex UI
- [ ] User data preservation during updates
- [ ] Basic statistics and progress tracking functional
- [ ] Non-blocking scraping operations with async patterns
- [ ] Easy development testing with mock data

---

*This ADR provides simple, practical data management for local development without complex synchronization engines or production optimization.*
