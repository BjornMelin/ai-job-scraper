# Implementation Blueprint: AI Job Scraper Optimizations

## Quick Start: Priority Implementations

### 1. Database Optimization (30 minutes)

```python
# src/models.py - Add indexes for performance
from sqlmodel import Field, SQLModel, Index
from datetime import datetime
from typing import Optional

class Job(SQLModel, table=True):
    """Optimized Job model with indexes."""
    __tablename__ = "jobs"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str = Field(index=True)  # Add index
    company_id: int = Field(foreign_key="companies.id", index=True)  # Add index
    posted_date: datetime = Field(index=True)  # Add index
    application_status: str = Field(default="new", index=True)  # Add index
    salary_min: Optional[int] = Field(default=None)
    salary_max: Optional[int] = Field(default=None)
    
    # Composite indexes for common queries
    __table_args__ = (
        Index("idx_company_status", "company_id", "application_status"),
        Index("idx_posted_salary", "posted_date", "salary_min"),
    )
```

### 2. Paginated UI Implementation (1 hour)

```python
# src/ui/pages/jobs.py - Paginated job display
import streamlit as st
from typing import Optional, Dict, Any
from src.services.job_repository import JobRepository

class JobsPage:
    """Optimized jobs page with pagination."""
    
    ITEMS_PER_PAGE = 50
    
    def __init__(self):
        self.repo = JobRepository()
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize pagination state."""
        if 'page' not in st.session_state:
            st.session_state.page = 0
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
    
    def render(self):
        """Render paginated job listings."""
        st.title("Job Listings")
        
        # Filters
        self._render_filters()
        
        # Get data for current page only
        jobs, total = self._get_page_data()
        
        # Pagination controls
        self._render_pagination(total)
        
        # Display jobs
        self._render_job_grid(jobs)
    
    def _render_filters(self):
        """Render filter controls."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            search = st.text_input("Search", key="job_search")
            if search:
                st.session_state.filters['search'] = search
        
        with col2:
            status = st.selectbox(
                "Status",
                ["All", "New", "Applied", "Interview", "Rejected"],
                key="job_status"
            )
            if status != "All":
                st.session_state.filters['status'] = status
        
        with col3:
            company = st.selectbox(
                "Company",
                ["All"] + self.repo.get_company_names(),
                key="job_company"
            )
            if company != "All":
                st.session_state.filters['company'] = company
    
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def _get_page_data(self) -> tuple[list, int]:
        """Get paginated data with caching."""
        offset = st.session_state.page * self.ITEMS_PER_PAGE
        
        jobs = self.repo.get_jobs_paginated(
            offset=offset,
            limit=self.ITEMS_PER_PAGE,
            filters=st.session_state.filters
        )
        
        total = self.repo.get_job_count(st.session_state.filters)
        
        return jobs, total
    
    def _render_pagination(self, total: int):
        """Render pagination controls."""
        total_pages = (total // self.ITEMS_PER_PAGE) + 1
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⏮️ First"):
                st.session_state.page = 0
                st.rerun()
        
        with col2:
            if st.button("◀️ Prev", disabled=st.session_state.page == 0):
                st.session_state.page -= 1
                st.rerun()
        
        with col3:
            st.write(f"Page {st.session_state.page + 1} of {total_pages} ({total} jobs)")
        
        with col4:
            if st.button("Next ▶️", disabled=st.session_state.page >= total_pages - 1):
                st.session_state.page += 1
                st.rerun()
        
        with col5:
            if st.button("Last ⏭️"):
                st.session_state.page = total_pages - 1
                st.rerun()
    
    def _render_job_grid(self, jobs: list):
        """Render job cards in a grid."""
        cols = st.columns(2)
        
        for idx, job in enumerate(jobs):
            with cols[idx % 2]:
                self._render_job_card(job)
    
    def _render_job_card(self, job):
        """Render individual job card."""
        with st.container():
            st.markdown(f"### {job.title}")
            st.markdown(f"**{job.company.name}** • {job.location}")
            
            if job.salary_min or job.salary_max:
                salary = f"${job.salary_min or 0:,} - ${job.salary_max or 0:,}"
                st.markdown(f"💰 {salary}")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("View Details", key=f"view_{job.id}"):
                    st.session_state.selected_job = job.id
            
            with col2:
                new_status = st.selectbox(
                    "Status",
                    ["New", "Applied", "Interview", "Rejected"],
                    index=["new", "applied", "interview", "rejected"].index(job.application_status),
                    key=f"status_{job.id}"
                )
                if new_status.lower() != job.application_status:
                    self.repo.update_job_status(job.id, new_status.lower())
                    st.rerun()
```

### 3. Hybrid Scraper Implementation (2 hours)

```python
# src/services/hybrid_scraper.py
from typing import Dict, List, Optional
from playwright.async_api import async_playwright
from scrapegraphai import SmartScraperGraph
import asyncio
from python_jobspy import scrape_jobs
import logging

logger = logging.getLogger(__name__)

class HybridScraper:
    """Intelligent scraper that chooses optimal method per site."""
    
    # Known patterns for fast Playwright extraction
    KNOWN_PATTERNS = {
        "greenhouse.io": {
            "job_selector": "div.opening",
            "title": "a.opening-link",
            "department": "span.department",
            "location": "span.location"
        },
        "lever.co": {
            "job_selector": "div.posting",
            "title": "h5[data-qa='posting-name']",
            "department": "span.posting-categories",
            "location": "span.location"
        },
        "workday.com": {
            "job_selector": "li[data-automation-id='job']",
            "title": "a[data-automation-id='jobTitle']",
            "department": "dd[data-automation-id='department']",
            "location": "dd[data-automation-id='location']"
        }
    }
    
    def __init__(self, proxy_manager=None):
        self.proxy_manager = proxy_manager
        self.scrapegraph_config = self._get_scrapegraph_config()
    
    async def scrape(self, url: str, company_name: str) -> List[Dict]:
        """Smart scraping with fallback strategy."""
        
        # Check if it's a job board (use JobSpy)
        if any(board in url for board in ["linkedin.com", "indeed.com", "glassdoor.com"]):
            return await self._jobspy_scrape(company_name)
        
        # Check for known patterns (use Playwright)
        for pattern_domain, selectors in self.KNOWN_PATTERNS.items():
            if pattern_domain in url:
                return await self._playwright_scrape(url, selectors)
        
        # Try Playwright with generic selectors
        generic_jobs = await self._playwright_generic(url)
        if generic_jobs:
            return generic_jobs
        
        # Fallback to ScrapeGraphAI for complex sites
        return await self._scrapegraph_scrape(url)
    
    async def _jobspy_scrape(self, company_name: str) -> List[Dict]:
        """Use JobSpy for job board scraping."""
        try:
            proxy = self.proxy_manager.get_optimal_proxy() if self.proxy_manager else None
            
            jobs_df = await asyncio.to_thread(
                scrape_jobs,
                site_name=["linkedin", "indeed", "glassdoor"],
                search_term=company_name,
                results_wanted=50,
                hours_old=168,  # Last 7 days
                proxies=proxy
            )
            
            return jobs_df.to_dict('records')
            
        except Exception as e:
            logger.error(f"JobSpy scraping failed: {e}")
            return []
    
    async def _playwright_scrape(self, url: str, selectors: Dict) -> List[Dict]:
        """Fast Playwright scraping for known patterns."""
        jobs = []
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                viewport={'width': 1920, 'height': 1080}
            )
            
            if self.proxy_manager:
                proxy = self.proxy_manager.get_optimal_proxy()
                context = await browser.new_context(proxy=proxy)
            
            page = await context.new_page()
            
            try:
                await page.goto(url, wait_until='networkidle', timeout=30000)
                await page.wait_for_selector(selectors['job_selector'], timeout=10000)
                
                job_elements = await page.query_selector_all(selectors['job_selector'])
                
                for element in job_elements:
                    job = {}
                    
                    for field, selector in selectors.items():
                        if field != 'job_selector':
                            el = await element.query_selector(selector)
                            if el:
                                job[field] = await el.inner_text()
                    
                    if job:
                        jobs.append(job)
                
                logger.info(f"Playwright extracted {len(jobs)} jobs from {url}")
                
            except Exception as e:
                logger.error(f"Playwright scraping failed: {e}")
            
            finally:
                await browser.close()
        
        return jobs
    
    async def _playwright_generic(self, url: str) -> List[Dict]:
        """Try generic patterns with Playwright."""
        # Common job listing patterns
        generic_selectors = [
            {"job": "[class*='job']", "title": "h2,h3,h4", "location": "[class*='location']"},
            {"job": "[data-*='job']", "title": "a", "location": "span"},
            {"job": "article", "title": "h1,h2,h3", "location": "[class*='loc']"}
        ]
        
        for selectors in generic_selectors:
            jobs = await self._try_selectors(url, selectors)
            if jobs:
                return jobs
        
        return []
    
    async def _scrapegraph_scrape(self, url: str) -> List[Dict]:
        """Use ScrapeGraphAI for complex/unknown sites."""
        try:
            prompt = """
            Extract all job listings from this page. For each job, extract:
            - title: Job title
            - department: Department or team
            - location: Job location
            - description: Brief job description
            - apply_url: Application link
            
            Return as a list of JSON objects.
            """
            
            graph = SmartScraperGraph(
                prompt=prompt,
                source=url,
                config=self.scrapegraph_config
            )
            
            result = await asyncio.to_thread(graph.run)
            
            logger.info(f"ScrapeGraphAI extracted {len(result)} jobs from {url}")
            return result
            
        except Exception as e:
            logger.error(f"ScrapeGraphAI scraping failed: {e}")
            return []
    
    def _get_scrapegraph_config(self) -> Dict:
        """Get ScrapeGraphAI configuration."""
        return {
            "llm": {
                "provider": "groq",
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.1
            },
            "embedder": {
                "provider": "openai",
                "model": "text-embedding-3-small"
            },
            "verbose": True,
            "headless": True,
            "cache_enabled": True
        }
```

### 4. Background Task Manager (1 hour)

```python
# src/services/background_scraper.py
import asyncio
import streamlit as st
from typing import List, Dict, Callable, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BackgroundScraperManager:
    """Manages background scraping with progress updates."""
    
    def __init__(self, hybrid_scraper, job_repository):
        self.scraper = hybrid_scraper
        self.repo = job_repository
        self.current_task: Optional[asyncio.Task] = None
        
    async def scrape_companies_with_progress(
        self,
        companies: List[Dict],
        progress_callback: Callable[[float, str], None]
    ) -> Dict:
        """Scrape companies with real-time progress updates."""
        
        results = {
            "total": len(companies),
            "successful": 0,
            "failed": 0,
            "jobs_found": 0,
            "errors": []
        }
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent scrapes
        
        async def scrape_company(company: Dict, index: int):
            """Scrape single company with semaphore."""
            async with semaphore:
                try:
                    progress_callback(
                        index / results["total"],
                        f"Scraping {company['name']}..."
                    )
                    
                    # Perform scraping
                    if company.get('careers_url'):
                        jobs = await self.scraper.scrape(
                            company['careers_url'],
                            company['name']
                        )
                    else:
                        jobs = await self.scraper._jobspy_scrape(company['name'])
                    
                    # Save jobs in batch
                    if jobs:
                        await self._save_jobs_batch(jobs, company['id'])
                        results["jobs_found"] += len(jobs)
                    
                    results["successful"] += 1
                    
                    progress_callback(
                        (index + 1) / results["total"],
                        f"✓ {company['name']}: {len(jobs)} jobs found"
                    )
                    
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append({
                        "company": company['name'],
                        "error": str(e)
                    })
                    
                    progress_callback(
                        (index + 1) / results["total"],
                        f"✗ {company['name']}: {str(e)}"
                    )
                    
                    logger.error(f"Failed to scrape {company['name']}: {e}")
        
        # Create tasks for all companies
        tasks = [
            scrape_company(company, idx)
            for idx, company in enumerate(companies)
        ]
        
        # Execute with gather
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def _save_jobs_batch(self, jobs: List[Dict], company_id: int):
        """Save jobs in efficient batches."""
        batch_size = 50
        
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            
            # Transform to Job models
            job_models = []
            for job_data in batch:
                job_model = self._transform_to_model(job_data, company_id)
                if job_model:
                    job_models.append(job_model)
            
            # Bulk insert
            if job_models:
                await self.repo.bulk_insert_jobs(job_models)
    
    def _transform_to_model(self, job_data: Dict, company_id: int) -> Optional[Dict]:
        """Transform scraped data to Job model."""
        try:
            return {
                "title": job_data.get("title", "Unknown Title"),
                "company_id": company_id,
                "location": job_data.get("location", "Remote"),
                "description": job_data.get("description", ""),
                "posted_date": job_data.get("posted_date", datetime.now()),
                "salary_min": job_data.get("salary_min"),
                "salary_max": job_data.get("salary_max"),
                "apply_url": job_data.get("apply_url", ""),
                "source_url": job_data.get("source_url", ""),
                "department": job_data.get("department", ""),
                "employment_type": job_data.get("employment_type", "Full-time"),
                "application_status": "new"
            }
        except Exception as e:
            logger.error(f"Failed to transform job data: {e}")
            return None
```

### 5. Streamlit Integration Example (30 minutes)

```python
# src/ui/pages/scraping.py
import streamlit as st
import asyncio
from src.services.background_scraper import BackgroundScraperManager
from src.services.hybrid_scraper import HybridScraper
from src.services.smart_proxy_manager import SmartProxyManager

class ScrapingPage:
    """Scraping page with async background tasks."""
    
    def __init__(self):
        self.proxy_manager = SmartProxyManager()
        self.hybrid_scraper = HybridScraper(self.proxy_manager)
        self.scraper_manager = BackgroundScraperManager(
            self.hybrid_scraper,
            self.job_repository
        )
    
    def render(self):
        """Render scraping interface."""
        st.title("🔍 Job Scraping")
        
        # Company selection
        companies = self._get_active_companies()
        selected = st.multiselect(
            "Select companies to scrape",
            options=[c['name'] for c in companies],
            default=[]
        )
        
        if st.button("Start Scraping", disabled=not selected):
            self._run_scraping(selected, companies)
    
    def _run_scraping(self, selected_names: List[str], all_companies: List[Dict]):
        """Run scraping with progress tracking."""
        
        # Filter selected companies
        companies_to_scrape = [
            c for c in all_companies if c['name'] in selected_names
        ]
        
        # Create containers for progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()
        
        # Progress callback
        def update_progress(progress: float, message: str):
            progress_bar.progress(progress)
            status_text.text(message)
        
        # Run async scraping
        try:
            results = asyncio.run(
                self.scraper_manager.scrape_companies_with_progress(
                    companies_to_scrape,
                    update_progress
                )
            )
            
            # Display results
            with results_container:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Jobs Found", results["jobs_found"])
                
                with col2:
                    st.metric("Successful", results["successful"])
                
                with col3:
                    st.metric("Failed", results["failed"])
                
                if results["errors"]:
                    with st.expander("Errors"):
                        for error in results["errors"]:
                            st.error(f"{error['company']}: {error['error']}")
            
            st.success("Scraping completed!")
            
        except Exception as e:
            st.error(f"Scraping failed: {e}")
```

## Installation Commands

```bash
# Add new dependencies
uv add playwright crawl4ai tenacity asyncio-throttle

# Install Playwright browsers
playwright install chromium

# Run database migrations for indexes
alembic revision --autogenerate -m "Add performance indexes"
alembic upgrade head
```

## Testing the Optimizations

```python
# tests/test_performance.py
import pytest
import asyncio
from src.services.hybrid_scraper import HybridScraper

@pytest.mark.asyncio
async def test_hybrid_scraper_performance():
    """Test that hybrid scraper is faster than ScrapeGraphAI alone."""
    scraper = HybridScraper()
    
    # Test known pattern (should use Playwright)
    start = asyncio.get_event_loop().time()
    jobs = await scraper.scrape("https://example.greenhouse.io", "Example Co")
    playwright_time = asyncio.get_event_loop().time() - start
    
    # Force ScrapeGraphAI
    start = asyncio.get_event_loop().time()
    jobs = await scraper._scrapegraph_scrape("https://example.greenhouse.io")
    scrapegraph_time = asyncio.get_event_loop().time() - start
    
    # Playwright should be at least 2x faster
    assert playwright_time < scrapegraph_time / 2

@pytest.mark.benchmark
def test_pagination_performance(benchmark):
    """Benchmark pagination vs full load."""
    repo = JobRepository()
    
    # Benchmark paginated query
    result = benchmark(repo.get_jobs_paginated, offset=0, limit=50)
    assert len(result) <= 50
```

## Monitoring & Metrics

```python
# src/utils/metrics.py
import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

def track_performance(name: str):
    """Decorator to track function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                logger.info(f"{name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"{name} failed after {duration:.2f}s: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.info(f"{name} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"{name} failed after {duration:.2f}s: {e}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

# Usage
@track_performance("job_scraping")
async def scrape_jobs(url: str):
    # Scraping logic
    pass
```

## Deployment Checklist

- [ ] Add database indexes (run migrations)
- [ ] Install Playwright browsers
- [ ] Update environment variables for new services
- [ ] Test pagination with 5000+ records
- [ ] Verify background tasks don't block UI
- [ ] Test proxy rotation under load
- [ ] Benchmark scraping performance
- [ ] Update documentation
- [ ] Deploy to production

## Expected Performance Improvements

| Operation | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Load 5000 jobs | 11s | 100ms | 110x |
| Scrape 100 jobs | 5 min | 2 min | 2.5x |
| UI responsiveness | Blocking | Non-blocking | ∞ |
| Memory usage (5000 jobs) | 500MB | 50MB | 10x |
| Greenhouse.io scrape | 3s (ScrapeGraphAI) | 290ms (Playwright) | 10x |

## Next Steps

1. Implement Phase 1 (Database + Pagination) - **Today**
2. Test with production data - **Tomorrow**
3. Implement Phase 2 (Hybrid Scraper) - **Day 3-4**
4. Full integration testing - **Day 5**
5. Deploy to production - **Day 6-7**
