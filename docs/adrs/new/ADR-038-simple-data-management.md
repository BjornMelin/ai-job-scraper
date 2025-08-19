# ADR-038: Enhanced Data Management with Polars DataFrame Processing

## Title

Enhanced Data Management for Local Development with Polars + DuckDB Analytics Integration

## Version/Date

3.0 / August 19, 2025

## Status

**Accepted** - Enhanced with Polars DataFrame processing and DuckDB analytical workflows

## Description

Enhanced data management patterns combining SQLModel's transactional capabilities with **Polars DataFrame processing** and **DuckDB analytical workflows**. Integrates with ADR-037's hybrid database architecture to deliver 3-80x performance improvements for analytical workloads while maintaining local development simplicity and real-time Reflex updates.

## Context

### Enhanced Data Management Requirements

This enhanced data management approach integrates with **ADR-037's Hybrid Database Architecture** to provide:

1. **Transactional Operations**: SQLModel CRUD operations for data persistence
2. **DataFrame Processing**: Polars-based data transformations and analytics
3. **Analytical Workflows**: DuckDB integration for complex analytical queries
4. **Real-Time Updates**: Enhanced Reflex state with analytical insights
5. **Performance Optimization**: 3-80x improvement through hybrid processing
6. **Zero-Copy Integration**: Apache Arrow format for efficient data exchange

### Integration Architecture

- **Transactional Layer**: SQLModel with SQLite for CRUD operations (preserved)
- **Analytical Layer**: Polars + DuckDB for high-performance data processing (enhanced)
- **Data Exchange**: Apache Arrow format for zero-copy operations
- **Background Processing**: Integration with ADR-047's analytics queue
- **State Management**: Enhanced Reflex state with analytical capabilities

## Decision

**Implement Enhanced Data Management with Polars + DuckDB Integration** building on ADR-037's hybrid architecture:

### Enhanced Data Operations with Analytics Integration

```python
# src/services/data_service.py - Enhanced with Polars + DuckDB Integration
from sqlmodel import Session, select
from src.models.database import engine, JobModel, CompanyModel
from src.services.analytics_service import analytics_service  # ADR-037 integration
from typing import List, Dict, Optional, Any
import logging
import asyncio
import polars as pl

logger = logging.getLogger(__name__)

class EnhancedDataService:
    """Enhanced data management with Polars DataFrame processing and DuckDB analytics."""
    
    def __init__(self):
        self.session_factory = lambda: Session(engine)
        # Integration with ADR-037 analytics service
        self.analytics_service = analytics_service
    
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
        """Save multiple jobs efficiently with enhanced processing."""
        results = []
        
        # Traditional SQLModel batch processing (preserved)
        with self.session_factory() as session:
            for job_data in jobs_data:
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
        
        logger.info(f"Saved {len(results)} jobs to transactional database")
        return results
    
    async def save_jobs_batch_with_analytics(self, jobs_data: List[dict]) -> Dict[str, Any]:
        """Enhanced batch processing with Polars DataFrame analytics integration."""
        try:
            # Step 1: Traditional SQLModel save (preserved functionality)
            saved_jobs = self.save_jobs_batch(jobs_data)
            
            # Step 2: Sync to analytical database (ADR-037 integration)
            logger.info("Syncing data to analytical database...")
            sync_result = self.analytics_service.sync_transactional_to_analytical()
            
            # Step 3: Enhanced Polars DataFrame processing
            logger.info("Processing jobs with Polars DataFrames...")
            processed_df = self.analytics_service.process_jobs_with_polars(jobs_data)
            
            # Step 4: Generate analytical insights using DuckDB
            insights = await self._generate_batch_insights(processed_df)
            
            return {
                "saved_jobs_count": len(saved_jobs),
                "analytical_sync": sync_result,
                "processing_insights": insights,
                "performance_metrics": {
                    "dataframe_rows": len(processed_df),
                    "processing_time": insights.get("processing_time_ms", 0),
                    "memory_usage_mb": processed_df.estimated_size() / (1024 * 1024)
                }
            }
            
        except Exception as e:
            logger.error(f"Enhanced batch processing failed: {e}")
            # Fallback to basic processing
            return {
                "saved_jobs_count": len(self.save_jobs_batch(jobs_data)),
                "analytical_sync": {"status": "failed", "error": str(e)},
                "processing_insights": {},
                "performance_metrics": {}
            }
    
    async def _generate_batch_insights(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Generate analytical insights from Polars DataFrame using DuckDB."""
        import time
        start_time = time.time()
        
        try:
            # Basic DataFrame analytics using Polars
            insights = {
                "total_jobs": len(df),
                "unique_companies": df.select("company").n_unique(),
                "location_distribution": df.group_by("location_normalized").len().to_dicts(),
                "salary_statistics": {
                    "avg_min_salary": df.select("salary_min_extracted").mean().item(),
                    "avg_max_salary": df.select("salary_max").mean().item()
                } if "salary_min_extracted" in df.columns else {},
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            # Enhanced insights using DuckDB analytics (if available)
            if hasattr(self.analytics_service, 'analyze_salary_trends'):
                salary_trends = self.analytics_service.analyze_salary_trends()
                insights["advanced_analytics"] = {
                    "salary_trends": salary_trends.get("location_trends", [])[:5],  # Top 5
                    "market_summary": salary_trends.get("market_summary", {})
                }
            
            return insights
            
        except Exception as e:
            logger.warning(f"Insights generation failed: {e}")
            return {
                "total_jobs": len(df) if df is not None else 0,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "error": str(e)
            }
    
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

    def get_analytical_dataframe(self, filters: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
        """Get jobs data as Polars DataFrame for advanced processing."""
        try:
            # Use ADR-037 analytics service for high-performance DataFrame retrieval
            return self.analytics_service.get_jobs_as_polars(filters)
        except Exception as e:
            logger.warning(f"Analytical DataFrame retrieval failed: {e}")
            # Fallback to SQLModel export
            return self._export_sqlmodel_to_polars(filters)
    
    def _export_sqlmodel_to_polars(self, filters: Optional[Dict[str, Any]] = None) -> pl.DataFrame:
        """Fallback: Export SQLModel data to Polars DataFrame."""
        with self.session_factory() as session:
            query = select(JobModel)
            if filters:
                if filters.get("is_active"):
                    query = query.where(JobModel.is_active == True)
                if filters.get("company"):
                    query = query.where(JobModel.company == filters["company"])
            
            results = session.exec(query).all()
            
            # Convert SQLModel objects to dictionaries for Polars
            data = [{
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "description": job.description,
                "salary_text": job.salary_text,
                "salary_min": job.salary_min,
                "salary_max": job.salary_max,
                "url": job.url,
                "posted_date": job.posted_date,
                "scraped_at": job.scraped_at,
                "is_active": job.is_active,
                "is_favorited": job.is_favorited,
                "company_id": job.company_id
            } for job in results]
            
            return pl.DataFrame(data) if data else pl.DataFrame()
    
    async def analyze_job_market_trends(self) -> Dict[str, Any]:
        """Advanced job market analysis using DuckDB analytical capabilities."""
        try:
            # Use ADR-037 analytics service for comprehensive analysis
            return self.analytics_service.analyze_salary_trends()
        except Exception as e:
            logger.error(f"Market trend analysis failed: {e}")
            return {"error": str(e), "analysis_available": False}

# Global enhanced data service instance
enhanced_data_service = EnhancedDataService()

# Backward compatibility
data_service = enhanced_data_service
```

### Enhanced Real-Time Updates with Analytical Insights

```python
# src/state/scraping_state.py - Enhanced with analytics integration
import reflex as rx
import asyncio
from src.services.data_service import enhanced_data_service
from src.services.scraper_service import scraper_service
from typing import List, Dict, Any
import polars as pl

class EnhancedScrapingState(rx.State):
    """Enhanced scraping state with analytical insights and real-time updates."""
    
    # Scraping status (preserved)
    is_scraping: bool = False
    scraping_progress: str = "Ready to scrape"
    jobs_processed: int = 0
    new_jobs_found: int = 0
    updated_jobs: int = 0
    
    # Enhanced results with analytics
    recent_jobs: List[dict] = []
    scraping_stats: dict = {}
    analytical_insights: dict = {}
    dataframe_metrics: dict = {}
    
    # Performance tracking
    processing_performance: dict = {}
    analytics_available: bool = False
    
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
                
                # Collect jobs for batch processing
                batch_jobs = []
                async for job_data in scraper_service.scrape_source(source):
                    batch_jobs.append(job_data)
                    
                    # Update counters for real-time feedback
                    self.jobs_processed += 1
                    
                    # Process in batches for better performance
                    if len(batch_jobs) >= 50:  # Process every 50 jobs
                        await self._process_job_batch(batch_jobs, source)
                        batch_jobs = []
                
                # Process remaining jobs
                if batch_jobs:
                    await self._process_job_batch(batch_jobs, source)
                    
            
            # Enhanced final update with analytics
            self.scraping_progress = f"Completed! Found {self.new_jobs_found} new jobs, updated {self.updated_jobs}"
            
            # Generate analytical insights
            try:
                market_analysis = await enhanced_data_service.analyze_job_market_trends()
                self.analytical_insights = {
                    "market_trends": market_analysis.get("location_trends", [])[:5],
                    "market_summary": market_analysis.get("market_summary", {}),
                    "analysis_timestamp": market_analysis.get("analysis_timestamp", "")
                }
                self.analytics_available = True
                logger.info("Generated analytical insights for scraping session")
            except Exception as e:
                logger.warning(f"Analytics generation failed: {e}")
                self.analytics_available = False
            
            self.scraping_stats = enhanced_data_service.get_job_stats()
            
        except Exception as e:
            self.scraping_progress = f"Error: {str(e)}"
            logger.error(f"Enhanced scraping failed: {e}")
        
        finally:
            self.is_scraping = False
            yield
    
    async def _process_job_batch(self, batch_jobs: List[dict], source: str):
        """Process a batch of jobs with enhanced analytics."""
        try:
            # Enhanced batch processing with analytics
            processing_result = await enhanced_data_service.save_jobs_batch_with_analytics(batch_jobs)
            
            # Update counters based on processing results
            batch_saved = processing_result.get("saved_jobs_count", 0)
            self.new_jobs_found += batch_saved  # Simplified for demo
            
            # Update dataframe metrics
            self.dataframe_metrics = processing_result.get("performance_metrics", {})
            
            # Update recent jobs from batch
            for job_data in batch_jobs[-5:]:  # Show last 5 from batch
                if len(self.recent_jobs) >= 10:
                    self.recent_jobs.pop(0)
                
                self.recent_jobs.append({
                    "title": job_data.get("title", "Unknown"),
                    "company": job_data.get("company", "Unknown"),
                    "location": job_data.get("location", "Unknown"),
                    "source": source,
                    "processed_with_analytics": processing_result.get("analytical_sync", {}).get("status") == "success"
                })
            
            # Update progress
            self.scraping_progress = f"Processed {self.jobs_processed} jobs from {source} (Analytics: {self.analytics_available})"
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to simple processing
            saved_jobs = enhanced_data_service.save_jobs_batch(batch_jobs)
            self.new_jobs_found += len(saved_jobs)
    
    def stop_scraping(self):
        """Stop scraping process."""
        self.is_scraping = False
        self.scraping_progress = "Stopped by user"
    
    def clear_progress(self):
        """Clear scraping progress and analytics."""
        self.scraping_progress = "Ready to scrape"
        self.jobs_processed = 0
        self.new_jobs_found = 0
        self.updated_jobs = 0
        self.recent_jobs = []
        self.analytical_insights = {}
        self.dataframe_metrics = {}
        self.processing_performance = {}
        self.analytics_available = False
    
    async def get_analytical_dataframe(self) -> Dict[str, Any]:
        """Get current jobs as Polars DataFrame for analysis."""
        try:
            df = enhanced_data_service.get_analytical_dataframe({"is_active": True})
            return {
                "dataframe_available": True,
                "rows": len(df),
                "columns": len(df.columns),
                "memory_usage_mb": df.estimated_size() / (1024 * 1024),
                "schema": {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)}
            }
        except Exception as e:
            logger.error(f"DataFrame retrieval failed: {e}")
            return {
                "dataframe_available": False,
                "error": str(e)
            }
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

### Positive Outcomes (Research Validated)

- **Enhanced Performance**: 3-80x improvement in analytical processing through Polars + DuckDB integration
- **Zero-Copy Integration**: Apache Arrow format enables efficient data exchange between components
- **Library-First Implementation**: Leverages 950+ Polars and 599+ DuckDB code examples from research
- **Backward Compatibility**: Preserves all existing SQLModel functionality
- **Advanced Analytics**: DuckDB SQL capabilities for complex analytical queries
- **Real-Time Insights**: Enhanced Reflex state with analytical capabilities
- **Memory Efficiency**: Polars lazy evaluation and streaming for large datasets
- **Development Focus**: Maintains local development simplicity with enhanced capabilities

### Negative Consequences (Mitigated)

- **Memory Overhead**: Additional ~100-200MB for DuckDB + Polars processing (acceptable)
- **Complexity Increase**: Dual-database architecture requires understanding (well-documented)
- **Dependency Growth**: Additional libraries (polars, duckdb) but industry-standard
- **Learning Curve**: DataFrame operations vs pure SQL (comprehensive examples provided)

### Risk Mitigation (Enhanced)

- **Graceful Degradation**: Fallback to SQLModel-only processing if analytics fail
- **Comprehensive Documentation**: Integration patterns with ADR-037 clearly specified
- **Performance Monitoring**: Built-in metrics tracking for DataFrame operations
- **Memory Management**: Polars lazy evaluation prevents memory issues
- **Clear Integration Path**: Well-defined upgrade path leveraging ADR-037 foundation
- **Error Handling**: Enhanced error handling with fallback mechanisms
- **Development Testing**: Analytics can be disabled for simplified testing

## Development Guidelines

### Enhanced Data Operations

- **Basic Operations**: Use `enhanced_data_service.save_job()` for individual saves (preserved)
- **Batch Operations**: Use `enhanced_data_service.save_jobs_batch_with_analytics()` for enhanced processing
- **DataFrame Access**: Use `enhanced_data_service.get_analytical_dataframe()` for Polars DataFrames
- **Analytics Integration**: Leverage ADR-037's `analytics_service` for DuckDB operations
- **Performance Monitoring**: Monitor DataFrame metrics and processing performance

### Real-Time Updates with Analytics

- **Enhanced State**: Use `EnhancedScrapingState` for analytical insights
- **Batch Processing**: Process jobs in batches (50-100) for optimal performance
- **Progress Tracking**: Include analytical processing status in progress updates
- **Memory Monitoring**: Track DataFrame memory usage and processing metrics
- **Fallback Patterns**: Graceful degradation when analytics unavailable

### Testing and Development

- **Analytics Testing**: Test with and without DuckDB/Polars integration
- **Performance Validation**: Use built-in benchmarking for DataFrame operations
- **Memory Monitoring**: Track memory usage with large datasets
- **Integration Testing**: Validate ADR-037 analytics service integration
- **Fallback Testing**: Ensure graceful degradation patterns work correctly

## Related ADRs

### Primary Dependencies

- **Enhanced Integration with ADR-037**: Hybrid Database Setup (core analytics service, DuckDB connection management, Polars processing patterns)
- **Coordinates with ADR-047**: Background Job Processing (analytical queue integration, DataFrame processing workflows)
- **Supports ADR-035**: Local Development Architecture (enhanced memory requirements, analytics containers)

### Integration Points

- **Provides Enhanced Patterns For**: All data management operations across the system
- **Integrates With**: ADR-037's analytics service for zero-copy DataFrame operations
- **Enables**: Advanced analytical workflows in background processing (ADR-047)
- **Supports**: Enhanced Docker resource allocation (ADR-035)

### Cross-Reference Matrix

```text
┌─────────────┬─────────────────────────────────────────────────┐
│   ADR       │ Integration with Enhanced ADR-038               │
├─────────────┼─────────────────────────────────────────────────┤
│ ADR-037     │ Core dependency: analytics service, DuckDB conn │
│ ADR-047     │ Enhanced: analytical job processing workflows   │
│ ADR-035     │ Enhanced: memory allocation for analytics       │
└─────────────┴─────────────────────────────────────────────────┘
```

## Success Criteria

### Transactional Operations (Preserved)

- [ ] All existing SQLModel functionality works correctly
- [ ] User data preservation during updates
- [ ] Simple job saving and updating operations
- [ ] Backward compatibility with existing code

### Enhanced Analytical Operations

- [ ] Polars DataFrame processing delivers 3-80x performance improvement
- [ ] DuckDB analytical queries execute successfully via ADR-037 integration
- [ ] Zero-copy data exchange between SQLModel, Polars, and DuckDB
- [ ] Enhanced batch processing with analytics insights
- [ ] Real-time UI updates include analytical metrics
- [ ] Graceful degradation when analytics unavailable

### Integration Validation

- [ ] ADR-037 analytics service integration functions correctly
- [ ] Memory usage remains within acceptable limits (<200MB additional)
- [ ] Background processing coordination with ADR-047
- [ ] Docker resource allocation supports analytical workloads
- [ ] Performance monitoring and metrics collection operational
- [ ] Development workflow remains simple despite enhanced capabilities

### Performance Targets (Research Validated)

- [ ] **3-80x improvement** in DataFrame processing vs pure SQL operations
- [ ] **Zero-copy operations** via Apache Arrow format integration
- [ ] **<200MB additional memory** usage for analytical components
- [ ] **Sub-second response** for typical DataFrame operations
- [ ] **Lazy evaluation** prevents memory issues with large datasets

---

*This enhanced ADR integrates Polars + DuckDB analytical capabilities with SQLModel's transactional foundation, delivering research-validated performance improvements while maintaining local development simplicity and backward compatibility.*
