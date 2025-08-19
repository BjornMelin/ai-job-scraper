# ADR-041: Local Development Performance

## Title

Simple Performance Considerations for Local Development

## Version/Date

2.0 / August 19, 2025

## Status

**Accepted** - Focused on local development needs

## Description

Basic performance considerations for local development environment. Provides simple optimization patterns that improve development experience without production-level complexity. Focuses on reasonable response times, efficient development workflows, and simple resource management.

## Context

### Local Development Performance Needs

This performance approach focuses on:

1. **Development Experience**: Reasonable response times for development workflow
2. **Simple Optimization**: Basic patterns without complex infrastructure
3. **Resource Management**: Efficient use of development machine resources
4. **Debugging Performance**: Easy identification of performance issues
5. **Scalable Patterns**: Foundations that can scale to production

### Development Scope

- **Data Scale**: Handle 100-1000 jobs efficiently for development
- **Response Times**: Reasonable (<1 second) for development operations
- **Memory Usage**: Efficient use of development machine memory
- **UI Responsiveness**: Smooth interaction during development testing
- **Database Performance**: Basic SQLite optimization for development

## Decision

**Implement Simple Performance Patterns** for local development:

### Basic Database Optimization

```python
# src/models/optimized_database.py
from sqlmodel import SQLModel, Field, create_engine, Session, Index
from typing import Optional
import sqlite3
from pathlib import Path

# Optimized SQLite configuration for development
DATABASE_URL = "sqlite:///./data/jobs.db"

def create_optimized_engine():
    """Create optimized SQLite engine for development."""
    engine = create_engine(
        DATABASE_URL,
        echo=False,  # Set True for SQL debugging
        connect_args={
            "check_same_thread": False,
            "timeout": 20,
        }
    )
    return engine

def optimize_sqlite_connection(connection):
    """Apply SQLite optimizations for development."""
    # Enable WAL mode for better concurrency
    connection.execute("PRAGMA journal_mode=WAL")
    
    # Optimize for development performance
    connection.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
    connection.execute("PRAGMA cache_size=10000")     # 10MB cache
    connection.execute("PRAGMA temp_store=MEMORY")    # Memory temp storage
    connection.execute("PRAGMA mmap_size=268435456")  # 256MB memory mapping
    
    connection.commit()

class OptimizedJobModel(SQLModel, table=True):
    """Job model with basic indexing for development."""
    __tablename__ = "jobs"
    
    # Primary key
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Indexed fields for common queries
    title: str = Field(index=True)
    company: str = Field(index=True)
    location: Optional[str] = Field(default=None, index=True)
    
    # Content fields
    description: Optional[str] = None
    salary_text: Optional[str] = None
    url: str = Field(unique=True)
    
    # Status fields with indexes for filtering
    is_active: bool = Field(default=True, index=True)
    is_favorited: bool = Field(default=False, index=True)
    
    # Date fields for sorting/filtering
    posted_date: Optional[str] = Field(default=None, index=True)
    scraped_at: str = Field(index=True)
    
    # Additional indexes for performance
    __table_args__ = (
        Index("idx_company_location", "company", "location"),
        Index("idx_active_date", "is_active", "scraped_at"),
        Index("idx_title_company", "title", "company"),
    )

def init_optimized_database():
    """Initialize database with optimizations."""
    engine = create_optimized_engine()
    
    # Create tables
    SQLModel.metadata.create_all(engine)
    
    # Apply SQLite optimizations
    with engine.connect() as connection:
        optimize_sqlite_connection(connection)
    
    return engine
```

### Efficient Data Operations

```python
# src/services/optimized_data_service.py
from sqlmodel import Session, select, func
from src.models.optimized_database import OptimizedJobModel, init_optimized_database
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class OptimizedDataService:
    """Data service with basic performance optimizations."""
    
    def __init__(self):
        self.engine = init_optimized_database()
    
    def get_jobs_paginated(self, page: int = 1, page_size: int = 50, filters: Dict = None) -> Tuple[List[OptimizedJobModel], int]:
        """Get jobs with efficient pagination."""
        with Session(self.engine) as session:
            # Build query with filters
            query = select(OptimizedJobModel).where(OptimizedJobModel.is_active == True)
            
            # Apply filters efficiently
            if filters:
                if filters.get("company"):
                    query = query.where(OptimizedJobModel.company.contains(filters["company"]))
                if filters.get("location"):
                    query = query.where(OptimizedJobModel.location.contains(filters["location"]))
                if filters.get("favorites_only"):
                    query = query.where(OptimizedJobModel.is_favorited == True)
            
            # Get total count (before pagination)
            count_query = select(func.count()).select_from(query.subquery())
            total_count = session.exec(count_query).one()
            
            # Apply pagination
            offset = (page - 1) * page_size
            query = query.offset(offset).limit(page_size)
            
            # Order by most recent
            query = query.order_by(OptimizedJobModel.scraped_at.desc())
            
            jobs = list(session.exec(query))
            
            return jobs, total_count
    
    def search_jobs_efficiently(self, search_term: str, limit: int = 100) -> List[OptimizedJobModel]:
        """Efficient job search for development."""
        with Session(self.engine) as session:
            # Simple text search with indexes
            query = select(OptimizedJobModel).where(
                OptimizedJobModel.is_active == True,
                (OptimizedJobModel.title.contains(search_term) |
                 OptimizedJobModel.company.contains(search_term) |
                 OptimizedJobModel.location.contains(search_term))
            ).limit(limit)
            
            return list(session.exec(query))
    
    def bulk_update_jobs(self, jobs_data: List[Dict]) -> int:
        """Bulk update jobs efficiently."""
        updated_count = 0
        
        with Session(self.engine) as session:
            # Process in batches for memory efficiency
            batch_size = 50
            
            for i in range(0, len(jobs_data), batch_size):
                batch = jobs_data[i:i + batch_size]
                
                for job_data in batch:
                    existing = session.exec(
                        select(OptimizedJobModel).where(
                            OptimizedJobModel.url == job_data["url"]
                        )
                    ).first()
                    
                    if existing:
                        # Update existing job
                        for key, value in job_data.items():
                            if key not in ["is_favorited", "notes"] and hasattr(existing, key):
                                setattr(existing, key, value)
                        session.add(existing)
                        updated_count += 1
                    else:
                        # Create new job
                        new_job = OptimizedJobModel(**job_data)
                        session.add(new_job)
                        updated_count += 1
                
                # Commit batch
                session.commit()
        
        return updated_count
    
    def get_statistics_cached(self) -> Dict:
        """Get statistics with simple caching."""
        # Simple in-memory caching for development
        if hasattr(self, '_stats_cache'):
            return self._stats_cache
        
        with Session(self.engine) as session:
            # Efficient aggregate queries
            total_jobs = session.exec(
                select(func.count()).select_from(OptimizedJobModel)
            ).one()
            
            active_jobs = session.exec(
                select(func.count()).where(OptimizedJobModel.is_active == True)
            ).one()
            
            favorited_jobs = session.exec(
                select(func.count()).where(OptimizedJobModel.is_favorited == True)
            ).one()
            
            # Count unique companies
            unique_companies = session.exec(
                select(func.count(func.distinct(OptimizedJobModel.company)))
            ).one()
            
            stats = {
                "total_jobs": total_jobs,
                "active_jobs": active_jobs,
                "favorited_jobs": favorited_jobs,
                "unique_companies": unique_companies
            }
            
            # Cache for 30 seconds in development
            self._stats_cache = stats
            return stats
    
    def clear_stats_cache(self):
        """Clear statistics cache."""
        if hasattr(self, '_stats_cache'):
            delattr(self, '_stats_cache')

# Global optimized data service
optimized_data_service = OptimizedDataService()
```

### Efficient UI Patterns

```python
# src/components/optimized_components.py
import reflex as rx
from typing import List, Dict

def paginated_job_list(jobs: List[Dict], current_page: int, total_pages: int, on_page_change) -> rx.Component:
    """Efficient paginated job list."""
    return rx.vstack(
        # Job count and pagination info
        rx.hstack(
            rx.text(f"Showing {len(jobs)} jobs", color="gray.600"),
            rx.spacer(),
            rx.text(f"Page {current_page} of {total_pages}", color="gray.600"),
            width="100%"
        ),
        
        # Job list (limited to current page)
        rx.vstack(
            rx.foreach(
                jobs,
                lambda job: rx.card(
                    rx.vstack(
                        rx.heading(job["title"], size="sm"),
                        rx.text(job["company"], color="blue.500"),
                        rx.text(job.get("location", ""), color="gray.600"),
                        spacing="1"
                    ),
                    padding="3",
                    margin="1"
                )
            ),
            spacing="2",
            width="100%"
        ),
        
        # Pagination controls
        rx.hstack(
            rx.button(
                "Previous",
                on_click=lambda: on_page_change(max(1, current_page - 1)),
                disabled=current_page <= 1
            ),
            rx.text(f"{current_page}", font_weight="bold"),
            rx.button(
                "Next",
                on_click=lambda: on_page_change(min(total_pages, current_page + 1)),
                disabled=current_page >= total_pages
            ),
            spacing="2",
            justify="center"
        ),
        
        spacing="4",
        width="100%"
    )

def efficient_search_component(on_search, is_searching: bool = False) -> rx.Component:
    """Search component with debouncing for performance."""
    return rx.hstack(
        rx.input(
            placeholder="Search jobs...",
            width="100%",
            on_change=on_search  # Could add debouncing in state
        ),
        rx.cond(
            is_searching,
            rx.spinner(size="sm"),
            rx.icon("search")
        ),
        spacing="2",
        width="100%"
    )

def performance_stats_display(stats: Dict) -> rx.Component:
    """Display performance statistics for development."""
    return rx.card(
        rx.vstack(
            rx.heading("Performance Stats", size="sm"),
            rx.hstack(
                rx.stat(
                    rx.stat_label("Total Jobs"),
                    rx.stat_number(stats.get("total_jobs", 0))
                ),
                rx.stat(
                    rx.stat_label("Active Jobs"),
                    rx.stat_number(stats.get("active_jobs", 0))
                ),
                rx.stat(
                    rx.stat_label("Companies"),
                    rx.stat_number(stats.get("unique_companies", 0))
                ),
                spacing="4"
            ),
            spacing="3"
        ),
        padding="3"
    )
```

### Development Performance Monitoring

```python
# src/utils/performance_monitor.py
import time
import logging
from functools import wraps
from typing import Callable, Any
import psutil
import os

logger = logging.getLogger(__name__)

def performance_timer(func: Callable) -> Callable:
    """Simple performance timing decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > 0.5:  # Log slow operations
                logger.warning(f"{func.__name__} took {execution_time:.2f}s")
            else:
                logger.debug(f"{func.__name__} took {execution_time:.3f}s")
            
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper

def memory_monitor(threshold_mb: int = 100):
    """Simple memory monitoring for development."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = memory_after - memory_before
            
            if memory_used > threshold_mb:
                logger.warning(f"{func.__name__} used {memory_used:.1f}MB memory")
            
            return result
        return wrapper
    return decorator

class SimplePerformanceProfiler:
    """Simple performance profiler for development."""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.timings[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing and log result."""
        if operation in self.timings:
            duration = time.time() - self.timings[operation]
            logger.info(f"Operation '{operation}' took {duration:.3f}s")
            del self.timings[operation]
            return duration
        return None
    
    def log_memory_usage(self, operation: str):
        """Log current memory usage."""
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage[operation] = memory_mb
        logger.debug(f"Memory usage during '{operation}': {memory_mb:.1f}MB")
        return memory_mb

# Global profiler instance
profiler = SimplePerformanceProfiler()
```

## Consequences

### Positive Outcomes

- **Reasonable Performance**: Good response times for development workflow
- **Simple Optimization**: Basic patterns easy to understand and debug
- **Development Focus**: Optimized for development machine resources
- **Monitoring Tools**: Simple performance monitoring for development
- **Scalable Foundation**: Patterns that can grow into production optimization

### Negative Consequences

- **Development Scope**: Optimized for development scale, not production
- **Limited Complexity**: Basic patterns without advanced optimization
- **Simple Caching**: Basic in-memory caching without sophisticated strategies
- **SQLite Limitations**: Database performance limited by SQLite capabilities

### Risk Mitigation

- **Clear Documentation**: Simple optimization patterns with examples
- **Performance Monitoring**: Basic tools to identify performance issues
- **Upgrade Path**: Clear migration to production performance optimization
- **Resource Awareness**: Monitor resource usage during development

## Development Guidelines

### Database Performance

- Use indexed fields for common queries
- Implement efficient pagination patterns
- Apply basic SQLite optimizations
- Monitor query performance with timing decorators

### UI Performance

- Implement pagination for large data sets
- Use efficient state update patterns
- Monitor component render performance
- Implement simple caching for expensive operations

### Memory Management

- Process data in batches for large operations
- Clear caches when appropriate
- Monitor memory usage with development tools
- Use efficient data structures for development

## Related ADRs

- **Supports ADR-035**: Local Development Architecture (performance component)
- **Uses ADR-037**: Local Database Setup (database optimization)
- **Uses ADR-040**: Reflex Local Development (UI performance)
- **Replaces Archived ADR-041**: Performance Optimization Strategy (production-focused)

## Success Criteria

- [ ] Database queries complete in reasonable time (<1 second for development)
- [ ] UI remains responsive during typical development operations
- [ ] Memory usage stays within reasonable limits for development machine
- [ ] Performance monitoring provides useful feedback during development
- [ ] Pagination and search work efficiently for development data sets
- [ ] Simple caching improves development experience

---

*This ADR provides simple, practical performance patterns for local development without production-level optimization complexity.*