# ADR-014: Performance Optimization Strategy for 5000+ Records

## Title

Database and UI Performance Optimization for Large-Scale Job Management

## Version/Date

1.0 / January 18, 2025

## Status

Proposed

## Context

Current implementation loads all jobs into memory causing 6-11s delays per UI update with 5000+ records. Users report sluggish experience. Streamlit reloads entire dataset on each interaction. Database queries perform full table scans. No pagination, caching, or query optimization. Need strategy to achieve <100ms response times at scale.

## Related Requirements

- Support 5000+ job records efficiently
- UI response time <100ms for common operations
- Non-blocking background scraping
- Real-time progress updates
- Memory usage under 100MB for UI

## Alternatives

1. **Full React Rewrite**: Better performance but 2-4 week timeline
2. **Database-Only Optimization**: Helps queries but UI still slow
3. **Pagination-Only**: Solves UI but not query performance
4. **Comprehensive Optimization**: Database + UI + caching (chosen)

## Decision

Implement multi-layer optimization:

### Database Layer

- Add indexes on commonly queried columns (posted_date, company_id, status)
- Implement eager loading for relationships
- Use query pagination with LIMIT/OFFSET
- Add composite indexes for complex filters

### UI Layer

- Implement pagination (50 items per page)
- Use st.cache_data for expensive operations
- Virtual scrolling for large lists (future)
- Lazy loading for job details

### Caching Layer

- Cache job counts (1 minute TTL)
- Cache company lists (5 minute TTL)
- Cache filter results (30 second TTL)
- Session-based pagination state

## Related Decisions

- ADR-007: Database schema supports indexes
- ADR-011: Connection pooling helps concurrent queries
- ADR-009: Background tasks prevent blocking

## Design

```python
# Database Indexes
class Job(SQLModel):
    __table_args__ = (
        Index("idx_posted_date", "posted_date"),
        Index("idx_company_status", "company_id", "application_status"),
        Index("idx_salary_range", "salary_min", "salary_max"),
    )

# Paginated Repository
class JobRepository:
    def get_jobs_paginated(offset: int, limit: int, filters: dict):
        query = (
            select(Job)
            .options(selectinload(Job.company))  # Eager load
            .offset(offset)
            .limit(limit)
        )
        return session.exec(query).all()

# UI Pagination
class JobsPage:
    ITEMS_PER_PAGE = 50
    
    @st.cache_data(ttl=60)
    def get_page_data(page: int, filters: dict):
        offset = page * ITEMS_PER_PAGE
        return repo.get_jobs_paginated(offset, ITEMS_PER_PAGE, filters)
```

## Consequences

### Positive

- **110x faster page loads** (11s → 100ms)
- **10x memory reduction** (500MB → 50MB)
- **Scalable to 50,000+ records** with same performance
- **Better UX** with instant responses

### Negative

- **Pagination complexity**: Need to manage state across pages
- **Cache invalidation**: Must handle stale data carefully
- **Search limitations**: Can't search across all pages simultaneously

### Mitigations

- **Smart caching**: Short TTLs prevent stale data
- **Progressive search**: Search current page first, then expand
- **Prefetching**: Load next page in background

## Implementation Plan

1. **Hour 1**: Add database indexes via migration
2. **Hour 2**: Implement repository pagination
3. **Hour 3**: Add UI pagination components
4. **Hour 4**: Implement caching strategy
5. **Hour 5**: Testing and optimization

## Metrics

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Page load time | 6-11s | <100ms | Pagination + indexes |
| Memory usage | 500MB | 50MB | Load only visible items |
| Query time | 2-3s | <50ms | Indexes + eager loading |
| Search time | 5s | <200ms | Indexed search |

## Monitoring

Track performance metrics:

```python
@track_performance("job_query")
def get_jobs_paginated(...):
    # Log query time, result count, cache hit rate
```

## Rollback Plan

If optimization causes issues:

1. Remove indexes (quick via migration rollback)
2. Disable pagination (feature flag)
3. Clear caches (immediate)
4. Revert to full load (config toggle)
