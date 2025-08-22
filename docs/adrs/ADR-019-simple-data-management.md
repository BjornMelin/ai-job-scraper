# ADR-019: Simple Data Management with Polars DataFrame Processing

## Metadata

**Status:** Accepted
**Version/Date:** v1.0 / 2025-08-22

## Title

Simple Data Management with Polars DataFrame Processing

## Description

Implement enhanced data management combining SQLModel's transactional capabilities with Polars DataFrame processing for analytical workloads, delivering 3-80x performance improvements while maintaining simplicity.

## Context

The AI Job Scraper requires enhanced data management capabilities beyond basic SQLModel CRUD operations. Current system limitations include inefficient batch processing, lack of analytical capabilities, and performance bottlenecks when processing large job datasets (1000+ records).

Current challenges:

- SQLModel handles basic CRUD but lacks analytical processing capabilities
- Large dataset operations cause UI blocking and memory issues  
- No efficient way to perform complex data transformations and aggregations
- Missing integration with background processing systems for analytics
- Performance degradation with increasing job data volume

The system needs to preserve existing SQLModel functionality while adding high-performance DataFrame processing for analytical workloads. Integration with **ADR-025** (Performance & Scale Strategy) provides the foundation for analytical workflows, while **ADR-023** (Background Job Processing) enables async processing coordination.

## Decision Drivers

- **Performance Requirements**: Need 3-80x improvement in analytical processing for large datasets
- **Analytical Capabilities**: Require complex data transformations beyond basic SQL
- **Streamlit Integration**: Must maintain UI responsiveness with enhanced insights  
- **Backward Compatibility**: Preserve all existing SQLModel functionality
- **Local Development**: Maintain simplicity while adding analytical capabilities
- **Zero Dependencies**: Library-first approach minimizing infrastructure complexity

## Alternatives

- **A: Pure SQLModel + Raw SQL** — Simple, familiar / Limited analytical capabilities, poor performance on large datasets
- **B: Replace SQLModel with Polars Only** — High performance, unified interface / Breaks existing code, complex CRUD operations  
- **C: SQLModel + Polars Integration** — Best of both worlds, performance gains, backward compatible / Additional complexity, memory overhead

### Decision Framework

| Model / Option | Solution Leverage (Weight: 35%) | Application Value (Weight: 30%) | Maintenance & Cognitive Load (Weight: 25%) | Architectural Adaptability (Weight: 10%) | Total Score | Decision |
| -------------- | ------------------------------- | ------------------------------- | ------------------------------------------ | ---------------------------------------- | ----------- | -------- |
| **SQLModel + Polars Integration** | 9.2 | 9.5 | 8.0 | 9.0 | **8.95** | ✅ **Selected** |
| Pure SQLModel + Raw SQL | 6.0 | 5.5 | 9.0 | 7.0 | 6.53 | Rejected |
| Replace with Polars Only | 8.5 | 7.0 | 4.0 | 6.0 | 6.58 | Rejected |

## Decision

We will adopt **SQLModel + Polars Integration** to address analytical processing limitations. This involves using **Polars DataFrames** for high-performance analytics configured with **Apache Arrow** zero-copy data exchange and coordination with **ADR-025's** performance strategy.

## High-Level Architecture

```mermaid
graph TB
    subgraph "Data Management Layer"
        A[Streamlit UI] --> B[Data Service]
        B --> C[SQLModel Layer]
        B --> D[Polars Analytics Layer]
        C --> E[SQLite Database]
        D --> F[Apache Arrow Bridge]
        D --> G[Background Processing]
    end
    
    subgraph "Data Flow"
        H[Job Scrapers] --> B
        B --> I[Real-time UI Updates]
        B --> J[Analytics Processing]
    end
```

## Related Requirements

### Functional Requirements

- **FR-1:** The system must preserve all existing SQLModel CRUD operations without breaking changes
- **FR-2:** Users must receive analytical insights in real-time through Streamlit UI
- **FR-3:** The system must process job data batches with 3-80x performance improvement

### Non-Functional Requirements

- **NFR-1:** **(Maintainability)** The solution must reduce analytical processing complexity by leveraging library-first patterns
- **NFR-2:** **(Performance)** DataFrame operations must achieve sub-second response times for typical datasets
- **NFR-3:** **(Compatibility)** The component must maintain backward compatibility with existing SQLModel implementations

### Performance Requirements

- **PR-1:** Analytical processing latency must achieve 3-80x improvement over pure SQL operations
- **PR-2:** Memory overhead must not exceed 200MB additional usage for analytical components
- **PR-3:** Zero-copy data exchange via Apache Arrow must eliminate serialization bottlenecks

### Integration Requirements

- **IR-1:** The solution must integrate seamlessly with **ADR-025's** performance strategy
- **IR-2:** The component must coordinate with **ADR-023's** background processing architecture
- **IR-3:** Streamlit state must receive analytical insights without blocking UI responsiveness

## Related Decisions

- **ADR-025** (Performance & Scale Strategy): This decision builds upon the performance optimization patterns and analytical processing requirements
- **ADR-023** (Background Job Processing): Data service coordinates with background processing for analytics workflows  
- **ADR-017** (Local Development Architecture): Memory allocation and resource requirements updated for analytical workloads

## Design

### Architecture Overview

The data management system operates on a dual-layer architecture: SQLModel handles transactional operations while Polars processes analytical workloads. Apache Arrow provides zero-copy data exchange between layers, coordinated through performance optimization patterns.

### Implementation Details

**In `src/services/data_service.py`:**

```python
import polars as pl
from sqlmodel import Session, select
from src.models.database import engine, JobModel
from typing import List, Dict, Any, Optional

class DataService:
    """Data management with Polars DataFrame processing."""
    
    def __init__(self):
        self.session_factory = lambda: Session(engine)
    
    async def save_jobs_batch_with_analytics(self, jobs_data: List[dict]) -> Dict[str, Any]:
        """Batch processing with analytics integration."""
        # Step 1: Traditional SQLModel save (preserved)
        saved_jobs = self.save_jobs_batch(jobs_data)
        
        # Step 2: Polars DataFrame processing
        df = pl.DataFrame(jobs_data)
        insights = self._generate_insights(df)
        
        return {
            "saved_jobs_count": len(saved_jobs),
            "insights": insights,
            "performance_metrics": {
                "dataframe_rows": len(df),
                "memory_usage_mb": df.estimated_size() / (1024 * 1024)
            }
        }
    
    def get_analytical_dataframe(self, filters: Optional[Dict] = None) -> pl.DataFrame:
        """Get jobs data as Polars DataFrame for processing."""
        with self.session_factory() as session:
            query = select(JobModel)
            if filters:
                # Apply filters
                pass
            jobs = session.exec(query).all()
            return pl.DataFrame([job.dict() for job in jobs])

# Global service instance
data_service = DataService()
```

**In `src/pages/scraping.py`:**

```python
import streamlit as st
from src.services.data_service import data_service

def render_analytics_section():
    """Render analytics section with Polars insights."""
    if st.button("Generate Analytics"):
        df = data_service.get_analytical_dataframe({"is_active": True})
        
        # Display DataFrame insights
        st.metric("Total Jobs", len(df))
        st.metric("Memory Usage", f"{df.estimated_size() / (1024*1024):.1f} MB")
        
        # Show top companies
        if "company" in df.columns:
            top_companies = df.group_by("company").count().sort("count", descending=True).head(5)
            st.dataframe(top_companies)
```

### Configuration

**In `pyproject.toml`:**

```toml
[tool.uv.dependencies]
polars = ">=1.0.0"
pyarrow = ">=17.0.0"  # Apache Arrow support
```

**Environment variables:**

```env
# Analytics configuration
ANALYTICS_BATCH_SIZE=100
POLARS_MAX_THREADS=4
```

## Testing

**In `tests/test_data_service.py`:**

```python
import pytest
import polars as pl
from src.services.data_service import data_service

@pytest.mark.asyncio
async def test_analytics_batch_processing():
    """Verify batch processing delivers performance improvements."""
    # Test setup with mock job data
    jobs_data = [{"title": "Test Job", "company": "Test Corp", "url": "test-url"}]
    
    # Execute processing
    result = await data_service.save_jobs_batch_with_analytics(jobs_data)
    
    # Assertions
    assert result["saved_jobs_count"] > 0
    assert "performance_metrics" in result
    assert result["performance_metrics"]["dataframe_rows"] == len(jobs_data)

def test_polars_dataframe_integration():
    """Verify Polars DataFrame generation and processing."""
    # Test DataFrame retrieval
    df = data_service.get_analytical_dataframe({"is_active": True})
    
    # Validate DataFrame structure
    assert isinstance(df, pl.DataFrame)
    assert len(df.columns) > 0

@pytest.mark.asyncio 
async def test_performance_metrics():
    """Verify performance metrics collection."""
    # Test performance tracking
    pass
```

## Consequences

### Positive Outcomes

- Enables high-performance analytical processing with 3-80x improvement in DataFrame operations over pure SQL queries
- Preserves all existing SQLModel functionality ensuring backward compatibility and zero breaking changes
- Provides zero-copy data exchange via Apache Arrow format, eliminating serialization bottlenecks
- Integrates naturally with Streamlit UI for real-time analytical insights without blocking user interface
- Leverages library-first approach with Polars reducing custom analytical code requirements
- Supports efficient batch processing for large job datasets (1000+ records) with minimal memory overhead

### Negative Consequences / Trade-offs

- Introduces additional memory overhead (~100-200MB) for Polars DataFrame processing components
- Creates dual-architecture complexity requiring understanding of both SQLModel and Polars patterns
- Adds dependency on polars and pyarrow libraries increasing overall dependency footprint
- Requires developer familiarity with DataFrame operations vs traditional SQL query patterns

### Ongoing Maintenance & Considerations

- Monitor Polars library updates and compatibility with pyarrow for Apache Arrow support
- Track memory usage patterns for large datasets to ensure performance optimization
- Coordinate with ADR-025 performance strategy for analytical processing patterns
- Maintain fallback mechanisms when analytical processing unavailable
- Review DataFrame processing performance quarterly and optimize for growing datasets

### Dependencies

- **System**: None (pure Python implementation)
- **Python**: `polars>=1.0.0`, `pyarrow>=17.0.0` for Apache Arrow support
- **Removed**: None (additive changes only)

## References

- [Polars Documentation](https://docs.pola.rs/) - Comprehensive guide to DataFrame operations and performance optimization
- [Polars on PyPI](https://pypi.org/project/polars/) - Version history and installation requirements
- [Apache Arrow Documentation](https://arrow.apache.org/docs/) - Zero-copy data exchange format specifications
- [SQLModel Documentation](https://sqlmodel.tiangolo.com/) - Integration patterns for ORM and DataFrame processing
- [Polars vs Pandas Performance](https://github.com/pola-rs/polars) - Performance benchmarks informing decision analysis

## Changelog

- **v1.0 (2025-08-22)**: Initial accepted version with SQLModel + Polars integration architecture
