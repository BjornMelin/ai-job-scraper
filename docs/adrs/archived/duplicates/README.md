# Archived Duplicate ADRs

This directory contains ADRs that were identified as duplicates during the comprehensive conflict resolution analysis performed on 2025-08-22.

## Archived Duplicates

### ADR-023-background-job-processing-DUPLICATE.md
- **Status**: Duplicate of ADR-012
- **Reason**: This ADR covered the same threading-based background task approach as ADR-012
- **Resolution**: ADR-012 is the definitive background task management solution (84.8% decision framework score)
- **Archived**: 2025-08-22
- **Superseded by**: ADR-012 (Background Task Management Using Standard Threading for Streamlit)

## Conflict Resolution Summary

The comprehensive analysis resolved conflicts across:
- Background task management (Threading vs RQ/Redis vs AsyncIO)
- Framework alignment (Streamlit vs Reflex references)  
- Caching strategies (st.cache_data vs Redis)
- Analytics approaches (Pandas vs Polars+DuckDB)

All conflicts have been resolved with data-driven decisions using 2024 benchmarks and decision framework scoring.

## Related Archives

- **performance-cluster/**: Contains archived competing approaches (RQ/Redis, Polars+DuckDB)
- **ui-cluster/reflex/**: Contains archived Reflex framework references