# ADR-036: Essential Column Configuration for Job Display

## Metadata

**Status:** Superseded by ADR-024
**Version/Date:** v2.0 / 2025-08-25

## Title

Essential Column Configuration for Personal Job Tracker

## Description

Configure only the essential columns needed for job hunting: Apply link, Salary, and Posted date. No complex formatting, just what's needed to find and apply to jobs.

## Context

**Personal Job Hunting Reality**: Only need to see the job title, company, salary, and apply link. Complex column configurations with 15+ types are massive overkill for personal use.

## Decision Drivers

- **Job Hunting Focus**: Only display what's needed to evaluate and apply to jobs
- **Simplicity**: 2-3 essential columns instead of 15+ complex configurations
- **Library-First**: Use basic `st.column_config` for just the essentials

## Decision

We will configure only **3 essential columns**: Apply link (clickable), Salary (formatted), and Posted date. Everything else is unnecessary complexity.

## Implementation

Complete column configuration in 5 lines:

```python
# Only what matters for job hunting
column_config = {
    "apply_url": st.column_config.LinkColumn("Apply"),
    "salary_min": st.column_config.NumberColumn("Salary", format="$%d"),
    "posted_at": st.column_config.DateColumn("Posted")
}

# Use with dataframe
st.dataframe(jobs_df, column_config=column_config, hide_index=True)
```

## Related Requirements

### Functional Requirements

- **FR-1:** Users must be able to click apply links directly from the table
- **FR-2:** Salary data must be formatted as currency (e.g., $75,000)
- **FR-3:** Posted dates must be human-readable

## Consequences

### Positive Outcomes

- **Job hunting optimized**: Shows exactly what's needed to evaluate jobs
- **Zero configuration overhead**: Just 3 essential column configs
- **Instant readability**: Apply links clickable, salary formatted, dates readable
- **No maintenance**: Simple configs that never need updating

### Trade-offs  

- **No fancy features**: No progress bars, images, or complex widgets (don't need them)
- **Basic formatting**: Standard currency and date formatting only (perfectly adequate)

## Dependencies

- **Python**: Standard Streamlit column config (`LinkColumn`, `NumberColumn`, `DateColumn`)

## References

- [Streamlit Column Config](https://docs.streamlit.io/library/api-reference/data/st.column_config) - Basic column configuration
- [KISS Principle](https://en.wikipedia.org/wiki/KISS_principle) - Keep It Simple, Stupid

## Changelog

- **v2.0 (2025-08-25)**: **SIMPLIFIED**: Reduced from 1,167 lines of complex column types to 3 essential columns (apply link, salary, date). Optimized for personal job hunting needs.
- **v1.0 (2025-08-25)**: ~~Initial complex column configuration~~ (over-engineered)
