# ADR-035: Simple Manual Refresh for Personal Job Tracker

## Metadata

**Status:** Superseded by ADR-024
**Version/Date:** v2.0 / 2025-08-25

## Title

Simple Manual Refresh Button Implementation

## Description

Implement a simple manual refresh button using Streamlit's cache clearing mechanism. Perfect for personal use where jobs are checked 1-2 times daily.

## Context

**Personal Use Reality**: A personal job tracker is accessed 1-2 times per day, not continuously. Auto-refresh would drain battery and add unnecessary complexity.

## Decision Drivers

- **Personal Use Optimization**: Manual refresh suits 1-2x daily usage pattern perfectly
- **Battery Conservation**: No auto-refresh means longer laptop battery life  
- **Simplicity**: Single refresh button is easier to understand and maintain
- **Library-First**: Use simple `st.cache_data.clear()` + `st.rerun()` built-in functions

## Decision

We will use a **Simple Manual Refresh Button** that clears cached data and reruns the app. No fragments, no auto-refresh, no complexity.

## Implementation

Complete implementation in 3 lines:

```python
# Simple manual refresh - that's it!
if st.button("🔄 Refresh Jobs"):
    st.cache_data.clear()
    st.rerun()
```

## Consequences

### Positive Outcomes

- **Perfect for personal use**: Exactly matches 1-2x daily usage pattern
- **Zero complexity**: No fragments, timers, or background processes
- **Battery friendly**: No auto-refresh draining laptop battery
- **Instant to implement**: 3 lines of code total

### Trade-offs

- **No real-time updates**: Users must manually refresh (perfect for personal use)
- **No automatic progress**: Manual refresh needed after scraping (acceptable)

## Dependencies

- **Python**: Only standard Streamlit functions (`st.button`, `st.cache_data.clear`, `st.rerun`)

## References

- [Streamlit Caching](https://docs.streamlit.io/library/api-reference/performance/st.cache_data) - Cache clearing documentation
- [KISS Principle](https://en.wikipedia.org/wiki/KISS_principle) - Keep It Simple, Stupid

## Changelog

- **v2.0 (2025-08-25)**: **SIMPLIFIED**: Replaced 1,042 lines of fragment complexity with 3-line manual refresh. Optimized for personal use case where 1-2x daily access is typical.
- **v1.0 (2025-08-25)**: ~~Initial complex fragments implementation~~ (over-engineered)
