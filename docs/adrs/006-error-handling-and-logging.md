# ADR-006: Error Handling and Logging

## Title

Comprehensive Error Management and Logging Strategy

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

App must handle errors in scraping/DB/UI (e.g., network fails, invalid data) with retries/logs/user feedback.

## Related Requirements

- Retries for scraping, try/except in all ops.
- Logging for debugging.

## Alternatives

- Minimal handling: Crashes.
- Sentry: Overkill for local.

## Decision

Use tenacity for retries, try/except with logger.error in key funcs, st.error in UI. logging.basicConfig(level=INFO).

## Related Decisions

- ADR-001 (Retries in scraping).
- ADR-003 (DB rollbacks).

## Design

- **Logging**: logger = logging.getLogger(**name**); logger.error(f"Failed: {e}").
- **Handling**: Try: ... except e: log, st.error, rollback if DB.
- **Integration**: In main funcs like update_db/scrape_all/display_jobs.
- **Implementation Notes**: Lazy formatting (f-strings).
- **Testing**: Simulate errors, assert log/st.error called, no crash.

## Consequences

- Robust (graceful failures).
- Debuggable (logs).

- Verbose if misconfigured (adjust level).
