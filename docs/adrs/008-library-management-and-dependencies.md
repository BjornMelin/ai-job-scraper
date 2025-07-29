# ADR-008: Library Management and Dependencies

## Title

Dependency Selection and Version Management

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

Use latest stable versions for features/security (e.g., pandas 2.3.1 over 2.2.3).

## Related Requirements

- Pin in pyproject.toml with uv.
- Revert Polars to Pandas for simplicity.

## Alternatives

- Older versions: Miss fixes.
- No pinning: Breaks.

## Decision

Pin latest (e.g., streamlit==1.47.1, pandas==2.3.1); Use Pandas only (no Polars).

## Related Decisions

- ADR-004 (Deps for UI).

## Design

- pyproject.toml: dependencies = ["crawl4ai==0.7.2", ...].
- Integration: uv sync.
- Implementation Notes**: Check compat on updates.
- **Testing**: Run with pinned, assert no errors.

## Consequences

- Up-to-date (best capabilities).
- Stable (pinned).
