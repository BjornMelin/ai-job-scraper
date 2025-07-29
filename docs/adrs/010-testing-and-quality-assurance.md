# ADR-010: Testing and Quality Assurance

## Title

Testing Strategy for Reliability and Coverage

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

Ensure components work (unit for funcs, integration for scrape/UI/DB).

## Related Requirements

- Pytest for async/sync tests.
- Cover edge cases (failures/empties).

## Alternatives

- No tests: Risky.
- Full e2e: Slow.

## Decision

Pytest with marks (asyncio), mocks for external (e.g., httpx). Cover scraper/UI/DB.

## Related Decisions

- All (Tests per ADR).

## Design

- tests/test_scraper.py: test_is_relevant, test_validate_link.
- Integration: Mock DB, assert end-to-end.
- Implementation Notes**: Run uv run pytest.
- **Testing**: Coverage >80%.

## Consequences

- Reliable (catches bugs).
