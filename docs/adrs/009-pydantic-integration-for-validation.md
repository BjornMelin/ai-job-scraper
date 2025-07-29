# ADR-009: Pydantic Integration for Validation

## Title

Data Validation Using Pydantic

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

Ensure scraped data integrity (e.g., valid links) before DB insert.

## Related Requirements

- Skip invalid on insert.
- Models for Job/Company.

## Alternatives

- Manual checks: Error-prone.
- No validation: Dirty data.

## Decision

Pydantic models (JobPydantic with Field(pattern for link)); Try/except skip invalid in update_db.

## Related Decisions

- ADR-003 (Validates before persist).

## Design

- class JobPydantic(BaseModel): link=Field(pattern=r"^https?://").
- Integration: try JobPydantic(**dict) except skip/log.
- Implementation Notes**: Optional fields with defaults.
- **Testing**: def test_job_validation(): Valid/invalid mocks, assert skips.

## Consequences

- Clean DB (validated data).
