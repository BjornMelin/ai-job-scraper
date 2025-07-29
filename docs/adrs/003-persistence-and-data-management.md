# ADR-003: Persistence and Data Management

## Title

Database Selection, Schema, and Update Logic

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

Need persistent storage for jobs and companies, with schema supporting user features (favorite/status/notes/location/posted_date), update logic (add/update/delete based on link/hash), and validation (Pydantic for data integrity). SQLite is ideal for local, lightweight use.

## Related Requirements

- Jobs table with extended fields; Companies table for dynamic management.
- Update preserves user edits; Delete missing jobs.
- Validation skips invalid entries.

## Alternatives

- JSON/CSV files: No efficient queries or concurrency support.
- PostgreSQL: Overkill for local app, requires server.

## Decision

Use SQLite with SQLAlchemy v2.0.42 ORM for schema (JobSQL with id/company/title/etc./favorite/status/notes; CompanySQL with name/url/active). Implement update logic comparing link/hash, preserving user fields. Use Pydantic v2.11.7 for validation (try/except skip invalid).

## Related Decisions

- ADR-001 (Data from scraping fed here).
- ADR-005 (UI interacts with stored data).

## Design

- **Schema**: class JobSQL(Base): ... favorite=Boolean(default=False), status=String(default='New'), notes=Text(default=''); class CompanySQL(Base): name=String(unique=True), url=String, active=Boolean(default=True).
- **Update Logic**: existing = {j.link: j for j in query.all()}; For new: if link in existing and hash !=: Update non-user fields; else add (with defaults); Delete absent links.
- **Validation**: In update_db: try JobPydantic(**dict) except log/skip.
- **Integration**: engine = create_engine('sqlite:///jobs.db'); Session for queries. Scrape from active companies.
- **Implementation Notes**: hash = hashlib.md5(desc.encode()).hexdigest(). Preserve favorite/status/notes always.
- **Testing**: def test_update_db(): Mock new/existing, assert adds/updates/deletes correctly, user fields preserved, invalid skipped.

## Consequences

- Reliable persistence (SQLite for local TB-scale).
- Data integrity (validation/updates).
- Flexible (dynamic companies).

- Single-threaded by default (mitigated by session management).

**Changelog:**  

- 1.0 (July 29, 2025): Consolidated schema, updates, validation, and company management into one ADR.
