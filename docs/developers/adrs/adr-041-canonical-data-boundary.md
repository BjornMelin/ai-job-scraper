# ADR-041: Use one application data boundary

## Metadata

**Status:** Accepted
**Version/date:** v1.0 / 2026-07-14
**Content type:** Conceptual

This decision makes SQLAlchemy and SQLModel the only owners of application data connections, transactions, search, and analytics.

## Why the data boundary changed

The application previously opened separate SQLite, sqlite-utils, and DuckDB connections. Those paths used different table names, transaction lifecycles, and cache rules. Tests could pass against one database while production services queried another.

Companies also mixed two responsibilities: job-derived facets and scrape configuration. Empty company URLs then became invalid scrape inputs, while hardcoded terms bypassed saved configuration.

## Decision

Use these canonical owners:

- `src/database.py` creates engines lazily and owns session context managers
- `AppSQLModel` owns the table registry so Streamlit source reloads cannot redefine tables in SQLModel's global registry
- `SavedSearchSQL` owns repeatable scrape definitions and latest run health
- `JobSQL` owns persisted postings with a nonblank application-link constraint
- `CompanySQL` identifies companies referenced by jobs; company metrics are query results
- `JobSearchService` searches through the canonical SQLAlchemy session and returns `Job` data transfer objects
- `AnalyticsService` computes current personal-scale aggregates through the same session boundary
- `CostMonitor` records typed cost events through the same transaction helpers

A saved search stores its query, location, sites, remote flag, job type, result limit, enabled state, and latest run health. Deleting or disabling a saved search never deletes jobs.

Each job stores one typed stage: `Inbox`, `Saved`, `Applied`, `Interviews`, or `Closed`. The database enforces the same set. Stars and notes remain independent from stage.

## Decision score

The score uses the repository decision framework.

| Option | Solution leverage, 35% | Application value, 30% | Maintenance load, 25% | Adaptability, 10% | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| One SQLAlchemy boundary | 9.5 | 9.4 | 9.8 | 8.8 | **9.48** |
| FTS5 on the same engine | 8.4 | 8.8 | 7.0 | 9.0 | 8.23 |
| Separate sqlite-utils and DuckDB owners | 7.5 | 8.5 | 3.5 | 7.5 | 6.80 |

The selected option removes more code and prevents cross-engine state drift. Add a specialized search or analytics engine only after measured workload evidence supports its operational cost.

## Run-health contract

Every saved-search run records one status: `never`, `running`, `succeeded`, `partial`, `failed`, or `cancelled`. The same response includes jobs seen, jobs inserted, duration, completion time, and the latest error.

Provider failures and empty successful searches remain distinct. A successful search with zero jobs records `succeeded`; a provider error records `failed`.

The `running` timestamp is also a 30-minute lease. One caller claims the lease atomically, and completion updates only the lease it started. The terminal write replaces the lease start with the actual completion time. A later stale-run recovery therefore cannot be overwritten by the older process.

## Migration and verification

Revision `8f4b2c91a3d7` removes company scrape-state columns and creates saved searches with nonnegative run-health constraints. It preserves malformed legacy jobs by replacing blank titles, blank links, and missing company references with explicit recovery values; stars, notes, and workflow state remain intact. SQL and JSON null salaries become the canonical empty range before the salary column becomes required. Orphaned company rows are also preserved. Every legacy company row becomes a `Migrated company:` saved search before those obsolete columns are removed, including active sources that had never run, and retains its former enabled state.

The revision also adopts cost history created by the former service-owned table. It rejects invalid timestamps, negative or nonnumeric costs, and blank ownership fields before schema DDL, converts valid JSON-object metadata directly, maps blank metadata to an empty object, and preserves malformed or non-object text under `legacy_raw`. An internal migration-state row records whether the cost table predated the revision so downgrade restores adopted history and drops only a table created by this revision.

Revision `c91e7a4d2b6f` repairs databases already stamped by the draft revision. For SQLite, it imports rows from the retired sibling `costs.db` without modifying or deleting the source. A source-identifier to target-identifier mapping preserves colliding rows without overwriting either record and makes downgrade/re-upgrade idempotent. The same revision validates already-adopted cost timestamps, converts every adopted metadata value to a readable JSON object while preserving non-object raw text, and losslessly normalizes two-element finite, integral numeric salary pairs to integer-or-null JSON. Any other salary shape aborts before the first mutation with its row identifier and raw value; after repairing those rows, the operator retries the revision. The successful retry reconciles the salary column to `NOT NULL`.

This repair is forward-only. Its downgrade is intentionally a no-op: it retains imported costs, the minimal identifier mapping, normalized salaries, and the required salary column. Re-upgrading therefore neither duplicates history nor reintroduces malformed data.

The migration maps the former job states to the five-stage workflow before adding the database constraint. It rejects an unknown legacy stage before making its first change. Alembic enables Python 3.12's native SQLite transaction mode, so an interruption after schema DDL rolls the complete revision back and a clean retry starts from the prior revision.

Verify the boundary with:

```bash
uv run --locked alembic upgrade head
uv run --locked alembic check
uv run --locked pytest -q
```

## Consequences

The application no longer depends on DuckDB or sqlite-utils. Search uses literal, case-insensitive, multi-term matching without stemming or relevance ranking. Analytics and cost monitoring no longer create engines or service-layer caches. The current workload favors the smaller operational surface; measured latency can justify a future indexed search design.
