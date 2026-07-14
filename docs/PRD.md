# What Job Tracker must do

**Content type:** Conceptual
**Status:** Active
**Updated:** 2026-07-14

This product requirements document defines the supported single-user workflow: configure saved searches, collect valid job postings, manage application state, and inspect the resulting job market data locally.

## Product goal

Help one job seeker maintain a trustworthy, searchable record of relevant openings without copying postings between tools.

The product succeeds when you can:

1. Define a repeatable job-board search
2. Run one search or every enabled search
3. See whether each run succeeded, failed, or was cancelled
4. Review newly persisted jobs without duplicate rows
5. Track favorites, notes, application status, and archived jobs
6. Search and summarize the committed job data

## Supported audience

The application targets one technical user running Streamlit on a workstation or single Docker host. Multi-user authorization, shared workspaces, and distributed workers are outside the current scope.

## Saved-search requirements

A saved search is the only scrape configuration source. It stores:

- Name and query
- Location
- One or more supported job sites
- Remote-only flag
- Optional job type
- Result limit from 1 through 1000
- Enabled state

The application must support create, read, update, delete, run-one, and run-enabled operations. Deleting or disabling a saved search must not delete jobs.

Each run must expose the same health contract:

- Completion time
- Status: `never`, `running`, `succeeded`, `partial`, `failed`, or `cancelled`
- Jobs seen
- Jobs inserted
- Duration in milliseconds
- Latest error, when present

A provider response with zero jobs is successful. A provider exception or explicit failed response is not.

## Scraping and persistence requirements

JobSpy supplies structured job-board results. The adapter must use typed site and job-type values, annualized salaries, and asynchronous thread delegation for its blocking provider call.

The ingestion boundary must reject a row without a title, company, or usable direct or listing URL. It must normalize provider scalar and list shapes before validation.

One provider result persists in one transaction. Any row-level persistence failure must roll back the result.

Persistence must apply these rules:

- Prefer the direct application URL over the listing URL
- Use the application URL as the durable job identity
- Create a company only for a valid persisted job
- Update provider-owned fields on a repeated URL
- Preserve favorite, notes, application status, application date, and archive state
- Report exact inserted, updated, and skipped counts

## Job-management requirements

You must be able to:

- Filter jobs by company, application status, date, salary, favorite state, and archive state
- Open one job by database identifier
- Update application status and preserve the first application date
- Toggle favorite state
- Update notes
- Archive a job
- Apply batch updates to user-owned fields

The default job list must exclude archived jobs.

## Company requirements

Companies are read-only facets derived from persisted jobs. A company response includes its identity, URL when known, total jobs, active jobs, and latest posting date.

Companies must not store scrape schedules, enabled flags, run counts, or run success rates.

## Search requirements

Search must query the canonical application database and return detached `Job` data transfer objects. Every whitespace-separated term must match at least one of title, description, company, or location.

Search must reuse the job filter contract. It must treat `%` and `_` as literal text, exclude archived jobs by default, and paginate the full matching set without a hidden result cap.

Stemming, Boolean syntax, relevance ranking, and a second search connection are outside the current scope.

## Analytics and cost requirements

Analytics must query the same committed database state as job lists. It must report:

- Daily job trends
- Top companies with salary summaries
- Salary count, average, range, and standard deviation
- Service status without claiming an external cache or engine

Cost monitoring must store typed JSON metadata in the application database. It must aggregate current-month costs by service and return budget health at 60%, 80%, and 100% utilization thresholds.

## Data-integrity requirements

`src/database.py` is the only application engine and session owner. Services may accept an explicit SQLAlchemy bind for tests, but they must not create engines.

The database must enforce:

- Unique, nonblank company names
- Nonblank job titles and unique, nonblank job links
- A required company foreign key for every job
- Nonblank saved-search text, valid result limits, and nonnegative run health
- Nonnegative costs and nonblank cost service and operation fields

Alembic migrations must run before the UI starts. A migration failure must stop startup.

## User-interface requirements

The Streamlit interface must provide exactly three top-level pages: **Jobs**, **Searches**, and **Insights**. Company facets belong in job filters and read-only insights. The interface must not expose the removed Companies, generic Scraping, Settings, or Analytics pages.

Each saved search must expose one explicit **Run now** action. The interface must not poll, auto-refresh, or expose a generic refresh control. It must expose loading and error states, support keyboard operation, and respect reduced-motion preferences.

## Quality requirements

Every change must pass the relevant tests, Ruff checks, and Alembic drift check. Tests must use a temporary database per test and inject it through the canonical database helper.

Performance claims require measured evidence. Current acceptance targets are:

- Search completes within 1000 ms on the maintained personal database
- Common filter and analytics queries complete within 500 ms on the maintained personal database
- The Docker health endpoint becomes healthy within its configured 40s startup period

These targets are release criteria, not benchmark claims.

## Technology constraints

The maintained stack is:

- Python 3.12 or newer
- uv for dependency and lockfile management
- Streamlit for the interface
- SQLModel and SQLAlchemy for persistence and queries
- SQLite for the single-user database
- Alembic for schema changes
- JobSpy for job-board collection
- Pydantic for boundary validation
- pytest and Ruff for verification

DuckDB and sqlite-utils are not runtime dependencies.

## Excluded scope

The current release does not promise:

- Multi-user accounts or permissions
- Distributed scraping workers
- Company-career-page crawling
- Automatic background refresh
- Enterprise-scale search
- A separate analytical warehouse
- AI-assisted extraction or summarization
- Guaranteed provider availability

## Architecture decision

[ADR-041](./developers/adrs/adr-041-canonical-data-boundary.md) records the weighted decision for the canonical data boundary and its migration consequences.
