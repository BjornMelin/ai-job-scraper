# Application API reference

**Content type:** Reference

This reference documents the maintained Python contracts used by services, scraping orchestration, and Streamlit pages.

## Enumerations

`src/models/job_models.py` defines the external values accepted by saved searches and JobSpy requests.

### `JobSite`

Supported values are `linkedin`, `indeed`, `glassdoor`, `zip_recruiter`, and `google`.

`JobSite.normalize(value)` accepts common case, space, and hyphen variants. It returns `None` for an unknown site.

### `JobType`

Supported values are `fulltime`, `parttime`, `contract`, `internship`, and `temporary`.

### `SavedSearchRunStatus`

The finite run states are `never`, `running`, `succeeded`, `partial`, `failed`, and `cancelled`.

## Saved-search schemas

`src/schemas.py` owns detached saved-search contracts.

### `SavedSearchCreate`

| Field | Type | Default |
| --- | --- | --- |
| `name` | `str` | Required |
| `query` | `str` | Required |
| `location` | `str` | `United States` |
| `sites` | `list[JobSite]` | LinkedIn and Indeed |
| `remote_only` | `bool` | `False` |
| `job_type` | `JobType | None` | `None` |
| `results_limit` | `int` | `50`, range 1 through 1000 |
| `enabled` | `bool` | `True` |

Name, query, and location are stripped and cannot be blank.

### `SavedSearchUpdate`

Every editable `SavedSearchCreate` field is optional. `model_dump(exclude_unset=True)` distinguishes an omitted field from an explicit `None` job type.

### `SavedSearchRunHealth`

| Field | Type | Default |
| --- | --- | --- |
| `last_run_at` | `datetime | None` | `None` |
| `last_run_status` | `SavedSearchRunStatus` | `never` |
| `jobs_seen` | `int` | `0` |
| `jobs_new` | `int` | `0` |
| `duration_ms` | `int | None` | `None` |
| `last_error` | `str | None` | `None` |

`SavedSearch` combines the definition, database identifier, and run-health fields.

## Job and company schemas

### `Job`

`Job` is the detached result returned to the UI. It includes database and company identifiers, resolved company name, provider fields, salary pair, favorite, notes, application state, archive state, and last-seen timestamp.

Computed fields are:

- `salary_range_display`
- `days_since_posted`
- `is_recently_posted`

### `Company`

`Company` is a read-only job-derived facet with `id`, `name`, `url`, `total_jobs`, `active_jobs`, and `last_job_posted`.

## JobSpy contracts

### `JobScrapeRequest`

The request accepts sites, query, location, distance, remote flag, job type, result limit, country, offset, age, easy-apply flag, and description options. Salaries default to annualized provider output.

### `JobPosting`

A posting requires `id`, `site`, `title`, `company`, and a direct or listing URL. Provider list fields normalize scalar strings to one-item lists.

### `JobScrapeResult`

The result contains validated jobs, `total_found`, the request, and metadata. JobService adds exact persistence counts under `metadata["persistence"]`.

## Database API

`src/database.py` owns every application connection.

### `get_engine(database_url: str | None = None) -> Engine`

Returns the lazily created process engine. In-memory SQLite uses `StaticPool`; file SQLite uses the normal SQLAlchemy pool.

### `get_session(bind: Engine | Connection | None = None) -> Session`

Opens a session with `expire_on_commit=False`.

### `db_session(bind=None)`

Commits after a successful context and rolls back after an exception.

### `db_session_no_autocommit(bind=None)`

Lets the caller control commits and rolls back any remaining transaction on exit.

### `create_db_and_tables(bind=None)`

Creates registered SQLModel tables. Production startup uses Alembic migrations instead.

### `get_connection_pool_status(bind=None) -> dict[str, Any]`

Returns bounded pool diagnostics and a password-hidden engine URL.

## Saved-search service

`SavedSearchService` supports:

```text
list(*, enabled_only: bool = False) -> list[SavedSearch]
get(search_id: int) -> SavedSearch | None
create(data: SavedSearchCreate) -> SavedSearch
update(search_id: int, data: SavedSearchUpdate) -> SavedSearch | None
delete(search_id: int) -> bool
claim_run(search_id: int, started_at: datetime) -> SavedSearch | None
record_run(search_id: int, health: SavedSearchRunHealth, *, expected_started_at: datetime | None = None) -> SavedSearch | None
```

Deleting a saved search never deletes jobs. A run claim lasts 30 minutes. A concurrent caller cannot claim the same search, and a superseded run cannot overwrite newer health. `record_run()` stamps the terminal write time as `last_run_at`; a caller-provided health timestamp cannot make completion appear to happen at lease start.

## Job service

`JobService` supports job reads and user-owned updates:

```text
get_filtered_jobs(filters: dict[str, object] | None = None) -> list[Job]
get_job_by_id(job_id: int) -> Job | None
get_recent_jobs(days: int = 7, limit: int = 100) -> list[Job]
get_job_counts_by_status() -> dict[str, int]
update_job_status(job_id: int, status: str) -> bool
toggle_favorite(job_id: int) -> bool
update_notes(job_id: int, notes: str) -> bool
archive_job(job_id: int) -> bool
bulk_update_jobs(updates: list[dict[str, object]]) -> bool
```

Supported filter keys are `company`, `application_status`, `date_from`, `date_to`, `favorites_only`, `salary_min`, `salary_max`, and `include_archived`.

The provider entry point is:

```text
await search_and_save_jobs(
    search_term: str,
    location: str | None = None,
    sites: list[str | JobSite] | None = None,
    is_remote: bool = False,
    job_type: JobType | None = None,
    results_wanted: int = 100,
    save_to_db: bool = True,
) -> JobScrapeResult
```

## Scraping orchestration

`JobSpyScraper.scrape_jobs_async(request)` delegates the blocking JobSpy call to a worker thread. Provider exceptions return an explicit failed `JobScrapeResult`; successful empty results remain successful.

Saved-search orchestration exports:

```text
await run_saved_search(search_id: int) -> SavedSearch | None
await scrape_all() -> list[SavedSearch]
scrape_all_sync() -> list[SavedSearch]
```

`scrape_all()` runs enabled saved searches in name order.

## Read services

### `CompanyService`

`get_all_companies() -> list[Company]` returns only companies referenced by jobs. Counts and latest dates are aggregate query results.

### `JobSearchService`

```text
JobSearchService(bind: Engine | Connection | None = None)
search_jobs(query: str, filters: dict | None = None, limit: int = 50, offset: int = 0) -> list[Job]
count_jobs(query: str, filters: dict | None = None) -> int
get_search_stats() -> dict[str, Any]
```

Search applies AND semantics across whitespace-separated terms and searches title, description, company, and location. The Jobs page combines the count, limit, and offset operations so every matching job remains reachable.

### `AnalyticsService`

```text
AnalyticsService(bind: Engine | Connection | None = None)
get_job_trends(days: int = 30) -> dict[str, Any]
get_company_analytics() -> dict[str, Any]
get_salary_analytics(days: int = 90) -> dict[str, Any]
get_status_report() -> dict[str, Any]
```

Successful analytics responses contain `status="success"` and `method="sqlalchemy"`. Query errors return a stable error envelope.

### `CostMonitor`

```text
CostMonitor(
    bind: Engine | Connection | None = None,
    *,
    monthly_budget: float = 50.0,
)
```

Tracking methods are `track_ai_cost`, `track_proxy_cost`, and `track_scraping_cost`. Reporting methods are `get_monthly_summary` and `get_cost_alerts`.

Cost entries use nonnegative USD values and typed JSON metadata. The monitor returns presentation-neutral warning or error records at budget thresholds.

## Startup and command-line API

`run_migrations()` upgrades to Alembic head once per process and raises on failure.

Installed commands are:

```bash
ai-job-scraper --address 127.0.0.1 --port 8501
ai-job-seed
```
