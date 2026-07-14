# Develop Job Tracker

**Content type:** How-to

This guide covers local setup, code ownership, database changes, and the checks required before review.

## Set up the repository

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Then install the locked development environment:

```bash
uv sync --locked
cp .env.example .env
uv run --locked alembic upgrade head
```

Start Streamlit through the packaged command:

```bash
uv run --locked ai-job-scraper
```

Run `uv run --locked ai-job-seed` when you need the five starter saved searches.

## Find the owning module

Keep business rules outside Streamlit pages:

| Path | Responsibility |
| --- | --- |
| `src/database.py` | Engine and session lifecycle |
| `src/database_models.py` | Canonical SQLModel tables and salary parsing |
| `src/schemas.py` | Detached service and UI contracts |
| `src/models/job_models.py` | JobSpy request, result, and provider-row contracts |
| `src/scraping/` | Provider adaptation and saved-search orchestration |
| `src/services/` | Queries, persistence, search, analytics, and costs |
| `src/ui/` | Streamlit rendering, interaction state, and navigation |
| `alembic/versions/` | Forward and reverse schema changes |
| `tests/` | Unit, service, integration, scraping, and UI proof |

[How the application is structured](./architecture-overview.md) documents the complete data flow.

## Preserve the canonical data boundary

Services must call `db_session()` or accept an explicit SQLAlchemy bind. Do not create an engine, open sqlite3 directly, or cache query results inside a service.

Use one transaction for one business operation:

```python
from src.database import db_session

with db_session() as session:
    session.add(record)
```

The context manager commits after a successful block and rolls back after an exception. Keep network calls outside the transaction.

SQLModel table constructors can bypass Pydantic errors. Use `model_validate()` or a model's `create_validated()` constructor at untrusted boundaries.

## Change the data model

Update the table model, service contract, migration, and tests together.

Create a revision after editing SQLModel metadata:

```bash
uv run --locked alembic revision --autogenerate -m "describe the schema change"
```

Review generated operations before running them. SQLite batch migrations can recreate tables, so check constraints, indexes, foreign keys, and data-copy behavior.

Verify a migration in both directions on a temporary database:

```bash
DB_URL=sqlite:////tmp/ai-job-scraper.db uv run --locked alembic upgrade head
DB_URL=sqlite:////tmp/ai-job-scraper.db uv run --locked alembic check
DB_URL=sqlite:////tmp/ai-job-scraper.db uv run --locked alembic downgrade base
DB_URL=sqlite:////tmp/ai-job-scraper.db uv run --locked alembic upgrade head
```

Rehearse destructive migrations against a copy of representative data. Never use the maintained `jobs.db` file for a rehearsal.

## Add a saved-search feature

Treat `SavedSearch` as the scrape unit. Companies are read-only facets and must not gain schedule or run-health fields.

When you change a run:

1. Mark the search `running`
2. Call the provider through `JobService`
3. Persist the provider result atomically
4. Record final status, counts, duration, and error
5. Return the same card-ready `SavedSearch` contract

An empty successful provider response must remain distinct from a provider failure. Mixed valid and invalid rows produce `partial`; a nonempty all-invalid response produces `failed`. Sparse repeated results must retain richer stored descriptions, locations, dates, and salary bounds.

## Change job ingestion

Validate provider rows before persistence. A valid row needs a title, company, and direct or listing URL.

The persistence path owns provider fields. Preserve these user-owned fields during a repeated scrape:

- Favorite
- Notes
- Application status
- Application date
- Archive state

Return exact inserted, updated, and skipped counts. Do not swallow transaction failures.

## Write isolated tests

The `session` fixture creates a temporary SQLite file and injects its engine through `src.database.get_engine`. Application services and factories then share one database.

Commit fixture data before calling a service that opens its own session:

```python
def test_service_reads_committed_data(session):
    company = CompanyFactory(name="Example")
    JobFactory(company_id=company.id)
    session.commit()

    assert JobService.get_filtered_jobs()
```

Mock provider boundaries, not the method under test. Do not call live job boards or sleep in unit tests.

## Run verification

Format and lint the repository:

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
```

Run the full test suite and migration drift check:

```bash
uv run --locked pytest -q
uv run --locked alembic check
```

During implementation, run the smallest relevant test cluster first. Run the full commands before review.

## Keep dependencies narrow

Use the package already selected by the repository when it owns the capability. Remove a dependency after its final import disappears, then update the lockfile with uv.

Check lockfile parity with:

```bash
uv lock --check
uv sync --locked
```

Do not add another database, cache, search engine, or orchestration layer without measured evidence and an architecture decision record.

## Update documentation

Update active docs when behavior or ownership changes. Mark old architecture decision records as superseded instead of rewriting their historical rationale.

The current decision is [ADR-041](./adrs/adr-041-canonical-data-boundary.md).
