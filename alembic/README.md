# Database migrations

Alembic owns every schema change for the application's SQLite database. The
application uses the SQLModel metadata in `src/database_models.py`, SQLite batch
operations, and Python 3.12 native transaction control so interrupted DDL can be
retried safely.

## Revisions

The active history is linear:

1. `d555e0170c65` creates the original company and job tables.
2. `8f4b2c91a3d7` adds saved searches, converts companies to read-only facets,
   adopts inline legacy costs, and normalizes workflow and salary values.
3. `c91e7a4d2b6f` imports a sibling `costs.db` without modifying it, repairs
   already-stamped cost metadata and salaries, and records import provenance
   for retry safety.

The final revision is intentionally forward-only: downgrading it retains
imported costs, provenance, and repaired salary data. Re-upgrading is
idempotent.

## Apply and inspect migrations

Run all commands from the repository root with the locked environment:

```bash
uv run --locked alembic current
uv run --locked alembic history
uv run --locked alembic upgrade head
uv run --locked alembic check
```

The packaged application also upgrades to `head` during startup and stops if a
migration fails.

## Create a revision

Update the table model, service contract, and tests before generating a schema
diff:

```bash
uv run --locked alembic revision --autogenerate -m "describe the change"
```

Review every generated operation. In particular, verify SQLite batch-table
copies preserve constraints, indexes, foreign keys, and data. Internal
`_alembic_*` provenance tables are deliberately excluded from autogeneration.

## Verify a migration

Use a temporary database for ordinary upgrade and downgrade tests:

```bash
DB_URL=sqlite:////tmp/ai-job-scraper.db uv run --locked alembic upgrade head
DB_URL=sqlite:////tmp/ai-job-scraper.db uv run --locked alembic check
DB_URL=sqlite:////tmp/ai-job-scraper.db uv run --locked alembic downgrade base
DB_URL=sqlite:////tmp/ai-job-scraper.db uv run --locked alembic upgrade head
```

Rehearse data migrations against a copy of representative data, including any
sibling `costs.db`. Confirm the source file's hash is unchanged before and after
the rehearsal. Run the migration test suite as the final gate:

```bash
uv run --locked pytest -q tests/unit/database/test_migrations.py
uv run --locked alembic check
```

## Recover from a failure

Do not stamp over a failed revision and do not edit `alembic_version` manually.
Stop application writes, preserve the failed database for diagnosis, and use
one of these paths:

1. Retry the upgrade when the revision is documented and tested as retry-safe.
2. Restore the pre-upgrade backup, correct the migration, and run it again.

Inspect state before deciding:

```bash
uv run --locked alembic current
uv run --locked alembic heads
uv run --locked alembic history
```

See [the deployment guide](../docs/developers/deployment.md) for backup and
upgrade procedures and
[the architecture overview](../docs/developers/architecture-overview.md) for
the current data-boundary contract.
