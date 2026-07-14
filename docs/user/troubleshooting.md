# Troubleshoot Job Tracker

**Content type:** Troubleshooting

Use these checks when Job Tracker does not start, collect jobs, or display expected data. Preserve `jobs.db` while diagnosing problems because it contains your stages and notes.

## Fix startup failures

### The shell cannot find `uv`

Install uv with the [official installer](https://docs.astral.sh/uv/getting-started/installation/), then restart your shell.

### Python reports a missing module

Restore the locked environment:

```bash
uv sync --locked
```

### Alembic reports an outdated schema

Apply every migration before starting the app:

```bash
uv run --locked alembic upgrade head
```

The workflow migration rewrites the former job states to `Inbox`, `Saved`, `Applied`, and `Closed`. It also enforces the five current stage values.

If migration reports invalid legacy cost entries, repair the listed timestamp, cost, service, or operation rows in a backup copy before retrying. The error includes each affected row identifier. Validation runs before schema DDL, and SQLite schema changes run in one transaction, so a corrected migration can be retried without removing an Alembic temporary table.

The follow-up migration imports the retired `costs.db` beside `jobs.db` into the canonical database once. It leaves `costs.db` unchanged and remaps an imported identifier if that identifier already belongs to another canonical cost. Back up both files before migration and keep the legacy file until you have verified the imported count.

Malformed legacy jobs are preserved. The Jobs page labels a recovered record when its original posting link was blank, and uses explicit placeholder text for a missing title or company. Every legacy company source appears under **Searches** with a `Migrated company:` prefix and keeps its former enabled state, including sources that had never run.

### Port 8501 is unavailable

Start the server on another port:

```bash
uv run --locked ai-job-scraper --port 8502
```

## Diagnose saved-search runs

### A run succeeds with zero jobs

Zero matches are a valid provider response. Review the query, location, job boards, remote preference, and results limit under **Searches**.

If only some provider rows validate, the run reports `partial` and explains how many rows were skipped. A nonempty response in which every row is invalid reports `failed`, not a successful zero-result run.

Broaden one input at a time, then select **Run now** again. The run card updates its jobs-seen and jobs-new counts.

### A run fails

Read the latest error on the saved-search card. Provider outages, rate limits, and changed job-board behavior can fail a run without changing previously collected jobs.

Retry the same search once. If it fails again, run the scraping tests and inspect the application log:

```bash
uv run --locked pytest -q tests/services/test_job_scraper.py tests/scraping
tail -n 100 app.log
```

### A saved search cannot run

Check its **Enabled** box. Disabled searches keep their definition and run history but disable **Run now**.

## Diagnose job views

### Collected jobs do not appear

Check the selected workflow stage on **Jobs**. New jobs appear in **Inbox**, not across every stage.

Clear **Search jobs**, **Companies**, and **Starred only**. The result count updates after each filter change.

### A company is missing from Insights

Companies are derived from persisted jobs. The facet appears after a successful run stores at least one valid job for that company.

You cannot create an empty company record or use a company as scrape configuration. Create a saved search for the target role or employer instead.

## Recover from database errors

### SQLite reports `database is locked`

Stop duplicate Job Tracker or migration processes. Then restart one application process:

```bash
uv run --locked ai-job-scraper
```

### The database contains unexpected data

Use SQLite's online backup command before any destructive action. It creates a
consistent snapshot even when the source database uses WAL:

```bash
mkdir -p backups
stamp=$(date +%Y%m%d-%H%M%S)
sqlite3 jobs.db ".backup 'backups/jobs-${stamp}.db'"
if [ -f costs.db ]; then
  sqlite3 costs.db ".backup 'backups/costs-${stamp}.db'"
fi
```

Run the database and migration tests:

```bash
uv run --locked pytest -q tests/unit/database tests/integration/test_database_transactions.py
uv run --locked alembic check
```

Do not delete `jobs.db` unless you intend to remove every job, stage, star, note, and saved search.

## Report an unresolved problem

Open a [GitHub issue](https://github.com/BjornMelin/ai-job-scraper/issues) with:

- Your operating system and Python version
- The exact command you ran
- The complete error message
- The smallest sequence that reproduces the failure

Remove API keys, tokens, and private job notes before attaching logs.
