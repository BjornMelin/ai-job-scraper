# Deploy Job Tracker

**Content type:** How-to

This guide deploys the single-user Streamlit application with a persistent SQLite database. Use one application process per database file.

## Prepare configuration

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/). Clone the repository, then create your environment file:

```bash
cp .env.example .env
```

Set `DB_URL` to the persistent SQLite location and keep `.env` out of version control.

## Run on the host

Install the locked environment:

```bash
uv sync --locked
```

Apply and verify the schema:

```bash
uv run --locked alembic upgrade head
uv run --locked alembic check
```

Create starter saved searches once:

```bash
uv run --locked ai-job-seed
```

Start the dashboard:

```bash
uv run --locked ai-job-scraper --address 127.0.0.1 --port 8501
```

Open `http://127.0.0.1:8501`. The application runs migrations once per process before Streamlit starts. A migration failure stops startup.

## Run with Docker Compose

### Migrate the legacy bind mount once

Releases before the named-volume cutover stored Docker data in `./dbdata`. If
`dbdata/jobs.db` exists, stop the old container and copy the complete stopped
SQLite directory into the new volume before the first start. This preserves the
database and any WAL sidecars without retaining a host-permission-sensitive bind
mount:

```bash
docker compose down
mkdir -p backups
stamp=$(date +%Y%m%d-%H%M%S)
cp -a dbdata "backups/dbdata-${stamp}"
docker compose build app
docker compose run --rm --no-deps \
  --entrypoint sh \
  -v "$PWD/dbdata:/legacy:ro" \
  app -c 'test ! -e /app/db/jobs.db && cp -R /legacy/. /app/db/'
```

The copy refuses to overwrite a database already present in `job-data`. Keep
the backup until the upgraded application is healthy and its data is verified.
Skip this one-time procedure on a fresh install or when `job-data` already owns
the database.

### Start the application

Start the container. Compose creates the persistent `job-data` volume with the
ownership required by the non-root runtime user:

```bash
docker compose up --build -d
```

The compose file mounts `job-data` at `/app/db` and sets
`DB_URL=sqlite:////app/db/jobs.db`. Set `SCRAPER_LOG_LEVEL` in the shell or an
optional `.env` file when the default `INFO` level is not appropriate. The
container runs as a non-root user.

Seed the persistent database after the first successful start:

```bash
docker compose exec app /app/.venv/bin/ai-job-seed
```

Check health and logs:

```bash
curl --fail http://127.0.0.1:8501/_stcore/health
docker compose logs --tail=200 app
```

The health check allows 40s for startup, then checks every 30s.

## Verify a deployment

Run schema and application checks against the deployed configuration:

```bash
uv run --locked alembic current
uv run --locked alembic check
uv run --locked pytest -q
```

Run the schema checks inside Docker with explicit executables:

```bash
docker compose exec app /app/.venv/bin/alembic current
docker compose exec app /app/.venv/bin/alembic check
```

Confirm these behaviors in the browser:

1. The jobs page loads without a schema error
2. Saved searches list seeded or user-created definitions
3. Running one saved search updates its status and duration
4. Jobs remain after deleting a saved search
5. Search and analytics reflect the latest committed data

## Back up SQLite

Stop writes, then copy the complete database directory so any WAL sidecars stay
with their database:

```bash
docker compose stop app
stamp=$(date +%Y%m%d-%H%M%S)
mkdir -p "backups/job-data-${stamp}"
docker cp ai-job-scraper:/app/db/. "backups/job-data-${stamp}/"
docker compose start app
```

Keep at least one backup outside the deployment host before applying a new migration.

## Upgrade the application

Back up the database, fetch the new code, rebuild the environment, and start one process:

```bash
git pull --ff-only
uv sync --locked
uv run --locked alembic upgrade head
uv run --locked alembic check
uv run --locked ai-job-scraper
```

For Docker, rebuild with `docker compose up --build -d`. Inspect the application logs until the health check passes.

## Diagnose migration failures

Inspect the current and expected revisions:

```bash
uv run --locked alembic current
uv run --locked alembic heads
uv run --locked alembic history
```

Do not stamp over a failed migration. Restore the backup or fix the schema mismatch, then run `alembic upgrade head` again.

## Choose a hosting target

SQLite requires a durable filesystem and one writer process. Use a virtual machine, home server, or container host with a persistent volume.

Do not deploy this database on ephemeral serverless storage. Move to PostgreSQL before adding replicas or multiple users.
