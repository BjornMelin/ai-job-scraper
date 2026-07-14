# Get started with Job Tracker

**Content type:** Tutorial

Install Job Tracker, create a saved search, and collect your first jobs. You need Python 3.12, Git, and [uv](https://docs.astral.sh/uv/).

## Install the application

Clone the repository and install the locked dependencies:

```bash
git clone https://github.com/BjornMelin/ai-job-scraper.git
cd ai-job-scraper
uv sync --locked
```

Copy the environment template:

```bash
cp .env.example .env
```

## Prepare the database

Apply every Alembic migration:

```bash
uv run --locked alembic upgrade head
```

Add the starter saved searches if you want examples to edit:

```bash
uv run --locked ai-job-seed
```

## Start Job Tracker

Launch the local Streamlit server:

```bash
uv run --locked ai-job-scraper
```

Open `http://localhost:8501`.

## Collect your first jobs

Create and run one saved search:

1. Select **Searches** in the top navigation
2. Open **New saved search**
3. Enter a name, keywords, location, and at least one job board
4. Select **Create saved search**
5. Select **Run now** on the saved-search card

The run reports jobs seen, jobs added, duration, and any provider error. A successful search with no matches reports zero jobs instead of a failure.

## Review the results

Select **Jobs** in the top navigation. New jobs start in **Inbox**.

Open **Review and update** on any job to:

- Change its workflow stage
- Mark it as starred
- Add private notes
- Open the original posting

Use [Job Tracker’s workflow](user-guide.md) for the complete interface reference.
