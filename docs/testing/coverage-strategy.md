# Choose tests by behavior and risk

**Content type:** Conceptual

Job Tracker uses focused behavior tests instead of a repository-wide coverage target. A change is ready when the relevant contract fails before the fix and passes after it.

## Test boundaries

Choose the smallest test boundary that proves the change:

| Boundary | Use it for | Location |
| --- | --- | --- |
| Unit | Validation, formatting, and deterministic helpers | `tests/unit/` |
| Service | Queries, transactions, and provider adapters | `tests/services/` |
| Integration | Multiple services sharing one database boundary | `tests/integration/` |
| Streamlit AppTest | Visible copy, controls, and user mutations | `tests/ui/` |
| Alembic | Upgrade paths and database constraints | `tests/unit/database/test_migrations.py` |

## High-risk contracts

Every change to these contracts needs direct evidence:

- Database migrations and constraints
- Job persistence and deduplication
- Saved-search run health
- Workflow-stage mutations
- Provider failures and invalid rows
- Search filters and company facets

The UI suite in `tests/ui/test_job_tracker.py` covers the three-page information architecture, job updates, saved-search runs, empty states, failures, responsive CSS, and reduced motion.

## Run the test layers

Start with the narrowest affected layer:

```bash
uv run --locked pytest -q tests/ui/test_job_tracker.py
uv run --locked pytest -q tests/services
uv run --locked pytest -q tests/integration
```

Run the complete suite before shipping:

```bash
uv run --locked pytest -q
```

Treat a coverage report as a discovery aid, not a release gate. A covered line can still assert the wrong behavior.
