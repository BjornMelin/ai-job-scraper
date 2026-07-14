# Repository instructions

## Runtime and data

- Use `uv` with `uv.lock`; do not switch package managers.
- Apply `uv run --locked alembic upgrade head` before local startup or schema checks.
- Keep `AppSQLModel` and `src/database.py` as the only application table registry and transaction boundary.
- Persist a saved-search result and its terminal run health in one transaction. Keep the run claim separate.
- Preserve workflow stage, star, notes, archive state, and richer provider fields on repeated scrapes.
- Treat mixed valid and invalid provider rows as `partial`; treat a nonempty all-invalid response as `failed`.

## Product surface

- Keep **Jobs**, **Searches**, and **Insights** as the only top-level pages.
- Keep routing in hidden `st.navigation` and render labeled `st.page_link` controls;
  Streamlit 1.59's responsive top drawer loses its accessible name and focus order.
- Saved searches own collection configuration. Companies are read-only, job-derived facets.
- Preserve keyboard focus, responsive layout, and `prefers-reduced-motion` behavior in UI changes.

## Verification

Run the relevant focused test first, then these release gates:

```bash
uv run --locked ruff format --check .
uv run --locked ruff check .
uv run --locked pytest -q
uv run --locked alembic check
uv lock --check
```
