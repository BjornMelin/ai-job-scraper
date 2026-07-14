# ADR-037: Keep one native Streamlit interface

**Status:** Accepted

**Version/date:** v3.0 / 2026-07-14

**Content type:** Conceptual

Job Tracker uses one three-page Streamlit interface and one shared design module. This decision removes the parallel component, sidebar, cache, URL-state, and company-scraping paths.

## Context

The former interface spread one personal workflow across five pages: Jobs, Companies, Scraping, Analytics, and Settings. Company records mixed derived job data with mutable scrape configuration.

The UI also duplicated native Streamlit behavior through custom cards, background helpers, session registries, mobile detection, service caches, and URL-state adapters. Those paths increased maintenance without creating a stronger workflow.

## Decision

Use these product boundaries:

- Name the user-facing product Job Tracker
- Use labeled `st.page_link` controls for top navigation with **Jobs**, **Searches**, and **Insights**
- Keep `st.navigation` hidden so routing stays native without exposing Streamlit 1.59's inaccessible responsive drawer
- Make saved searches the only collection configuration and manual run surface
- Treat companies as read-only facets derived from persisted jobs
- Track jobs through `Inbox`, `Saved`, `Applied`, `Interviews`, and `Closed`
- Keep stars independent from workflow stage
- Use native Streamlit forms, status containers, charts, tables, and navigation
- Keep visual tokens and CSS in `src/ui/design.py`
- Use CSS transitions only for state clarification
- Respect `prefers-reduced-motion` and preserve visible keyboard focus

The interface does not add a JavaScript component, animation runtime, or second design system.

The native top-position drawer was rejected because its mobile trigger has no useful accessible name and moves focus behind the open menu. Hidden native routing plus visible page links preserves direct URLs, session behavior, labels, and keyboard order without DOM mutation.

## Decision score

The score uses the repository decision framework.

| Option | Solution leverage, 35% | Application value, 30% | Maintenance load, 25% | Adaptability, 10% | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Three-page native Streamlit interface | 9.5 | 9.4 | 9.7 | 6.8 | **9.25** |
| Streamlit shell with a custom JavaScript component | 7.0 | 9.0 | 6.0 | 8.5 | 7.50 |
| Preserve the five-page component architecture | 4.0 | 5.0 | 2.0 | 5.0 | 3.90 |

The selected option meets the product workflow with the installed platform. A custom component becomes relevant only when a measured interaction cannot be implemented accessibly in Streamlit.

## Consequences

The hard cut deletes the former Companies, Scraping, Settings, and Analytics pages. It also deletes their UI helpers, tests, and cache manager.

The **Insights** page replaces Analytics with read-only summaries. The **Searches** page replaces company scraping and generic refresh controls with saved-search forms and explicit runs.

The interface favors responsive vertical flow over dense desktop dashboards. It uses warm neutral surfaces, one green accent, system typography, visible labels, native focus behavior, and reduced-motion CSS.

## Verification

The UI suite verifies the information architecture, stage mutation, saved-search creation and runs, company facets, empty states, failures, responsive CSS, and reduced motion.

```bash
uv run --locked pytest -q tests/ui/test_job_tracker.py
uv run --locked ruff check src/ui tests/ui
```
