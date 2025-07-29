# ADR-007: Dockerization and Deployment

## Title

Containerization for Consistent Setup and Deployment

## Version/Date

1.0 / July 29, 2025

## Status

Accepted

## Context

Easy/reproducible env (Docker for deps/browsers/DB persistence).

## Related Requirements

- Volume for jobs.db.
- Install system deps for Playwright.

## Alternatives

- Virtualenv only: Manual browser installs.
- Kubernetes: Overkill.

## Decision

Dockerfile with python-slim/base deps/uv sync/playwright install; docker-compose with volume/ports.

## Related Decisions

- ADR-003 (DB volume).

## Design

- **Dockerfile**: FROM python:3.12-slim; RUN apt-get for Playwright; uv sync; CMD streamlit run.
- **Compose**: services: app: build ., volumes: ./jobs.db:/app/jobs.db, ports: 8501.
- **Integration**: docker-compose up.
- **Implementation Notes**: Expose 8501; Browser install in build.
- **Testing**: Build/run, access UI, assert scraping works.

## Consequences

- Consistent (cross-platform).
- Persistent (volume).

- Build time (for deps).
