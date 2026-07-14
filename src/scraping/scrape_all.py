"""Run saved searches and persist truthful run health."""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from time import perf_counter

from src.database import db_session
from src.models.job_models import SavedSearchRunStatus
from src.schemas import SavedSearch, SavedSearchRunHealth
from src.services.job_service import job_service
from src.services.saved_search_service import saved_search_service

logger = logging.getLogger(__name__)


class SavedSearchRunInProgressError(RuntimeError):
    """Raised when a saved search already has a live run lease."""


class _SavedSearchLeaseLost(RuntimeError):
    """Abort a result transaction whose run lease is no longer current."""


def _health(
    status: SavedSearchRunStatus,
    started_clock: float,
    *,
    jobs_seen: int = 0,
    jobs_new: int = 0,
    error: str | None = None,
) -> SavedSearchRunHealth:
    return SavedSearchRunHealth(
        last_run_at=datetime.now(UTC),
        last_run_status=status,
        jobs_seen=jobs_seen,
        jobs_new=jobs_new,
        duration_ms=round((perf_counter() - started_clock) * 1000),
        last_error=error,
    )


async def run_saved_search(search_id: int) -> SavedSearch | None:
    """Run one saved search and return its updated card-ready state."""
    started_at = datetime.now(UTC)
    started_clock = perf_counter()
    search = saved_search_service.claim_run(search_id, started_at)
    if search is None:
        if saved_search_service.get(search_id) is None:
            return None
        raise SavedSearchRunInProgressError(
            f"Saved search {search_id} already has an active run"
        )
    result = None
    raw_found = 0
    try:
        result = await job_service.search_and_save_jobs(
            search_term=search.query,
            location=search.location,
            sites=search.sites,
            is_remote=search.remote_only,
            job_type=search.job_type,
            results_wanted=search.results_limit,
            save_to_db=False,
        )
        raw_found = int(result.metadata.get("raw_found", result.total_found))
        invalid_rows = int(result.metadata.get("invalid_rows", 0))
        if not result.metadata.get("success", False):
            health = _health(
                SavedSearchRunStatus.FAILED,
                started_clock,
                jobs_seen=raw_found,
                error=str(result.metadata.get("error", "Scrape failed")),
            )
        else:
            status = (
                SavedSearchRunStatus.PARTIAL
                if invalid_rows
                else SavedSearchRunStatus.SUCCEEDED
            )
            health = _health(
                status,
                started_clock,
                jobs_seen=raw_found,
                error=(
                    str(
                        result.metadata.get(
                            "warning",
                            f"{invalid_rows} provider rows failed validation",
                        )
                    )
                    if invalid_rows
                    else None
                ),
            )
    except asyncio.CancelledError:
        saved_search_service.record_run(
            search_id,
            _health(
                SavedSearchRunStatus.CANCELLED,
                started_clock,
                error="Run cancelled",
            ),
            expected_started_at=started_at,
        )
        raise
    except Exception as error:
        logger.exception("Saved search %s failed", search_id)
        health = _health(
            SavedSearchRunStatus.FAILED,
            started_clock,
            error=str(error),
        )
    try:
        with db_session() as session:
            if result is not None and result.metadata.get("success", False):
                persistence = job_service._save_jobs_to_database(
                    result.jobs,
                    session=session,
                )
                result.metadata["persistence"] = persistence
                health.jobs_new = persistence["inserted"]
            completed = saved_search_service.record_run(
                search_id,
                health,
                expected_started_at=started_at,
                session=session,
            )
            if completed is None:
                raise _SavedSearchLeaseLost
    except _SavedSearchLeaseLost:
        if saved_search_service.get(search_id) is None:
            return None
        raise SavedSearchRunInProgressError(
            f"Saved search {search_id} was superseded by a newer run"
        ) from None
    except Exception as error:
        logger.exception("Saved search %s could not commit its result", search_id)
        completed = saved_search_service.record_run(
            search_id,
            _health(
                SavedSearchRunStatus.FAILED,
                started_clock,
                jobs_seen=raw_found,
                error=str(error),
            ),
            expected_started_at=started_at,
        )
    if completed is None and saved_search_service.get(search_id) is not None:
        raise SavedSearchRunInProgressError(
            f"Saved search {search_id} was superseded by a newer run"
        )
    return completed


async def scrape_all() -> list[SavedSearch]:
    """Run every enabled saved search and return their latest health."""
    results: list[SavedSearch] = []
    for search in saved_search_service.list(enabled_only=True):
        try:
            result = await run_saved_search(search.id)
        except SavedSearchRunInProgressError:
            logger.info("Skipping saved search %s with an active run", search.id)
            continue
        if result is not None:
            results.append(result)
    return results


def scrape_all_sync() -> list[SavedSearch]:
    """Run enabled saved searches from synchronous callers."""
    return asyncio.run(scrape_all())
