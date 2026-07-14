"""Tests for saved-search orchestration and card-ready run health."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlmodel import select
from src.database_models import JobSQL
from src.models.job_models import (
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    SavedSearchRunStatus,
)
from src.schemas import SavedSearchCreate
from src.scraping.scrape_all import (
    SavedSearchRunInProgressError,
    run_saved_search,
    scrape_all,
    scrape_all_sync,
)
from src.services.job_service import job_service
from src.services.saved_search_service import SavedSearchService


def _posting(slot: int) -> JobPosting:
    return JobPosting(
        id=f"job-{slot}",
        site=JobSite.LINKEDIN,
        title=f"AI Engineer {slot}",
        company="Acme",
        job_url=f"https://example.test/jobs/{slot}",
    )


def _result(
    *,
    found: int = 0,
    inserted: int = 0,
    invalid: int = 0,
    success: bool = True,
):
    request = JobScrapeRequest(site_name=JobSite.LINKEDIN, search_term="AI")
    valid = max(found - invalid, 0)
    distinct = max(min(inserted, valid), 1)
    jobs = [_posting(index % distinct) for index in range(valid)] if success else []
    metadata = {
        "success": success,
        "raw_found": found,
        "valid_rows": len(jobs),
        "invalid_rows": invalid,
    }
    if not success:
        metadata["error"] = "provider unavailable"
    elif invalid:
        metadata["warning"] = f"{invalid} of {found} provider rows failed validation"
    return JobScrapeResult(
        jobs=jobs,
        total_found=len(jobs),
        request_params=request,
        metadata=metadata,
    )


def _searches():
    return SavedSearchService()


@pytest.mark.asyncio
async def test_run_saved_search_returns_success_health(session, monkeypatch):
    search = _searches().create(
        SavedSearchCreate(
            name="Remote AI",
            query="AI engineer",
            sites=[JobSite.LINKEDIN],
            remote_only=True,
            results_limit=40,
        )
    )
    mocked = AsyncMock(return_value=_result(found=7, inserted=3))
    monkeypatch.setattr(job_service, "search_and_save_jobs", mocked)

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.SUCCEEDED
    assert completed.jobs_seen == 7
    assert completed.jobs_new == 3
    assert completed.duration_ms is not None
    mocked.assert_awaited_once_with(
        search_term="AI engineer",
        location="United States",
        sites=[JobSite.LINKEDIN],
        is_remote=True,
        job_type=None,
        results_wanted=40,
        save_to_db=False,
    )


@pytest.mark.asyncio
async def test_run_saved_search_records_failed_result(session, monkeypatch):
    search = _searches().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    monkeypatch.setattr(
        job_service,
        "search_and_save_jobs",
        AsyncMock(return_value=_result(success=False)),
    )

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.FAILED
    assert completed.last_error == "provider unavailable"


@pytest.mark.asyncio
async def test_failed_result_does_not_require_persistence_metadata(
    session,
    monkeypatch,
):
    search = _searches().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    failed_result = _result(success=False)
    monkeypatch.setattr(
        job_service,
        "search_and_save_jobs",
        AsyncMock(return_value=failed_result),
    )

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.FAILED
    assert completed.last_error == "provider unavailable"


@pytest.mark.asyncio
async def test_run_saved_search_records_partial_validation_loss(session, monkeypatch):
    search = _searches().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    monkeypatch.setattr(
        job_service,
        "search_and_save_jobs",
        AsyncMock(return_value=_result(found=2, inserted=1, invalid=1)),
    )

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.PARTIAL
    assert completed.jobs_seen == 2
    assert completed.jobs_new == 1
    assert completed.last_error == "1 of 2 provider rows failed validation"


@pytest.mark.asyncio
async def test_job_persistence_rolls_back_if_terminal_health_cannot_commit(
    session,
    monkeypatch,
):
    search = _searches().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    monkeypatch.setattr(
        job_service,
        "search_and_save_jobs",
        AsyncMock(return_value=_result(found=1, inserted=1)),
    )
    original_record_run = SavedSearchService.record_run
    failed_once = False

    def fail_terminal_once(
        service,
        search_id,
        health,
        *,
        expected_started_at=None,
        session=None,
    ):
        nonlocal failed_once
        if session is not None and not failed_once:
            failed_once = True
            raise RuntimeError("terminal write failed")
        return original_record_run(
            service,
            search_id,
            health,
            expected_started_at=expected_started_at,
            session=session,
        )

    monkeypatch.setattr(SavedSearchService, "record_run", fail_terminal_once)

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.FAILED
    assert completed.jobs_seen == 1
    assert completed.jobs_new == 0
    assert completed.last_error == "terminal write failed"
    session.expire_all()
    assert session.exec(select(JobSQL)).all() == []


@pytest.mark.asyncio
async def test_run_saved_search_records_raised_failure(session, monkeypatch):
    search = _searches().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    monkeypatch.setattr(
        job_service,
        "search_and_save_jobs",
        AsyncMock(side_effect=ConnectionError("offline")),
    )

    completed = await run_saved_search(search.id)

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.FAILED
    assert completed.last_error == "offline"


@pytest.mark.asyncio
async def test_run_saved_search_records_cancellation_and_reraises(session, monkeypatch):
    search = _searches().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    monkeypatch.setattr(
        job_service,
        "search_and_save_jobs",
        AsyncMock(side_effect=asyncio.CancelledError),
    )

    with pytest.raises(asyncio.CancelledError):
        await run_saved_search(search.id)

    recorded = _searches().get(search.id)
    assert recorded is not None
    assert recorded.last_run_status is SavedSearchRunStatus.CANCELLED
    assert recorded.last_error == "Run cancelled"


@pytest.mark.asyncio
async def test_run_saved_search_rejects_an_overlapping_run(session, monkeypatch):
    search = _searches().create(
        SavedSearchCreate(name="AI", query="AI", sites=[JobSite.LINKEDIN])
    )
    first_run_started = asyncio.Event()
    release_first_run = asyncio.Event()

    async def slow_search(**_kwargs):
        first_run_started.set()
        await release_first_run.wait()
        return _result(found=1, inserted=1)

    mocked = AsyncMock(side_effect=slow_search)
    monkeypatch.setattr(job_service, "search_and_save_jobs", mocked)

    first_run = asyncio.create_task(run_saved_search(search.id))
    await first_run_started.wait()
    with pytest.raises(SavedSearchRunInProgressError):
        await run_saved_search(search.id)
    release_first_run.set()
    completed = await first_run

    assert completed is not None
    assert completed.last_run_status is SavedSearchRunStatus.SUCCEEDED
    assert mocked.await_count == 1


@pytest.mark.asyncio
async def test_run_saved_search_returns_none_for_missing_definition(session):
    assert await run_saved_search(404) is None


@pytest.mark.asyncio
async def test_scrape_all_runs_only_enabled_searches(session, monkeypatch):
    searches = _searches()
    searches.create(SavedSearchCreate(name="A", query="AI"))
    searches.create(SavedSearchCreate(name="B", query="ML"))
    searches.create(SavedSearchCreate(name="Disabled", query="Data", enabled=False))
    mocked = AsyncMock(side_effect=[_result(found=2, inserted=2), _result(found=1)])
    monkeypatch.setattr(job_service, "search_and_save_jobs", mocked)

    results = await scrape_all()

    assert [result.name for result in results] == ["A", "B"]
    assert [result.jobs_new for result in results] == [2, 0]
    assert mocked.await_count == 2


@pytest.mark.asyncio
async def test_scrape_all_skips_an_already_running_search(session, monkeypatch):
    searches = _searches()
    running = searches.create(SavedSearchCreate(name="A", query="AI"))
    searches.create(SavedSearchCreate(name="B", query="ML"))
    assert searches.claim_run(running.id, datetime.now(UTC)) is not None
    mocked = AsyncMock(return_value=_result(found=1, inserted=1))
    monkeypatch.setattr(job_service, "search_and_save_jobs", mocked)

    results = await scrape_all()

    assert [result.name for result in results] == ["B"]
    assert mocked.await_count == 1


@pytest.mark.asyncio
async def test_scrape_all_without_enabled_searches_is_noop(session, monkeypatch):
    _searches().create(SavedSearchCreate(name="Disabled", query="AI", enabled=False))
    mocked = AsyncMock()
    monkeypatch.setattr(job_service, "search_and_save_jobs", mocked)

    assert await scrape_all() == []
    mocked.assert_not_awaited()


def test_sync_wrapper_returns_same_health_contract(session, monkeypatch):
    _searches().create(SavedSearchCreate(name="AI", query="AI"))
    monkeypatch.setattr(
        job_service,
        "search_and_save_jobs",
        AsyncMock(return_value=_result(found=4, inserted=1)),
    )

    results = scrape_all_sync()

    assert len(results) == 1
    assert results[0].last_run_status is SavedSearchRunStatus.SUCCEEDED
    assert results[0].jobs_seen == 4
    assert results[0].jobs_new == 1
