"""Saved-search CRUD and run-health contracts."""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from sqlmodel import Session, select
from src.database_models import CompanySQL, JobSQL, SavedSearchSQL
from src.models.job_models import JobSite, SavedSearchRunStatus
from src.schemas import SavedSearchCreate, SavedSearchRunHealth, SavedSearchUpdate
from src.services.saved_search_service import SavedSearchService


def test_saved_search_crud_and_health(session: Session) -> None:
    service = SavedSearchService()
    created = service.create(
        SavedSearchCreate(
            name="Remote ML",
            query="machine learning engineer",
            sites=[JobSite.LINKEDIN],
            remote_only=True,
        )
    )
    assert created.last_run_status is SavedSearchRunStatus.NEVER

    updated = service.update(
        created.id,
        SavedSearchUpdate(results_limit=75, enabled=False),
    )
    assert updated is not None
    assert updated.results_limit == 75
    assert updated.enabled is False

    reported_at = datetime(2026, 7, 14, 12, tzinfo=UTC)
    completed_at = reported_at + timedelta(seconds=5)
    with patch("src.services.saved_search_service.datetime") as clock:
        clock.now.return_value = completed_at
        finished = service.record_run(
            created.id,
            SavedSearchRunHealth(
                last_run_at=reported_at,
                last_run_status=SavedSearchRunStatus.SUCCEEDED,
                jobs_seen=12,
                jobs_new=4,
                duration_ms=250,
            ),
        )
    assert finished is not None
    assert finished.jobs_seen == 12
    assert finished.jobs_new == 4
    assert finished.last_run_status is SavedSearchRunStatus.SUCCEEDED
    assert finished.last_run_at == completed_at


def test_deleting_search_preserves_jobs(session: Session) -> None:
    search = SavedSearchService().create(
        SavedSearchCreate(name="Disposable", query="data engineer")
    )
    company = CompanySQL(name="Keep Jobs", url=None)
    session.add(company)
    session.flush()
    job = JobSQL.create_validated(
        company_id=company.id,
        title="Data Engineer",
        description="Keep this",
        link="https://keep.example/jobs/1",
        location="Remote",
    )
    session.add(job)
    session.commit()

    assert SavedSearchService().delete(search.id) is True
    assert session.exec(select(SavedSearchSQL)).all() == []
    assert session.exec(select(JobSQL)).all() == [job]


def test_run_claim_is_exclusive_and_old_run_cannot_overwrite_reclaim(
    session: Session,
) -> None:
    service = SavedSearchService()
    search = service.create(SavedSearchCreate(name="Remote ML", query="ML"))
    old_started_at = datetime.now(UTC) - timedelta(hours=1)
    current_started_at = datetime.now(UTC)

    assert service.claim_run(search.id, old_started_at) is not None
    assert service.claim_run(search.id, old_started_at + timedelta(seconds=1)) is None
    assert service.claim_run(search.id, current_started_at) is not None

    stale_completion = service.record_run(
        search.id,
        SavedSearchRunHealth(
            last_run_at=old_started_at,
            last_run_status=SavedSearchRunStatus.SUCCEEDED,
        ),
        expected_started_at=old_started_at,
    )
    current = service.get(search.id)

    assert stale_completion is None
    assert current is not None
    assert current.last_run_status is SavedSearchRunStatus.RUNNING
    assert current.last_run_at == current_started_at


@pytest.mark.parametrize("field", ["name", "query", "location"])
def test_saved_search_updates_reject_blank_text(field: str) -> None:
    with pytest.raises(ValidationError):
        SavedSearchUpdate.model_validate({field: "   "})
