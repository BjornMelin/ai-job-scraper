"""Behavior tests for the Job Tracker information architecture."""

from datetime import UTC, datetime
from pathlib import Path

from sqlmodel import Session, select
from src.database_models import JobSQL
from src.models.job_models import (
    ApplicationStage,
    JobSite,
    SavedSearchRunStatus,
)
from src.schemas import SavedSearchCreate, SavedSearchRunHealth
from src.services.saved_search_service import saved_search_service

from tests.factories import CompanyFactory, JobFactory


def _seed_job(session: Session) -> JobSQL:
    company = CompanyFactory(name="Acme Research")
    job = JobFactory(
        company_id=company.id,
        title="Machine Learning Engineer",
        description="Build production learning systems.",
        application_status=ApplicationStage.INBOX,
    )
    session.commit()
    return job


def test_jobs_empty_state_is_actionable(jobs_app_test, session: Session) -> None:
    app = jobs_app_test.run()

    assert app.exception == []
    assert app.title[0].value == "Jobs"
    assert "Nothing here yet" in [heading.value for heading in app.subheader]
    assert "Run a saved search" in app.markdown[-1].value


def test_jobs_expose_canonical_filters_and_company_facets(
    jobs_app_test,
    session: Session,
) -> None:
    _seed_job(session)

    app = jobs_app_test.run()

    assert app.exception == []
    assert app.radio[0].label == "Workflow stage"
    assert app.radio[0].options == [
        "Inbox  1",
        "Saved  0",
        "Applied  0",
        "Interviews  0",
        "Closed  0",
    ]
    assert app.text_input[0].label == "Search jobs"
    assert app.multiselect[0].options == ["Acme Research"]
    assert app.subheader[-1].value == "Machine Learning Engineer"


def test_job_review_form_persists_stage_star_and_notes(
    jobs_app_test,
    session: Session,
) -> None:
    job = _seed_job(session)
    app = jobs_app_test.run()

    next(item for item in app.selectbox if item.label == "Stage").set_value(
        ApplicationStage.APPLIED
    )
    next(item for item in app.checkbox if item.label == "Starred").check()
    app.text_area[0].set_value("Follow up Thursday")
    next(item for item in app.button if item.label == "Save changes").click().run()

    session.expire_all()
    updated = session.exec(select(JobSQL).where(JobSQL.id == job.id)).one()
    assert updated.application_status == ApplicationStage.APPLIED
    assert updated.favorite is True
    assert updated.notes == "Follow up Thursday"
    assert updated.application_date is not None
    assert app.success[0].value == "Job updated."
    assert "Nothing here yet" in [heading.value for heading in app.subheader]


def test_recovered_job_does_not_render_a_broken_original_link(
    jobs_app_test,
    session: Session,
) -> None:
    job = _seed_job(session)
    job.link = f"legacy://recovered-job/{job.id}"
    session.add(job)
    session.commit()

    app = jobs_app_test.run()

    assert app.exception == []
    assert not any("Open original posting" in item.value for item in app.markdown)
    assert any("Original posting unavailable" in item.value for item in app.caption)


def test_jobs_show_database_errors_without_crashing(
    jobs_app_test,
    session: Session,
    monkeypatch,
) -> None:
    from src.services.job_service import JobService

    def fail() -> dict[str, int]:
        raise RuntimeError("database offline")

    monkeypatch.setattr(JobService, "get_job_counts_by_status", staticmethod(fail))
    app = jobs_app_test.run()

    assert app.exception == []
    assert app.error[0].value == (
        "Jobs could not be loaded. Check the database and try again."
    )


def test_jobs_paginate_dense_results(jobs_app_test, session: Session) -> None:
    company = CompanyFactory(name="Acme Research")
    JobFactory.create_batch(30, company_id=company.id)
    session.commit()

    app = jobs_app_test.run()

    assert app.exception == []
    assert app.selectbox(key="jobs-per-page").value == 25
    assert (
        len([item for item in app.subheader if item.value.startswith("Software")]) == 25
    )
    assert app.caption[0].value == "Showing 1 to 25 of 30 jobs"


def test_jobs_reset_an_out_of_range_page_after_filtering(
    jobs_app_test,
    session: Session,
) -> None:
    company = CompanyFactory(name="Acme Research")
    jobs = JobFactory.create_batch(30, company_id=company.id)
    jobs[0].title = "Needle Role"
    session.commit()

    app = jobs_app_test.run()
    app.selectbox(key="jobs-page").set_value(2).run()
    app.text_input(key="jobs-query").set_value("Needle").run()

    assert app.exception == []
    assert app.selectbox(key="jobs-page").value == 1
    assert app.caption[0].value == "Showing 1 to 1 of 1 jobs"
    assert app.subheader[-1].value == "Needle Role"


def test_jobs_can_reach_results_beyond_the_former_thousand_job_cap(
    jobs_app_test,
    session: Session,
) -> None:
    company = CompanyFactory(name="Acme Research")
    JobFactory.create_batch(1001, company_id=company.id)
    session.commit()

    app = jobs_app_test.run()

    assert app.caption[0].value == "Showing 1 to 25 of 1001 jobs"
    app.selectbox(key="jobs-page").set_value(41).run()
    assert app.exception == []
    assert app.caption[0].value == "Showing 1001 to 1001 of 1001 jobs"
    assert (
        len([item for item in app.subheader if item.value.startswith("Software")]) == 1
    )


def test_saved_search_can_be_created_from_the_only_run_configuration(
    searches_app_test,
    session: Session,
) -> None:
    app = searches_app_test.run()
    app.text_input[0].set_value("Remote ML")
    app.text_input[1].set_value("machine learning engineer")
    app.multiselect[0].set_value([JobSite.LINKEDIN])
    app.button[0].click().run()

    assert app.exception == []
    assert app.success[0].value == "Saved search created."
    created = saved_search_service.list()
    assert len(created) == 1
    assert created[0].name == "Remote ML"
    assert created[0].sites == [JobSite.LINKEDIN]


def test_saved_search_requires_at_least_one_job_board(
    searches_app_test,
    session: Session,
) -> None:
    app = searches_app_test.run()
    app.text_input[0].set_value("Remote ML")
    app.text_input[1].set_value("machine learning engineer")
    app.multiselect[0].set_value([])
    app.button[0].click().run()

    assert app.exception == []
    assert app.error[0].value == "At least one job board is required."
    assert saved_search_service.list() == []


def test_invalid_saved_search_stays_open_with_feedback(
    searches_app_test,
    session: Session,
) -> None:
    app = searches_app_test.run()
    app.button[0].click().run()

    assert app.exception == []
    assert app.error[0].value == "Name and keywords are required."
    assert app.text_input(key="create-search-name").value == ""
    assert app.text_input(key="create-search-query").value == ""
    assert saved_search_service.list() == []


def test_failed_saved_search_toggle_uses_persisted_enabled_state(
    searches_app_test,
    session: Session,
    monkeypatch,
) -> None:
    saved_search_service.create(
        SavedSearchCreate(name="Remote ML", query="ML", enabled=False)
    )
    app = searches_app_test.run()

    def fail_update(*_args, **_kwargs):
        raise RuntimeError("database offline")

    monkeypatch.setattr(saved_search_service, "update", fail_update)
    app.checkbox(key="enabled-1").check().run()

    assert app.exception == []
    assert app.error[0].value == "The saved search could not be changed. Try again."
    assert next(item for item in app.button if item.label == "Run now").disabled
    assert saved_search_service.list()[0].enabled is False


def test_saved_search_delete_removes_the_stale_card(
    searches_app_test,
    session: Session,
) -> None:
    saved_search_service.create(SavedSearchCreate(name="Remote ML", query="ML"))
    app = searches_app_test.run()

    next(
        item for item in app.button if item.label == "Delete permanently"
    ).click().run()

    assert app.exception == []
    assert app.success[0].value == "Saved search deleted."
    assert saved_search_service.list() == []
    assert "Create your first search" in [heading.value for heading in app.subheader]


def test_saved_search_run_reports_success(
    searches_app_test,
    session: Session,
    monkeypatch,
) -> None:
    search = saved_search_service.create(
        SavedSearchCreate(name="Remote ML", query="machine learning engineer")
    )

    async def complete(search_id: int):
        return saved_search_service.record_run(
            search_id,
            SavedSearchRunHealth(
                last_run_at=datetime.now(UTC),
                last_run_status=SavedSearchRunStatus.SUCCEEDED,
                jobs_seen=7,
                jobs_new=3,
                duration_ms=20,
            ),
        )

    monkeypatch.setattr("src.scraping.scrape_all.run_saved_search", complete)
    app = searches_app_test.run()
    next(item for item in app.button if item.label == "Run now").click().run()

    assert app.exception == []
    assert app.status[0].state == "complete"
    assert app.success[0].value == "Found 7 jobs; 3 were new."
    assert any("Last run:** Succeeded" in item.value for item in app.markdown)
    completed_search = saved_search_service.get(search.id)
    assert completed_search is not None
    assert completed_search.jobs_new == 3


def test_saved_search_run_reports_failure(
    searches_app_test,
    session: Session,
    monkeypatch,
) -> None:
    saved_search_service.create(SavedSearchCreate(name="Remote ML", query="ML"))

    async def fail(search_id: int):
        return saved_search_service.record_run(
            search_id,
            SavedSearchRunHealth(
                last_run_at=datetime.now(UTC),
                last_run_status=SavedSearchRunStatus.FAILED,
                last_error="Job board unavailable",
            ),
        )

    monkeypatch.setattr("src.scraping.scrape_all.run_saved_search", fail)
    app = searches_app_test.run()
    next(item for item in app.button if item.label == "Run now").click().run()

    assert app.exception == []
    assert app.status[0].state == "error"
    assert app.error[0].value == "Job board unavailable"


def test_saved_search_run_reports_partial_validation_loss(
    searches_app_test,
    session: Session,
    monkeypatch,
) -> None:
    saved_search_service.create(SavedSearchCreate(name="Remote ML", query="ML"))

    async def finish_partially(search_id: int):
        return saved_search_service.record_run(
            search_id,
            SavedSearchRunHealth(
                last_run_at=datetime.now(UTC),
                last_run_status=SavedSearchRunStatus.PARTIAL,
                jobs_seen=7,
                jobs_new=3,
                duration_ms=20,
                last_error="1 of 7 provider rows failed validation",
            ),
        )

    monkeypatch.setattr(
        "src.scraping.scrape_all.run_saved_search",
        finish_partially,
    )
    app = searches_app_test.run()
    next(item for item in app.button if item.label == "Run now").click().run()

    assert app.exception == []
    assert app.status[0].state == "complete"
    assert app.warning[0].value == "1 of 7 provider rows failed validation"
    assert app.error == []
    assert any("Last run:** Partial" in item.value for item in app.markdown)


def test_insights_are_read_only_and_derived(
    insights_app_test,
    session: Session,
) -> None:
    _seed_job(session)
    app = insights_app_test.run()

    assert app.exception == []
    assert app.title[0].value == "Insights"
    assert [metric.label for metric in app.metric[:3]] == [
        "Tracked jobs",
        "Companies",
        "Saved searches",
    ]
    assert app.dataframe[0].value.iloc[0]["Company"] == "Acme Research"
    assert "cannot be edited" in app.caption[-1].value
    assert all("company" not in button.label.lower() for button in app.button)


def test_shell_has_three_labeled_page_links_and_named_routes(
    app_test,
    session: Session,
) -> None:
    app = app_test.from_file("src/main.py").run()
    main = Path("src/main.py").read_text()
    project = Path("pyproject.toml").read_text()
    pages = Path("src/ui/pages")
    page_links = app.get("page_link")

    assert app.exception == []
    assert [(link.label, link.proto.page) for link in page_links] == [
        ("Jobs", "jobs"),
        ("Searches", "searches"),
        ("Insights", "insights"),
    ]
    assert 'page_title="Job Tracker"' in main
    assert 'position="hidden"' in main
    assert 'position="top"' not in main
    assert sorted(path.name for path in pages.glob("*.py")) == [
        "__init__.py",
        "insights.py",
        "jobs.py",
        "searches.py",
    ]
    assert not Path("src/services/cache_manager.py").exists()
    assert not Path("examples/search_integration_example.py").exists()
    assert '"streamlit>=1.59.2,<2.0.0"' in project


def test_design_has_keyboard_focus_responsive_and_reduced_motion_guards() -> None:
    design = Path("src/ui/design.py").read_text()

    assert ":focus-visible" in design
    assert "outline: 3px solid var(--focus)" in design
    assert "button p { color: inherit !important; }" in design
    assert "color-mix(in srgb, var(--focus)" not in design
    assert "@media (max-width: 700px)" in design
    assert "@media (prefers-reduced-motion: reduce)" in design
    assert 'data-testid="stNumberInputStepDown"' in design
    assert "gradient" not in design
    assert "backdrop-filter" not in design
