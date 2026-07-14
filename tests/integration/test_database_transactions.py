"""Integration coverage for the canonical database owner."""

from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select
from src.database import (
    _create_engine_impl,
    create_db_and_tables,
    db_session,
    db_session_no_autocommit,
    get_connection_pool_status,
)
from src.database_models import CompanySQL, JobSQL, SavedSearchSQL


def test_db_session_commits_on_success(test_engine) -> None:
    with db_session(test_engine) as session:
        session.add(CompanySQL(name="Committed", url=None))

    with Session(test_engine) as session:
        assert session.exec(select(CompanySQL.name)).all() == ["Committed"]


def test_db_session_rolls_back_on_failure(test_engine) -> None:
    with (
        pytest.raises(ValueError, match="rollback"),
        db_session(test_engine) as session,
    ):
        session.add(CompanySQL(name="Rolled back", url=None))
        raise ValueError("rollback")

    with Session(test_engine) as session:
        assert session.exec(select(CompanySQL)).all() == []


def test_no_autocommit_rolls_back_uncommitted_work(test_engine) -> None:
    with db_session_no_autocommit(test_engine) as session:
        session.add(CompanySQL(name="Transient", url=None))
        session.flush()

    with Session(test_engine) as session:
        assert session.exec(select(CompanySQL)).all() == []


def test_no_autocommit_preserves_explicit_commit(test_engine) -> None:
    with db_session_no_autocommit(test_engine) as session:
        session.add(CompanySQL(name="Durable", url=None))
        session.commit()

    with Session(test_engine) as session:
        assert session.exec(select(CompanySQL.name)).all() == ["Durable"]


def test_foreign_keys_are_enforced(test_engine) -> None:
    with pytest.raises(IntegrityError), db_session(test_engine) as session:
        session.add(
            JobSQL.create_validated(
                company_id=999,
                title="Invalid owner",
                description="Should roll back",
                link="https://example.com/jobs/invalid-owner",
                location="Remote",
            )
        )


def test_blank_job_links_are_rejected(test_engine) -> None:
    with pytest.raises(ValidationError):
        JobSQL(
            company_id=1,
            title="Invalid link",
            description="Should roll back",
            link="",
            location="Remote",
        )


def test_blank_company_names_are_rejected_by_database(test_engine) -> None:
    with pytest.raises(IntegrityError), db_session(test_engine) as session:
        session.execute(CompanySQL.__table__.insert().values(name="   ", url=None))


def test_blank_job_titles_are_rejected_by_database(test_engine) -> None:
    with db_session(test_engine) as session:
        company = CompanySQL(name="Valid owner", url=None)
        session.add(company)
        session.flush()
        company_id = company.id

    with pytest.raises(IntegrityError), db_session(test_engine) as session:
        session.execute(
            JobSQL.__table__.insert().values(
                company_id=company_id,
                title="   ",
                description="Invalid title",
                link="https://example.com/jobs/invalid-title",
                location="Remote",
            )
        )


def test_negative_saved_search_health_is_rejected_by_database(test_engine) -> None:
    with pytest.raises(IntegrityError), db_session(test_engine) as session:
        session.execute(
            SavedSearchSQL.__table__.insert().values(
                name="Invalid health",
                query="data engineer",
                location="Remote",
                sites=["linkedin"],
                remote_only=True,
                results_limit=50,
                enabled=True,
                jobs_seen=-1,
            )
        )


def test_create_tables_targets_explicit_engine(tmp_path: Path) -> None:
    engine = _create_engine_impl(f"sqlite:///{tmp_path / 'schema.db'}")
    try:
        create_db_and_tables(engine)
        with engine.connect() as connection:
            tables = set(
                connection.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).scalars()
            )
        assert {"companysql", "jobsql", "savedsearchsql"} <= tables
    finally:
        engine.dispose()


def test_pool_status_targets_explicit_engine(test_engine) -> None:
    status = get_connection_pool_status(test_engine)
    assert status["pool_type"] == type(test_engine.pool).__name__
    assert status["engine_url"] == test_engine.url.render_as_string(hide_password=True)


def test_sqlite_pragmas_apply_to_connections(test_engine) -> None:
    with test_engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
