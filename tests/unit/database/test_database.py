"""Tests for database operations and integration.

This module contains comprehensive tests for database functionality including:
- Basic connection testing
- CRUD operations for companies and jobs
- Database constraints and integrity testing
- Transaction rollback testing
- Query filtering and data retrieval
"""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select
from src.config import DatabaseURLError
from src.database import _create_engine_impl, get_engine
from src.database_models import CompanySQL, JobSQL


@pytest.mark.parametrize("factory", [_create_engine_impl, get_engine])
@pytest.mark.parametrize(
    "database_url",
    [
        "postgresql://localhost/jobs",
        "sqlite:garbage",
        "sqlite://host/jobs.db",
        "sqlite://host:invalid/jobs.db",
    ],
)
def test_engine_factories_enforce_sqlite_url_contract(factory, database_url):
    """Explicit URLs cannot bypass the app's SQLite-only contract."""
    with pytest.raises(DatabaseURLError):
        factory(database_url)


def test_database_connection(session: Session):
    """Test basic database connection functionality.

    Verifies that the database session can execute a simple query
    and return expected results.
    """
    result = session.exec(select(1))
    assert result.first() == 1


def test_company_crud_operations(session: Session):
    """Test Create, Read, Update, Delete operations for companies.

    Validates that companies can be properly created, retrieved,
    updated, and deleted from the database with correct data persistence.
    """
    company = CompanySQL(name="CRUD Co", url="https://crud.co")
    session.add(company)
    session.commit()
    session.refresh(company)

    retrieved = (
        session.exec(select(CompanySQL).where(CompanySQL.name == "CRUD Co"))
    ).first()
    assert retrieved.url == "https://crud.co"

    retrieved.url = "https://crud.co/jobs"
    session.commit()

    updated = (
        session.exec(select(CompanySQL).where(CompanySQL.name == "CRUD Co"))
    ).first()
    assert updated.url == "https://crud.co/jobs"

    session.delete(updated)
    session.commit()

    deleted = (
        session.exec(select(CompanySQL).where(CompanySQL.name == "CRUD Co"))
    ).first()
    assert deleted is None


def test_job_crud_operations(session: Session):
    """Test Create, Read, Update, Delete operations for jobs.

    Validates that jobs can be properly created, retrieved, updated,
    and deleted from the database with proper field handling including
    salary tuples and user-specific fields like favorites and notes.
    """
    company = CompanySQL(name="CRUD Job Co", url=None)
    session.add(company)
    session.flush()
    job = JobSQL.create_validated(
        company_id=company.id,
        title="CRUD Job",
        description="Test desc",
        link="https://crud.co/job",
        location="Remote",
        posted_date=datetime.now(UTC),
        salary=(100000, 150000),
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    retrieved = (session.exec(select(JobSQL).where(JobSQL.title == "CRUD Job"))).first()
    assert retrieved.location == "Remote"

    retrieved.favorite = True
    retrieved.notes = "Updated"
    session.commit()

    updated = (session.exec(select(JobSQL).where(JobSQL.title == "CRUD Job"))).first()
    assert updated.favorite is True

    session.delete(updated)
    session.commit()

    deleted = (session.exec(select(JobSQL).where(JobSQL.title == "CRUD Job"))).first()
    assert deleted is None


def test_job_filtering_queries(session: Session):
    """Test database query filtering capabilities.

    Creates sample jobs with different attributes and tests
    filtering by location and date ranges to ensure
    query operations work correctly.
    """
    now = datetime.now(UTC)
    yesterday = now - timedelta(days=1)
    company = CompanySQL(name="Filter Co", url=None)
    session.add(company)
    session.flush()

    jobs = [
        JobSQL.create_validated(
            company_id=company.id,
            title="AI Eng",
            description="AI",
            link="a1",
            location="SF",
            posted_date=now,
            salary=(None, None),
        ),
        JobSQL.create_validated(
            company_id=company.id,
            title="ML Eng",
            description="ML",
            link="b1",
            location="Remote",
            posted_date=yesterday,
            salary=(None, None),
        ),
    ]
    session.add_all(jobs)
    session.commit()

    sf_jobs = (session.exec(select(JobSQL).where(JobSQL.location == "SF"))).all()
    assert len(sf_jobs) == 1

    recent = (session.exec(select(JobSQL).where(JobSQL.posted_date >= yesterday))).all()
    assert len(recent) == 2


def test_database_constraints(session: Session):
    """Test database integrity constraints and unique field validation.

    Verifies that unique constraints are properly enforced for:
    - Company names (must be unique)
    - Job links (must be unique)
    Ensures IntegrityError is raised when constraints are violated.
    """
    company1 = CompanySQL(name="Const Co", url="https://const1.co")
    session.add(company1)
    session.commit()

    company2 = CompanySQL(name="Const Co", url="https://const2.co")
    session.add(company2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    job1 = JobSQL.create_validated(
        company_id=company1.id,
        title="Job1",
        description="Desc",
        link="https://const.co/job",
        location="Loc",
        salary=(None, None),
    )
    session.add(job1)
    session.commit()

    job2 = JobSQL.create_validated(
        company_id=company1.id,
        title="Job2",
        description="Desc2",
        link="https://const.co/job",
        location="Loc2",
        salary=(None, None),
    )
    session.add(job2)
    with pytest.raises(IntegrityError):
        session.commit()


def test_database_rollback(session: Session):
    """Test transaction rollback functionality using savepoints.

    Creates a company, then uses a savepoint to test that failed
    transactions are properly rolled back without affecting
    previously committed data within the same session.
    """
    # Add and commit company first
    company = CompanySQL(name="Rollback Co", url="https://rollback.co")
    session.add(company)
    session.commit()
    session.refresh(company)  # Ensure company.id is available

    # Verify company was added
    companies_before = session.exec(
        select(CompanySQL).where(CompanySQL.name == "Rollback Co")
    ).all()
    assert len(companies_before) == 1

    # Create savepoint for rollback testing
    savepoint = session.begin_nested()

    try:
        # Add jobs that should be rolled back
        job = JobSQL.create_validated(
            title="Rollback Job",
            description="Desc",
            link="https://rollback.co/job",
            location="Loc",
            salary=(None, None),
            company_id=company.id,
        )
        session.add(job)
        session.flush()  # Make sure job is in session but not committed

        # Verify job is in session before rollback
        jobs_before_rollback = session.exec(
            select(JobSQL).where(JobSQL.link.contains("rollback.co"))
        ).all()
        assert len(jobs_before_rollback) == 1

        # Force rollback of the savepoint
        savepoint.rollback()

    except Exception:
        savepoint.rollback()

    # After rollback, jobs should be gone but company should remain
    jobs_after_rollback = session.exec(
        select(JobSQL).where(JobSQL.link.contains("rollback.co"))
    ).all()
    assert len(jobs_after_rollback) == 0

    companies_after_rollback = session.exec(
        select(CompanySQL).where(CompanySQL.name == "Rollback Co")
    ).all()
    assert len(companies_after_rollback) == 1
