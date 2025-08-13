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

from src.models import CompanySQL, JobSQL


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
    company = CompanySQL(name="CRUD Co", url="https://crud.co", active=True)
    session.add(company)
    session.commit()
    session.refresh(company)

    retrieved = (
        session.exec(select(CompanySQL).where(CompanySQL.name == "CRUD Co"))
    ).first()
    assert retrieved.url == "https://crud.co"

    retrieved.active = False
    session.commit()

    updated = (
        session.exec(select(CompanySQL).where(CompanySQL.name == "CRUD Co"))
    ).first()
    assert updated.active is False

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
    job = JobSQL.create_validated(
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

    jobs = [
        JobSQL.create_validated(
            title="AI Eng",
            description="AI",
            link="a1",
            location="SF",
            posted_date=now,
            salary=(None, None),
        ),
        JobSQL.create_validated(
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
    company1 = CompanySQL(name="Const Co", url="https://const1.co", active=True)
    session.add(company1)
    session.commit()

    company2 = CompanySQL(name="Const Co", url="https://const2.co", active=False)
    session.add(company2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    job1 = JobSQL.create_validated(
        title="Job1",
        description="Desc",
        link="https://const.co/job",
        location="Loc",
        salary=(None, None),
    )
    session.add(job1)
    session.commit()

    job2 = JobSQL.create_validated(
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
    """Test transaction rollback functionality.

    Creates a company, then attempts to insert jobs that violate
    constraints. Verifies that failed transactions are properly
    rolled back without affecting previously committed data.
    """
    company = CompanySQL(name="Rollback Co", url="https://rollback.co", active=True)
    session.add(company)
    session.commit()

    try:
        # Ensure autoflush is disabled so both jobs are added together
        session.autoflush = False

        job = JobSQL.create_validated(
            title="Rollback Job",
            description="Desc",
            link="https://rollback.co/job",
            location="Loc",
            salary=(None, None),
            company_id=company.id,  # Set company_id
        )
        session.add(job)

        invalid_job = JobSQL.create_validated(
            title="Invalid",
            description="Invalid",
            link="https://rollback.co/job",  # Duplicate link will
            # cause constraint violation
            location="Invalid",
            salary=(None, None),
            company_id=company.id,  # Set company_id
        )
        session.add(invalid_job)

        # Force an exception to simulate rollback scenario
        raise RuntimeError("Database Transaction failed in unit test")
    except Exception:
        session.rollback()
    finally:
        session.autoflush = True  # Restore autoflush

    jobs = (
        session.exec(select(JobSQL).where(JobSQL.link.contains("rollback.co")))
    ).all()
    assert len(jobs) == 0

    companies = (
        session.exec(select(CompanySQL).where(CompanySQL.name == "Rollback Co"))
    ).all()
    assert len(companies) == 1
