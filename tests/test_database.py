"""Tests for database operations and integration."""

from datetime import datetime, timedelta, timezone

import pytest

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select
from src.models import CompanySQL, JobSQL


def test_database_connection(session: Session):
    """Test database connection."""
    result = session.exec(select(1))
    assert result.first() == 1


def test_company_crud_operations(session: Session):
    """Test CRUD for companies."""
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
    """Test CRUD for jobs."""
    job = JobSQL(
        company="CRUD Co",
        title="CRUD Job",
        description="Test desc",
        link="https://crud.co/job",
        location="Remote",
        posted_date=datetime.now(timezone.utc),
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
    """Test job filtering queries."""
    now = datetime.now(timezone.utc)
    yesterday = now - timedelta(days=1)

    jobs = [
        JobSQL(
            company="A",
            title="AI Eng",
            description="AI",
            link="a1",
            location="SF",
            posted_date=now,
            salary=(None, None),
        ),
        JobSQL(
            company="B",
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

    company_a = (session.exec(select(JobSQL).where(JobSQL.company == "A"))).all()
    assert len(company_a) == 1

    recent = (session.exec(select(JobSQL).where(JobSQL.posted_date >= yesterday))).all()
    assert len(recent) == 2


def test_database_constraints(session: Session):
    """Test database integrity constraints."""
    company1 = CompanySQL(name="Const Co", url="https://const1.co", active=True)
    session.add(company1)
    session.commit()

    company2 = CompanySQL(name="Const Co", url="https://const2.co", active=False)
    session.add(company2)
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()

    job1 = JobSQL(
        company="Const Co",
        title="Job1",
        description="Desc",
        link="https://const.co/job",
        location="Loc",
        salary=(None, None),
    )
    session.add(job1)
    session.commit()

    job2 = JobSQL(
        company="Const Co",
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
    """Test transaction rollback."""
    company = CompanySQL(name="Rollback Co", url="https://rollback.co", active=True)
    session.add(company)
    session.commit()

    try:
        job = JobSQL(
            company="Rollback Co",
            title="Rollback Job",
            description="Desc",
            link="https://rollback.co/job",
            location="Loc",
            salary=(None, None),
        )
        session.add(job)

        invalid_job = JobSQL(
            company="Rollback Co",
            title="Invalid",
            description="Invalid",
            link="https://rollback.co/job",
            location="Invalid",
            salary=(None, None),
        )
        session.add(invalid_job)
        session.commit()  # Fails
    except Exception:
        session.rollback()

    jobs = (session.exec(select(JobSQL).where(JobSQL.company == "Rollback Co"))).all()
    assert len(jobs) == 0

    companies = (
        session.exec(select(CompanySQL).where(CompanySQL.name == "Rollback Co"))
    ).all()
    assert len(companies) == 1
