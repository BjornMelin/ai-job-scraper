"""Tests for database operations and integration."""

import datetime

import pytest

from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from src.models import CompanySQL, JobSQL


@pytest.mark.asyncio
async def test_database_connection(temp_db: AsyncSession):
    """Test async database connection."""
    result = await temp_db.exec(select(1))
    assert result.first() == 1


@pytest.mark.asyncio
async def test_company_crud_operations(temp_db: AsyncSession):
    """Test async CRUD for companies."""
    company = CompanySQL(name="CRUD Co", url="https://crud.co", active=True)
    temp_db.add(company)
    await temp_db.commit()
    await temp_db.refresh(company)

    retrieved = (
        await temp_db.exec(select(CompanySQL).where(CompanySQL.name == "CRUD Co"))
    ).first()
    assert retrieved.url == "https://crud.co"

    retrieved.active = False
    await temp_db.commit()

    updated = (
        await temp_db.exec(select(CompanySQL).where(CompanySQL.name == "CRUD Co"))
    ).first()
    assert updated.active is False

    await temp_db.delete(updated)
    await temp_db.commit()

    deleted = (
        await temp_db.exec(select(CompanySQL).where(CompanySQL.name == "CRUD Co"))
    ).first()
    assert deleted is None


@pytest.mark.asyncio
async def test_job_crud_operations(temp_db: AsyncSession):
    """Test async CRUD for jobs."""
    job = JobSQL(
        company="CRUD Co",
        title="CRUD Job",
        description="Test desc",
        link="https://crud.co/job",
        location="Remote",
        posted_date=datetime.datetime.now(),
        salary=(100000, 150000),
    )
    temp_db.add(job)
    await temp_db.commit()
    await temp_db.refresh(job)

    retrieved = (
        await temp_db.exec(select(JobSQL).where(JobSQL.title == "CRUD Job"))
    ).first()
    assert retrieved.location == "Remote"

    retrieved.favorite = True
    retrieved.notes = "Updated"
    await temp_db.commit()

    updated = (
        await temp_db.exec(select(JobSQL).where(JobSQL.title == "CRUD Job"))
    ).first()
    assert updated.favorite is True

    await temp_db.delete(updated)
    await temp_db.commit()

    deleted = (
        await temp_db.exec(select(JobSQL).where(JobSQL.title == "CRUD Job"))
    ).first()
    assert deleted is None


@pytest.mark.asyncio
async def test_job_filtering_queries(temp_db: AsyncSession):
    """Test async job filtering queries."""
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)

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
    temp_db.add_all(jobs)
    await temp_db.commit()

    company_a = (await temp_db.exec(select(JobSQL).where(JobSQL.company == "A"))).all()
    assert len(company_a) == 1

    recent = (
        await temp_db.exec(select(JobSQL).where(JobSQL.posted_date >= yesterday))
    ).all()
    assert len(recent) == 2


@pytest.mark.asyncio
async def test_database_constraints(temp_db: AsyncSession):
    """Test database integrity constraints async."""
    company1 = CompanySQL(name="Const Co", url="https://const1.co", active=True)
    temp_db.add(company1)
    await temp_db.commit()

    company2 = CompanySQL(name="Const Co", url="https://const2.co", active=False)
    temp_db.add(company2)
    with pytest.raises(IntegrityError):
        await temp_db.commit()
    await temp_db.rollback()

    job1 = JobSQL(
        company="Const Co",
        title="Job1",
        description="Desc",
        link="https://const.co/job",
        location="Loc",
        salary=(None, None),
    )
    temp_db.add(job1)
    await temp_db.commit()

    job2 = JobSQL(
        company="Const Co",
        title="Job2",
        description="Desc2",
        link="https://const.co/job",
        location="Loc2",
        salary=(None, None),
    )
    temp_db.add(job2)
    with pytest.raises(IntegrityError):
        await temp_db.commit()


@pytest.mark.asyncio
async def test_database_rollback(temp_db: AsyncSession):
    """Test async transaction rollback."""
    company = CompanySQL(name="Rollback Co", url="https://rollback.co", active=True)
    temp_db.add(company)
    await temp_db.commit()

    try:
        job = JobSQL(
            company="Rollback Co",
            title="Rollback Job",
            description="Desc",
            link="https://rollback.co/job",
            location="Loc",
            salary=(None, None),
        )
        temp_db.add(job)

        invalid_job = JobSQL(
            company="Rollback Co",
            title="Invalid",
            description="Invalid",
            link="https://rollback.co/job",
            location="Invalid",
            salary=(None, None),
        )
        temp_db.add(invalid_job)
        await temp_db.commit()  # Fails
    except Exception:
        await temp_db.rollback()

    jobs = (
        await temp_db.exec(select(JobSQL).where(JobSQL.company == "Rollback Co"))
    ).all()
    assert len(jobs) == 0

    companies = (
        await temp_db.exec(select(CompanySQL).where(CompanySQL.name == "Rollback Co"))
    ).all()
    assert len(companies) == 1
