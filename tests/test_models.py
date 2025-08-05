"""Tests for database models and Pydantic validation."""

import datetime

import pytest

from sqlalchemy.exc import IntegrityError
from sqlmodel import select
from src.models import CompanySQL, JobSQL


@pytest.mark.asyncio
async def test_company_sql_creation(temp_db):
    """Test creating and querying CompanySQL."""
    company = CompanySQL(name="Test Co", url="https://test.co/careers", active=True)
    temp_db.add(company)
    await temp_db.commit()
    await temp_db.refresh(company)

    result = await temp_db.exec(select(CompanySQL).where(CompanySQL.name == "Test Co"))
    retrieved = result.first()
    assert retrieved.name == "Test Co"
    assert retrieved.active is True


@pytest.mark.asyncio
async def test_company_unique_name(temp_db):
    """Test company name uniqueness."""
    company1 = CompanySQL(name="Unique Co", url="https://unique1.co", active=True)
    temp_db.add(company1)
    await temp_db.commit()

    company2 = CompanySQL(name="Unique Co", url="https://unique2.co", active=False)
    temp_db.add(company2)
    with pytest.raises(IntegrityError):
        await temp_db.commit()


@pytest.mark.asyncio
async def test_job_sql_creation(temp_db):
    """Test creating and querying JobSQL."""
    job_data = {
        "company": "Test Co",
        "title": "AI Engineer",
        "description": "AI role",
        "link": "https://test.co/job",
        "location": "Remote",
        "posted_date": datetime.datetime.now(datetime.UTC),
        "salary": "$100k-150k",
    }
    job = JobSQL.model_validate(job_data)
    temp_db.add(job)
    await temp_db.commit()
    await temp_db.refresh(job)

    result = await temp_db.exec(select(JobSQL).where(JobSQL.title == "AI Engineer"))
    retrieved = result.first()
    assert retrieved.company == "Test Co"
    assert list(retrieved.salary) == [
        100000,
        150000,
    ]  # JSON column converts tuple to list


@pytest.mark.asyncio
async def test_job_unique_link(temp_db):
    """Test job link uniqueness."""
    job1_data = {
        "company": "Test Co",
        "title": "Job1",
        "description": "Desc1",
        "link": "https://test.co/job",
        "location": "Remote",
        "salary": (None, None),
    }
    job1 = JobSQL.model_validate(job1_data)
    temp_db.add(job1)
    await temp_db.commit()

    job2_data = {
        "company": "Test Co",
        "title": "Job2",
        "description": "Desc2",
        "link": "https://test.co/job",
        "location": "Office",
        "salary": (None, None),
    }
    job2 = JobSQL.model_validate(job2_data)
    temp_db.add(job2)
    with pytest.raises(IntegrityError):
        await temp_db.commit()


@pytest.mark.parametrize(
    ("salary_input", "expected"),
    [
        ("$100k-150k", (100000, 150000)),
        ("£80,000 - £120,000", (80000, 120000)),
        ("$100k", (100000, None)),
        ("invalid", (None, None)),
        ("$100k-150k-200k", (100000, 200000)),
        ("80k +", (80000, None)),
        ("", (None, None)),
        (None, (None, None)),
        ((50000, 100000), (50000, 100000)),
        ("100-120k USD", (100, 120000)),  # Reveals issue: inconsistent scale
        ("From $90,000 a year", (90000, None)),
    ],
)
def test_salary_parsing(salary_input, expected):
    """Test salary parsing validator, revealing potential issue with mixed scales."""
    job_data = {
        "company": "Test Co",
        "title": "Test Job",
        "description": "Test description",
        "link": "https://test.com/job",
        "location": "Test Location",
        "salary": salary_input,
    }
    job = JobSQL.model_validate(job_data)
    assert job.salary == expected
