"""Pytest configuration and fixtures for AI Job Scraper tests."""

import asyncio

from datetime import datetime

import pytest
import pytest_asyncio

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.sql.expression import Select, SelectOfScalar
from src.config import Settings
from src.models import CompanySQL, JobSQL

# Patch SQLModel for async query caching
Select.inherit_cache = True  # type: ignore[attr-defined]
SelectOfScalar.inherit_cache = True  # type: ignore[attr-defined]


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def temp_db():
    """Create a temporary in-memory database for async testing."""
    # Create async engine for testing with in-memory SQLite
    test_engine: AsyncEngine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", echo=False, future=True
    )

    # Create tables
    async with test_engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)

    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session


@pytest.fixture
def test_settings():
    """Create test settings with temporary values."""
    return Settings(
        openai_api_key="test-key-123",
        groq_api_key="test-groq-key",
        use_groq=False,
        proxy_pool=[],
        use_proxies=False,
        use_checkpointing=False,
        db_url="sqlite:///:memory:",
        extraction_model="gpt-4o-mini",
    )


@pytest_asyncio.fixture
async def sample_company(temp_db: AsyncSession):
    """Create and insert a sample company for testing."""
    company = CompanySQL(
        name="Test Company", url="https://test.com/careers", active=True
    )
    temp_db.add(company)
    await temp_db.commit()
    await temp_db.refresh(company)
    return company


@pytest_asyncio.fixture
async def sample_job(temp_db: AsyncSession):
    """Create and insert a sample job for testing."""
    job = JobSQL(
        company="Test Company",
        title="Senior AI Engineer",
        description="We are looking for an experienced AI engineer to join our team.",
        link="https://test.com/careers/ai-engineer-123",
        location="San Francisco, CA",
        posted_date=datetime.now(datetime.UTC),
        salary=(100000, 150000),
        favorite=False,
        notes="",
    )
    temp_db.add(job)
    await temp_db.commit()
    await temp_db.refresh(job)
    return job


@pytest.fixture
def sample_job_dict():
    """Create a sample job dictionary for testing."""
    return {
        "company": "Test Company",
        "title": "Senior AI Engineer",
        "description": "We are looking for an experienced AI engineer.",
        "link": "https://test.com/careers/ai-engineer-123",
        "location": "San Francisco, CA",
        "posted_date": datetime.now(datetime.UTC),
        "salary": "$100k-150k",
    }
