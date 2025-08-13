"""Pytest configuration and fixtures for AI Job Scraper tests.

This module provides test fixtures and configuration for the AI Job Scraper
test suite, including database session management, sample data creation,
and test settings configuration.
"""

from datetime import UTC, datetime

import pytest

from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from src.config import Settings
from src.models import CompanySQL, JobSQL


@pytest.fixture(scope="session")
def engine():
    """Create a temporary in-memory SQLite engine for the test session.

    Uses StaticPool to ensure schema and data persist across session connections.
    Optimized for session-level reuse to reduce setup overhead.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={
            "check_same_thread": False,
            # Performance optimizations for testing
            "isolation_level": None,  # Autocommit mode for faster tests
        },
        # Reduce connection overhead
        pool_pre_ping=True,
        pool_recycle=3600,
    )
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a new database session for each test.

    Uses transaction rollback for isolation without recreation overhead.
    """
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


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


@pytest.fixture
def sample_company(session: Session):
    """Create and insert a sample company for testing."""
    company = CompanySQL(
        name="Test Company",
        url="https://test.com/careers",
        active=True,
    )
    session.add(company)
    session.commit()
    session.refresh(company)
    return company


@pytest.fixture
def sample_job(session: Session):
    """Create and insert a sample job for testing."""
    job = JobSQL(
        company="Test Company",
        title="Senior AI Engineer",
        description="We are looking for an experienced AI engineer to join our team.",
        link="https://test.com/careers/ai-engineer-123",
        location="San Francisco, CA",
        posted_date=datetime.now(UTC),
        salary=(100000, 150000),
        favorite=False,
        notes="",
    )
    session.add(job)
    session.commit()
    session.refresh(job)
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
        "posted_date": datetime.now(UTC),
        "salary": "$100k-150k",
    }
