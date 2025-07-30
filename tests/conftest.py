"""Pytest configuration and fixtures for AI Job Scraper tests."""

import asyncio
import os
import tempfile

from datetime import datetime

import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Settings
from models import Base, CompanySQL, JobSQL


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    # Create test engine
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    # Create test session factory
    test_session = sessionmaker(bind=engine)

    yield test_session

    # Cleanup
    os.unlink(db_path)


@pytest.fixture
def test_settings():
    """Create test settings with temporary values."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Settings(
            openai_api_key="test-key-123",
            db_url="sqlite:///:memory:",
            cache_dir=temp_dir,
            min_jobs_for_cache=1,
        )


@pytest.fixture
def sample_company():
    """Create a sample company for testing."""
    return CompanySQL(name="Test Company", url="https://test.com/careers", active=True)


@pytest.fixture
def sample_job():
    """Create a sample job for testing."""
    return JobSQL(
        company="Test Company",
        title="Senior AI Engineer",
        description="We are looking for an experienced AI engineer to join our team.",
        link="https://test.com/careers/ai-engineer-123",
        location="San Francisco, CA",
        posted_date=datetime.now(),
        hash="test_hash_123",
        last_seen=datetime.now(),
        favorite=False,
        status="New",
        notes="",
    )


@pytest.fixture
def sample_job_dict():
    """Create a sample job dictionary for testing."""
    return {
        "company": "Test Company",
        "title": "Senior AI Engineer",
        "description": "We are looking for an experienced AI engineer.",
        "link": "https://test.com/careers/ai-engineer-123",
        "location": "San Francisco, CA",
        "posted_date": datetime.now(),
    }
