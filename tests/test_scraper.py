"""Tests for the AI Job Scraper functionality.

This module contains unit tests for the core scraping functionality,
including job relevance filtering, link validation, and data validation.
"""

import pytest

from models import JobPydantic
from scraper import is_relevant, validate_link


def test_is_relevant() -> None:
    """Test job relevance filtering based on title keywords."""
    assert is_relevant({"title": "AI Engineer"})
    assert not is_relevant({"title": "Sales"})


@pytest.mark.asyncio
async def test_validate_link() -> None:
    """Test URL validation for job posting links."""
    valid = await validate_link("https://google.com")
    assert valid is not None
    invalid = await validate_link("https://invalid.url")
    assert invalid is None


def test_job_validation() -> None:
    """Test Pydantic job model validation."""
    job = JobPydantic(
        company="Test", title="AI Engineer", description="Desc", link="https://test.com"
    )
    assert job.title == "AI Engineer"
