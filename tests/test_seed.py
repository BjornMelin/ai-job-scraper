"""Tests for database seeding functionality."""

from unittest.mock import patch

import pytest

from sqlmodel import select
from src.models import CompanySQL
from src.seed import app, seed
from typer.testing import CliRunner


@pytest.fixture
def expected_companies():
    """Fixture providing expected seeded companies."""
    return [
        {
            "name": "Anthropic",
            "url": "https://www.anthropic.com/careers",
            "active": True,
        },
        {"name": "OpenAI", "url": "https://openai.com/careers", "active": True},
        {
            "name": "Google DeepMind",
            "url": "https://deepmind.google/about/careers/",
            "active": True,
        },
        {"name": "xAI", "url": "https://x.ai/careers/", "active": True},
        {"name": "Meta", "url": "https://www.metacareers.com/jobs", "active": True},
        {
            "name": "Microsoft",
            "url": "https://jobs.careers.microsoft.com/global/en/search",
            "active": True,
        },
        {
            "name": "NVIDIA",
            "url": "https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite",
            "active": True,
        },
    ]


@pytest.mark.asyncio
async def test_seed_success(temp_db, expected_companies):
    """Test successful seeding."""
    with patch("src.seed.engine", temp_db.bind):
        seed()

        companies = (await temp_db.exec(select(CompanySQL))).all()
        assert len(companies) == len(expected_companies)
        for comp in companies:
            expected = next(e for e in expected_companies if e["name"] == comp.name)
            assert comp.url == expected["url"]
            assert comp.active == expected["active"]


@pytest.mark.asyncio
async def test_seed_idempotent(temp_db, expected_companies):
    """Test seeding idempotency."""
    with patch("src.seed.engine", temp_db.bind):
        seed()
        seed()

        companies = (await temp_db.exec(select(CompanySQL))).all()
        assert len(companies) == len(expected_companies)


@pytest.mark.asyncio
async def test_seed_partial_existing(temp_db, expected_companies):
    """Test seeding with existing companies."""
    existing = CompanySQL(name="Anthropic", url="custom-url", active=False)
    temp_db.add(existing)
    await temp_db.commit()

    with patch("src.seed.engine", temp_db.bind):
        seed()

        companies = (await temp_db.exec(select(CompanySQL))).all()
        assert len(companies) == len(expected_companies)

        anthropic = next(c for c in companies if c.name == "Anthropic")
        assert anthropic.url == "custom-url"  # Preserved
        assert anthropic.active is False


@pytest.mark.asyncio
async def test_seed_data_integrity(expected_companies):
    """Test seeded data integrity."""
    assert len(expected_companies) > 0
    for comp in expected_companies:
        assert isinstance(comp["name"], str), "Company name should be a string"
        assert len(comp["name"]) > 0, "Company name should not be empty"
        assert comp["url"].startswith("https://"), (
            "Company URL should start with https://"
        )
        assert comp["active"] is True, "Company should be active"


def test_seed_cli_execution(temp_db):
    """Test CLI execution of seed."""
    runner = CliRunner()
    with patch("src.seed.engine", temp_db.bind):
        result = runner.invoke(app, ["seed"])
        assert result.exit_code == 0
        assert "Seeded" in result.output
