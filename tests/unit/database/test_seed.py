"""Tests for database seeding functionality.

This module contains comprehensive tests for the database seeding system including:
- Initial seeding with predefined companies
- Idempotent seeding (no duplicates on re-run)
- Partial seeding with existing data preservation
- Data integrity validation
- CLI interface testing
"""

from typing import Any
from unittest.mock import patch

import pytest
import sqlmodel

from sqlmodel import Session, select
from typer.testing import CliRunner

from src.models import CompanySQL
from src.seed import app, seed


@pytest.fixture
def expected_companies() -> list[dict[str, "Any"]]:
    """Fixture providing expected seeded companies.

    Returns:
        List of company dictionaries with name, url, and active status
        for the default companies that should be seeded into the database.
    """
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


def test_seed_success(
    session: Session,
    expected_companies: list[dict[str, "Any"]],
) -> None:
    """Test successful database seeding with all expected companies.

    Validates that all predefined companies are properly inserted
    into the database with correct URLs and active status.
    """
    with patch("src.seed.engine", session.bind):
        seed()

        companies = (session.exec(select(CompanySQL))).all()
        assert len(companies) == len(expected_companies)
        for comp in companies:
            expected = next(e for e in expected_companies if e["name"] == comp.name)
            assert comp.url == expected["url"]
            assert comp.active == expected["active"]


def test_seed_idempotent(
    session: Session,
    expected_companies: list[dict[str, "Any"]],
) -> None:
    """Test that seeding is idempotent (can be run multiple times safely).

    Validates that running seed() multiple times doesn't create
    duplicate entries due to unique constraints.
    """
    with patch("src.seed.engine", session.bind):
        seed()
        seed()

        companies = (session.exec(select(CompanySQL))).all()
        assert len(companies) == len(expected_companies)


def test_seed_partial_existing(
    session: Session,
    expected_companies: list[dict[str, "Any"]],
) -> None:
    """Test seeding behavior when some companies already exist.

    Validates that existing company data is preserved (not overwritten)
    when seeding runs with partial existing data.
    """
    # Clear any existing companies to ensure clean test state
    session.exec(sqlmodel.text("DELETE FROM companysql"))
    session.commit()

    existing = CompanySQL(name="Anthropic", url="custom-url", active=False)
    session.add(existing)
    session.commit()

    with patch("src.seed.engine", session.bind):
        seed()

        companies = (session.exec(select(CompanySQL))).all()
        assert len(companies) == len(expected_companies)

        anthropic = next(c for c in companies if c.name == "Anthropic")
        assert anthropic.url == "custom-url"  # Preserved
        assert anthropic.active is False


def test_seed_data_integrity(expected_companies: list[dict[str, "Any"]]) -> None:
    """Test integrity and validity of seeded company data.

    Validates that all expected companies have:
    - Non-empty string names
    - HTTPS URLs
    - Active status set to True
    """
    assert len(expected_companies) > 0
    for comp in expected_companies:
        assert isinstance(comp["name"], str), "Company name should be a string"
        assert len(comp["name"]) > 0, "Company name should not be empty"
        assert comp["url"].startswith("https://"), (
            "Company URL should start with https://"
        )
        assert comp["active"] is True, "Company should be active"


def test_seed_cli_execution(session: Session) -> None:
    """Test CLI interface for seed command execution.

    Validates that the Typer CLI application properly executes
    the seed command and returns successful exit code.
    """
    runner = CliRunner()
    with patch("src.seed.engine", session.bind):
        result = runner.invoke(
            app,
            [],
        )  # No arguments needed, seed() is the default command
        assert result.exit_code == 0
        assert "Seeded" in result.output
