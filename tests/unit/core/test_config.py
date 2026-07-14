"""Tests for the maintained runtime settings."""

import pytest
from pydantic import ValidationError
from src.config import Settings


def test_settings_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("SCRAPER_LOG_LEVEL", raising=False)

    settings = Settings()

    assert settings.db_url == "sqlite:///jobs.db"
    assert settings.log_level == "INFO"
    assert "PRAGMA journal_mode = WAL" in settings.sqlite_pragmas
    assert "PRAGMA foreign_keys = ON" in settings.sqlite_pragmas


def test_dotenv_and_environment_precedence(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "DB_URL=sqlite:///dotenv.db\nSCRAPER_LOG_LEVEL=warning\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DB_URL", raising=False)
    monkeypatch.delenv("SCRAPER_LOG_LEVEL", raising=False)

    dotenv_settings = Settings()
    assert dotenv_settings.db_url == "sqlite:///dotenv.db"
    assert dotenv_settings.log_level == "WARNING"

    monkeypatch.setenv("DB_URL", "sqlite:///environment.db")
    assert Settings().db_url == "sqlite:///environment.db"


def test_settings_validate_database_and_log_level(monkeypatch):
    assert Settings(db_url="relative.db").db_url == "sqlite:///relative.db"

    with pytest.raises(ValidationError, match="Database URL configuration"):
        Settings(db_url="")
    with pytest.raises(ValidationError, match="Only SQLite"):
        Settings(db_url="postgresql://localhost/jobs")
    with pytest.raises(ValidationError, match="Invalid SQLite"):
        Settings(db_url="sqlite:garbage")
    with pytest.raises(ValidationError, match="Invalid SQLite"):
        Settings(db_url="sqlite://host/jobs.db")
    monkeypatch.setenv("SCRAPER_LOG_LEVEL", "verbose")
    with pytest.raises(ValidationError, match="Invalid logging configuration"):
        Settings()
