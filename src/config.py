"""Configuration settings for Job Tracker."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class DatabaseURLError(ValueError):
    """Custom exception for database URL configuration errors."""


class LogLevelError(ValueError):
    """Custom exception for invalid log level configuration."""


def normalize_sqlite_url(value: str) -> str:
    """Return one canonical SQLAlchemy SQLite URL or reject it."""
    value = value.strip()
    if not value:
        raise DatabaseURLError(
            "Database URL configuration is missing or invalid. "
            "Please provide a valid SQLite database URL.",
        )

    candidate = value if ":" in value else f"sqlite:///{value}"
    try:
        url = make_url(candidate)
        has_authority = any((url.username, url.password, url.host, url.port))
    except (ArgumentError, ValueError) as error:
        raise DatabaseURLError("Invalid SQLite database URL.") from error

    if url.drivername != "sqlite":
        raise DatabaseURLError("Only SQLite database URLs are supported.")
    if has_authority or url.database == "":
        raise DatabaseURLError("Invalid SQLite database URL.")
    return url.render_as_string(hide_password=False)


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file.

    Attributes:
        db_url: Database connection URL.
        sqlite_pragmas: SQLite PRAGMA statements applied to each connection.
        log_level: Application logging level.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    db_url: str = "sqlite:///jobs.db"

    # Database optimization settings
    sqlite_pragmas: list[str] = [
        "PRAGMA journal_mode = WAL",  # Write-Ahead Logging for better concurrency
        "PRAGMA synchronous = NORMAL",  # Balanced safety/performance
        "PRAGMA cache_size = 64000",  # 64MB cache (default is 2MB)
        "PRAGMA temp_store = MEMORY",  # Store temp tables in memory
        "PRAGMA mmap_size = 134217728",  # 128MB memory-mapped I/O
        "PRAGMA foreign_keys = ON",  # Enable foreign key constraints
        "PRAGMA optimize",  # Auto-optimize indexes
    ]

    # Configuration fields with validation and aliases
    log_level: str = Field(
        default="INFO",
        description="Logging level for the application",
        validation_alias="SCRAPER_LOG_LEVEL",
    )

    @field_validator("db_url")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        """Normalize the app's only supported database URL."""
        return normalize_sqlite_url(v)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level.

        Args:
            v (str): Log level to validate.

        Returns:
            str: Validated log level in uppercase.

        Raises:
            LogLevelError: If the log level is invalid.
        """
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise LogLevelError(
                f"Invalid logging configuration: '{v}' is not a valid log level. "
                f"Supported levels are: {', '.join(valid_levels)}",
            )
        return v.upper()
