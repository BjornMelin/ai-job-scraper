"""Configuration settings for the AI Job Scraper application."""

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseURLError(ValueError):
    """Custom exception for database URL configuration errors."""


class LogLevelError(ValueError):
    """Custom exception for invalid log level configuration."""


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file.

    Attributes:
        openai_api_key: OpenAI API key for LLM operations (cloud fallback).
        ai_token_threshold: Token threshold for local vs cloud model routing.
        proxy_pool: List of proxy URLs for scraping.
        use_proxies: Flag to enable proxy usage.
        use_checkpointing: Flag to enable checkpointing in workflows.
        db_url: Database connection URL.
        sqlite_pragmas: List of SQLite PRAGMA statements for optimization.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    openai_api_key: str = ""
    ai_token_threshold: int = Field(
        default=8000,
        description="Token threshold for routing between local and cloud models",
        ge=1000,
        le=32000,
    )
    proxy_pool: list[str] = []
    use_proxies: bool = False
    use_checkpointing: bool = False
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
        """Validate database URL format.

        Args:
            v (str): Database URL to validate.

        Returns:
            str: Validated database URL.

        Raises:
            DatabaseURLError: If the database URL is invalid.
        """
        if not v:
            raise DatabaseURLError(
                "Database URL configuration is missing or invalid. "
                "Please provide a valid database connection URL.",
            )

        supported_schemes = ("sqlite://", "postgresql://", "mysql://")
        if not v.startswith(supported_schemes) and not v.startswith("sqlite:"):
            # For relative paths, assume SQLite
            return f"sqlite:///{v}"
        return v

    @field_validator("proxy_pool")
    @classmethod
    def validate_proxy_urls(cls, v: list[str]) -> list[str]:
        """Validate proxy URLs format."""
        validated_proxies = []
        for original_proxy in v:
            if original_proxy and not original_proxy.startswith(
                ("http://", "https://", "socks5://"),
            ):
                # Assume HTTP proxy if no scheme specified
                formatted_proxy = f"http://{original_proxy}"
            else:
                formatted_proxy = original_proxy
            if formatted_proxy:  # Only add non-empty proxies
                validated_proxies.append(formatted_proxy)
        return validated_proxies

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
