"""Configuration settings for the AI Job Scraper application."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.

    Settings are loaded from environment variables or .env file.
    Environment variables take precedence over .env file values.
    """

    # OpenAI API key for enhanced job content extraction
    openai_api_key: str

    # Database connection URL (SQLite, PostgreSQL, MySQL supported)
    db_url: str = "sqlite:///jobs.db"

    # Cache directory for job schema storage
    cache_dir: str = "./cache"

    # Minimum jobs required before saving schema cache
    min_jobs_for_cache: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",  # Ignore extra environment variables not defined in model
    )
