"""Configuration settings for the AI Job Scraper application."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.

    This class defines the configuration settings for the AI Job Scraper,
    including database connection, API keys, and file paths. Settings can be loaded
    from environment variables or a .env file.

    Attributes:
        openai_api_key (str): OpenAI API key for enhanced job extraction.
        db_url (str): Database connection URL, defaults to SQLite file.
        cache_dir (str): Directory path for schema cache files.
        min_jobs_for_cache (int): Minimum jobs required to save schema cache.

    """

    openai_api_key: str
    db_url: str = "sqlite:///jobs.db"
    cache_dir: str = "./cache"
    min_jobs_for_cache: int = 1

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)
