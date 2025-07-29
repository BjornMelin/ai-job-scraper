"""Configuration settings for the AI Job Scraper application."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support.

    This class defines the configuration settings for the AI Job Scraper,
    including database connection and API keys. Settings can be loaded
    from environment variables or a .env file.

    Attributes:
        openai_api_key (str): OpenAI API key for enhanced job extraction.
        db_url (str): Database connection URL, defaults to SQLite file.

    """

    openai_api_key: str
    db_url: str = "sqlite:///jobs.db"

    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)
