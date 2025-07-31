"""Configuration settings for the AI Job Scraper application."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with .env file priority over environment variables.

    Settings are loaded with .env file taking precedence over environment variables.
    This ensures consistent behavior regardless of shell environment.
    """

    # OpenAI API key for enhanced job content extraction
    openai_api_key: str

    # OpenAI base URL for enhanced job content extraction
    openai_base_url: str = "https://api.openai.com/v1"

    # Groq API key for fast LLM inference (alternative to OpenAI)
    groq_api_key: str = ""

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

    def __init__(self, **kwargs):
        """Initialize settings with .env file priority over environment variables."""
        # Load .env file values first
        env_file_path = Path(".env")
        env_file_values = {}

        if env_file_path.exists():
            with open(env_file_path, encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        # Convert to lowercase w/ underscores for field matching
                        pydantic_key = key.lower()
                        env_file_values[pydantic_key] = value

        # Merge with any provided kwargs, with kwargs taking highest priority
        final_values = {**env_file_values, **kwargs}

        # Call parent init with merged values
        super().__init__(**final_values)
