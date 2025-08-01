"""Configuration settings for the AI Job Scraper application."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file.

    Attributes:
        openai_api_key: OpenAI API key for LLM operations.
        groq_api_key: Groq API key for alternative LLM provider.
        use_groq: Flag to prefer Groq over OpenAI.
        proxy_pool: List of proxy URLs for scraping.
        use_proxies: Flag to enable proxy usage.
        use_checkpointing: Flag to enable checkpointing in workflows.
        db_url: Database connection URL.
        extraction_model: LLM model name for extraction tasks.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_ignore_empty=True, extra="ignore"
    )

    openai_api_key: str
    groq_api_key: str
    use_groq: bool = False
    proxy_pool: list[str] = []
    use_proxies: bool = False
    use_checkpointing: bool = False
    db_url: str = "sqlite:///jobs.db"
    extraction_model: str = "gpt-4o-mini"
