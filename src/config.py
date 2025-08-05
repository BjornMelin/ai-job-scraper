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
        sqlite_pragmas: List of SQLite PRAGMA statements for optimization.
        db_monitoring: Flag to enable database performance monitoring.
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
    db_monitoring: bool = False  # Toggle slow-query logging on/off
