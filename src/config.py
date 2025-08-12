"""Configuration settings for the AI Job Scraper application."""

from pydantic import Field, field_validator
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

    openai_api_key: str = ""
    groq_api_key: str = ""
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

    # Enhanced configuration with validation and aliases
    log_level: str = Field(
        default="INFO",
        description="Logging level for the application",
        validation_alias="SCRAPER_LOG_LEVEL",
    )

    @field_validator("db_url")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        """Validate database URL format."""
        if not v:
            raise ValueError("Database URL cannot be empty")

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
                ("http://", "https://", "socks5://")
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
        """Validate logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            error_msg = f"Invalid log level: {v}. Must be one of {valid_levels}"
            raise ValueError(error_msg)
        return v.upper()
