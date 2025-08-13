"""AI Job Scraper Core Modules.

This package contains the core modules for the AI Job Scraper.
"""

# Configuration and Settings
from src.config import Settings

# Constants
from src.constants import AI_REGEX, RELEVANT_PHRASES, SEARCH_KEYWORDS, SEARCH_LOCATIONS

# Seed module - import removed to prevent double model imports in Streamlit
# Import seed function directly when needed
# Utilities
# Import the missing function separately to avoid linter issues
from src.core_utils import (
    get_extraction_model,
    get_llm_client,
    get_proxy,
    random_delay,
    random_user_agent,
    resolve_jobspy_proxies,
)

# Database - explicit import from database.py module
from src.database import (
    SessionLocal,
    create_db_and_tables,
    db_session,
    engine,
    get_session,
)

# Models - removed from __init__.py to prevent double import conflicts with Alembic
# Import models directly in modules where they are needed instead
# Scraper modules - imports removed to prevent double model imports in Streamlit
# Import scraper functions directly in modules where needed instead
from src.scraper_job_boards import scrape_job_boards

__all__ = [
    # Constants
    "AI_REGEX",
    "RELEVANT_PHRASES",
    "SEARCH_KEYWORDS",
    "SEARCH_LOCATIONS",
    # Database
    "SessionLocal",
    # Configuration
    "Settings",
    # Database
    "create_db_and_tables",
    "db_session",
    "engine",
    # Utilities
    "get_extraction_model",
    "get_llm_client",
    "get_proxy",
    "get_session",
    "random_delay",
    "random_user_agent",
    "resolve_jobspy_proxies",
    # Main scraper functions - removed to prevent double model imports
    # Job board scraper
    "scrape_job_boards",
    # Seed module - removed to prevent double model imports
]
