"""AI Job Scraper Core Modules.

This package contains the core modules for the AI Job Scraper.
"""

# Configuration and Settings
from src.config import Settings

# Constants
from src.constants import AI_REGEX, RELEVANT_PHRASES, SEARCH_KEYWORDS, SEARCH_LOCATIONS

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
# Scraper modules
from src.scraper import scrape, scrape_all
from src.scraper_company_pages import (
    State,
    extract_details,
    extract_job_lists,
    load_active_companies,
    normalize_jobs,
    scrape_company_pages,
)
from src.scraper_job_boards import scrape_job_boards

# Seed module
from src.seed import seed

# Utilities
from src.utils import (
    get_extraction_model,
    get_llm_client,
    get_proxy,
    random_delay,
    random_user_agent,
)

__all__ = [
    # Constants
    "AI_REGEX",
    "RELEVANT_PHRASES",
    "SEARCH_KEYWORDS",
    "SEARCH_LOCATIONS",
    # Database
    "SessionLocal",
    "create_db_and_tables",
    "db_session",
    "engine",
    "get_session",
    # Configuration
    "Settings",
    # Company pages scraper
    "State",
    "extract_details",
    "extract_job_lists",
    "load_active_companies",
    "normalize_jobs",
    "scrape_company_pages",
    # Job board scraper
    "scrape_job_boards",
    # Main scraper functions
    "scrape",
    "scrape_all",
    # Utilities
    "get_extraction_model",
    "get_llm_client",
    "get_proxy",
    "random_delay",
    "random_user_agent",
    # Seed
    "seed",
]
