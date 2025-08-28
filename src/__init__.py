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
from src.core_utils import (
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
# Scraper modules - removed during SPEC-001 Phase 4 scraping service deletion
# Custom scraping logic replaced with direct JobSpy library usage

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
    "get_proxy",
    "get_session",
    "random_delay",
    "random_user_agent",
    "resolve_jobspy_proxies",
    # Main scraper functions - removed during SPEC-001 Phase 4 deletion
    # Custom scraping replaced with direct JobSpy library usage
    # Seed module - removed to prevent double model imports
]
