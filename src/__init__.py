"""AI Job Scraper Core Modules.

This package contains the core modules for the AI Job Scraper.
"""

# Configuration and Settings
from .config import Settings

# Constants
from .constants import AI_REGEX, RELEVANT_PHRASES, SEARCH_KEYWORDS, SEARCH_LOCATIONS

# Database
from .database import (
    SessionLocal,
    create_db_and_tables,
    engine,
    get_session,
)

# Models
from .models import CompanySQL, JobSQL

# Scraper modules
from .scraper import scrape, scrape_all, update_db
from .scraper_company_pages import (
    State,
    extract_details,
    extract_job_lists,
    load_active_companies,
    normalize_jobs,
    save_jobs,
    scrape_company_pages,
)
from .scraper_job_boards import scrape_job_boards

# Seed module
from .seed import seed

# Utilities
from .utils import (
    get_extraction_model,
    get_llm_client,
    get_proxy,
    random_delay,
    random_user_agent,
)

__all__ = [
    # Configuration
    "Settings",
    # Database
    "engine",
    "SessionLocal",
    "create_db_and_tables",
    "get_session",
    # Models
    "CompanySQL",
    "JobSQL",
    # Constants
    "AI_REGEX",
    "RELEVANT_PHRASES",
    "SEARCH_KEYWORDS",
    "SEARCH_LOCATIONS",
    # Utilities
    "get_extraction_model",
    "get_llm_client",
    "get_proxy",
    "random_delay",
    "random_user_agent",
    # Main scraper functions
    "scrape",
    "scrape_all",
    "update_db",
    # Job board scraper
    "scrape_job_boards",
    # Company pages scraper
    "State",
    "load_active_companies",
    "extract_job_lists",
    "extract_details",
    "normalize_jobs",
    "save_jobs",
    "scrape_company_pages",
    # Seed
    "seed",
]
