"""AI Job Scraper Core Modules,

This package contains the core modules for the AI Job Scraper.
"""

from .config import Settings
from .database import get_session
from .models import CompanySQL, JobSQL
from .scraper_company_pages import (
    extract_details,
    extract_job_lists,
    load_active_companies,
    normalize_jobs,
    save_jobs,
    scrape_company_pages,
)
from .scraper_job_boards import scrape_job_boards
from .utils import (
    get_extraction_model,
    get_llm_client,
    get_proxy,
    random_delay,
    random_user_agent,
)

__all__ = [
    "Settings",
    "get_session",
    "CompanySQL",
    "JobSQL",
    "get_extraction_model",
    "get_llm_client",
    "get_proxy",
    "random_delay",
    "random_user_agent",
    "scrape_job_boards",
    "load_active_companies",
    "extract_job_lists",
    "extract_details",
    "normalize_jobs",
    "save_jobs",
    "scrape_company_pages",
]
