"""Module for scraping job listings from company career pages via agentic workflow.

This module uses ScrapeGraphAI for prompt-based extraction and LangGraph to
orchestrate multi-step scraping: first extracting job lists with URLs, then
details from individual job pages. It integrates proxies, user agents, and
delays for evasion, normalizes data to JobSQL models, and saves to the
database. Checkpointing is optional for resumability.
"""

import logging

from datetime import datetime, timezone
from typing import TypedDict
from urllib.parse import urljoin

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from scrapegraphai.graphs import SmartScraperMultiGraph
from sqlmodel import select

from .config import Settings
from .database import SessionLocal
from .models import CompanySQL, JobSQL
from .utils import (
    get_extraction_model,
    get_llm_client,
    get_proxy,
    random_delay,
    random_user_agent,
)

settings = Settings()
llm_client = get_llm_client()
extraction_model = get_extraction_model()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default maximum jobs per company
DEFAULT_MAX_JOBS_PER_COMPANY = 50


class State(TypedDict):
    """State for the scraping workflow."""

    companies: list[CompanySQL]
    partial_jobs: list[dict]
    raw_full_jobs: list[dict]
    normalized_jobs: list[JobSQL]
    max_jobs_per_company: int


def load_active_companies() -> list[CompanySQL]:
    """Load active companies from the database.

    Returns:
        list[CompanySQL]: List of active CompanySQL instances.
    """
    session = SessionLocal()
    try:
        return session.exec(select(CompanySQL).where(CompanySQL.active)).all()
    finally:
        session.close()


def _add_proxy_config(config: dict, extraction_type: str) -> dict:
    """Add proxy configuration to ScrapeGraphAI config if enabled.

    Args:
        config: Base configuration dictionary.
        extraction_type: Type of extraction for logging (e.g., "job list",
            "job details").

    Returns:
        dict: Updated configuration with proxy settings if enabled.
    """
    if settings.use_proxies and settings.proxy_pool:
        proxy_url = get_proxy()
        if proxy_url:
            config["loader_kwargs"] = {
                "proxy": {
                    "server": proxy_url,
                }
            }
            logger.info("Using proxy for %s extraction: %s", extraction_type, proxy_url)
    return config


def extract_job_lists(state: State) -> dict[str, list[dict]]:
    """Extract job listings with titles and URLs from company career pages.

    Uses SmartScraperMultiGraph for parallel extraction across companies.
    Respects the max_jobs_per_company limit from session state.

    Args:
        state: Current workflow state.

    Returns:
        dict[str, list[dict]]: Updated state with partial job info.
    """
    companies = state["companies"]
    sources = [c.url for c in companies]
    company_map = {c.url: c for c in companies}

    if not sources:
        return {"partial_jobs": []}

    logger.info("-" * 50)
    logger.info("📋 Starting job list extraction from %d companies", len(companies))
    for i, company in enumerate(companies, 1):
        logger.info("  %d. %s", i, company.name)

    # Get job limit from state
    max_jobs_per_company = state["max_jobs_per_company"]

    prompt = (
        f"Extract up to {max_jobs_per_company} job listings from the page. "
        "Return a JSON with a single key 'jobs' containing a list of objects, "
        "each with 'title' (string) and 'url' (string, the full URL to the "
        "job detail page). Limit results to the most recent or relevant jobs."
    )

    config = {
        "llm": {"client": llm_client, "model": extraction_model},
        "verbose": True,
        "headers": {"User-Agent": random_user_agent()},
    }

    config = _add_proxy_config(config, "job list")

    multi_graph = SmartScraperMultiGraph(prompt, sources, config)
    result = multi_graph.run()

    partial_jobs = []
    company_job_counts = {}

    for source_url, extracted in result.items():
        # Validate extracted data is in expected format
        if not isinstance(extracted, dict) or "jobs" not in extracted:
            logger.warning("❌ Failed to extract jobs from %s", source_url)
            continue

        company = company_map[source_url]

        # Use list comprehension for better performance
        limited_jobs = [
            {
                "company": company.name,
                "title": job["title"],
                "url": urljoin(source_url, job["url"]),
            }
            for job in extracted["jobs"]
            if "title" in job and "url" in job
        ]

        # Apply limit using slicing
        company_jobs = limited_jobs[:max_jobs_per_company]
        company_job_counts[company.name] = len(company_jobs)
        partial_jobs.extend(company_jobs)

        logger.info("  ✅ %s: Extracted %d jobs", company.name, len(company_jobs))

    # Log summary of extraction results
    logger.info("-" * 50)
    logger.info("📊 Job list extraction summary:")
    logger.info("  • Companies processed: %d", len(company_job_counts))
    logger.info("  • Total job listings found: %d", len(partial_jobs))
    if company_job_counts:
        avg_jobs = sum(company_job_counts.values()) / len(company_job_counts)
        logger.info("  • Average jobs per company: %.1f", avg_jobs)

    random_delay()
    return {
        "partial_jobs": partial_jobs,
        "max_jobs_per_company": state["max_jobs_per_company"],
    }


def extract_details(state: State) -> dict[str, list[dict]]:
    """Extract detailed job information from individual job pages.

    Uses SmartScraperMultiGraph for parallel extraction across job URLs.

    Args:
        state: Current workflow state.

    Returns:
        dict[str, list[dict]]: Updated state with full raw job data.
    """
    partial_jobs = state.get("partial_jobs", [])
    urls = [j["url"] for j in partial_jobs]
    if not urls:
        return {"raw_full_jobs": []}

    logger.info("-" * 50)
    logger.info("🔍 Starting detailed extraction for %d job pages", len(urls))

    prompt = (
        "Extract the following information from the job detail page. "
        "Return a JSON with keys: "
        "'description' (full job description text as string), "
        "'location' (job location as string), "
        "'posted_date' (date posted in YYYY-MM-DD format if possible, else "
        "original string), "
        "'salary' (salary range as string, like '$100k-150k'), "
        "'link' (application link if available, else the page URL)."
    )

    config = {
        "llm": {"client": llm_client, "model": extraction_model},
        "verbose": True,
        "headers": {"User-Agent": random_user_agent()},
    }

    config = _add_proxy_config(config, "job details")

    multi_graph = SmartScraperMultiGraph(prompt, urls, config)
    result = multi_graph.run()

    # Map partial jobs by URL for merging
    partial_map = {j["url"]: j for j in partial_jobs}
    raw_full_jobs = []
    success_count = 0

    for url, details in result.items():
        # Skip if details not in expected dict format
        if not isinstance(details, dict):
            logger.warning("❌ Failed to extract details from %s", url)
            continue

        partial = partial_map.get(url)
        if partial:
            full_job = {**partial, **details}
            raw_full_jobs.append(full_job)
            success_count += 1

    logger.info("📊 Detail extraction summary:")
    logger.info("  • Job pages processed: %d", len(result))
    logger.info("  • Successfully extracted: %d", success_count)
    logger.info("  • Failed extractions: %d", len(result) - success_count)

    random_delay()
    return {"raw_full_jobs": raw_full_jobs}


def normalize_jobs(state: State) -> dict[str, list[JobSQL]]:
    """Normalize raw job data into JobSQL models.

    Parses dates and relies on model validators for salary.

    Args:
        state: Current workflow state.

    Returns:
        dict[str, list[JobSQL]]: Updated state with normalized jobs.
    """
    raw_jobs = state.get("raw_full_jobs", [])
    normalized = []
    # Common date formats to attempt parsing
    date_formats = ["%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%d %B %Y"]

    if raw_jobs:
        logger.info("-" * 50)
        logger.info("🔄 Normalizing %d raw job records", len(raw_jobs))

    for raw in raw_jobs:
        posted_str = raw.get("posted_date")
        posted = None
        if posted_str:
            # Try parsing with each format until success
            for fmt in date_formats:
                try:
                    posted = datetime.strptime(posted_str, fmt).replace(
                        tzinfo=timezone.utc
                    )
                    break
                except ValueError:
                    pass
            if not posted:
                logger.warning("Could not parse date: %s", posted_str)

        # Use JobSQL validator for salary parsing and new schema
        try:
            # Get or create company ID using a session
            session = SessionLocal()
            try:
                company_name = raw["company"]
                company = session.exec(
                    select(CompanySQL).where(CompanySQL.name == company_name)
                ).first()

                if not company:
                    # Create new company
                    company = CompanySQL(
                        name=company_name,
                        url="",  # Will be updated later if available
                        active=True,
                    )
                    session.add(company)
                    session.commit()
                    session.refresh(company)

                company_id = company.id
            finally:
                session.close()

            # Use factory method to ensure proper validation and hash generation
            job = JobSQL.create_validated(
                company_id=company_id,
                title=raw["title"],
                description=raw.get("description", ""),
                link=raw.get("link", raw["url"]),
                location=raw.get("location", ""),
                posted_date=posted,
                salary=raw.get("salary", ""),
                application_status="New",
                last_seen=datetime.now(timezone.utc),
            )
            normalized.append(job)
        except Exception:
            logger.exception("Failed to normalize job %s", raw.get("url"))

    if raw_jobs:
        success_rate = (len(normalized) / len(raw_jobs)) * 100
        logger.info("📊 Normalization summary:")
        logger.info("  • Raw jobs processed: %d", len(raw_jobs))
        logger.info(
            "  • Successfully normalized: %d (%.1f%%)", len(normalized), success_rate
        )
        logger.info("  • Failed normalizations: %d", len(raw_jobs) - len(normalized))

    return {"normalized_jobs": normalized}


def get_normalized_jobs(state: State) -> dict:
    """Get normalized jobs from state for external processing.

    This node no longer writes to the database directly. Instead, it retrieves
    the normalized jobs from the workflow state and returns them to the caller
    for processing by the SmartSyncEngine.

    Args:
        state: Current workflow state.

    Returns:
        dict: Contains normalized_jobs for external processing.
    """
    normalized_jobs = state.get("normalized_jobs", [])
    return {"normalized_jobs": normalized_jobs}


def scrape_company_pages(
    max_jobs_per_company: int = DEFAULT_MAX_JOBS_PER_COMPANY,
) -> list[JobSQL]:
    """Run the agentic scraping workflow for active companies.

    Args:
        max_jobs_per_company: Limit for jobs per company (default: 50).

    Returns:
        list[JobSQL]: List of normalized job objects scraped from company pages.
    """
    companies = load_active_companies()
    if not companies:
        logger.info("No active companies to scrape.")
        return []

    logger.info("=" * 50)
    logger.info("🏢 STARTING COMPANY PAGES SCRAPING")
    logger.info("=" * 50)
    logger.info("Active companies found: %d", len(companies))
    logger.info("Max jobs per company: %d", max_jobs_per_company)

    workflow = StateGraph(State)
    workflow.add_node("extract_lists", extract_job_lists)
    workflow.add_node("extract_details", extract_details)
    workflow.add_node("normalize", normalize_jobs)
    workflow.add_node("save", get_normalized_jobs)

    workflow.add_edge("extract_lists", "extract_details")
    workflow.add_edge("extract_details", "normalize")
    workflow.add_edge("normalize", "save")
    workflow.add_edge("save", END)

    workflow.set_entry_point("extract_lists")

    checkpointer = None
    if settings.use_checkpointing:
        checkpointer = SqliteSaver.from_conn_string("checkpoints.sqlite")

    graph = workflow.compile(checkpointer=checkpointer)

    # Include max_jobs_per_company in initial state
    initial_state = {
        "companies": companies,
        "max_jobs_per_company": max_jobs_per_company,
    }

    try:
        final_state = graph.invoke(initial_state)
        normalized_jobs = final_state.get("normalized_jobs", [])
    except Exception:
        logger.exception("❌ Company pages scraping workflow failed")
        return []
    else:
        logger.info("=" * 50)
        logger.info("✅ COMPANY PAGES SCRAPING COMPLETED")
        logger.info("Total jobs scraped from company pages: %d", len(normalized_jobs))
        logger.info("=" * 50)

        return normalized_jobs
