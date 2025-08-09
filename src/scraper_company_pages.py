"""Module for scraping job listings from company career pages via agentic workflow.

This module uses ScrapeGraphAI for prompt-based extraction and LangGraph to
orchestrate multi-step scraping: first extracting job lists with URLs, then
details from individual job pages. It integrates proxies, user agents, and
delays for evasion, normalizes data to JobSQL models, and saves to the
database. Checkpointing is optional for resumability.
"""

import hashlib
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


class State(TypedDict):
    """State for the scraping workflow."""

    companies: list[CompanySQL]
    partial_jobs: list[dict]
    raw_full_jobs: list[dict]
    normalized_jobs: list[JobSQL]


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


def extract_job_lists(state: State) -> dict[str, list[dict]]:
    """Extract job listings with titles and URLs from company career pages.

    Uses SmartScraperMultiGraph for parallel extraction across companies.

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

    prompt = (
        "Extract all job listings from the page. "
        "Return a JSON with a single key 'jobs' containing a list of objects, "
        "each with 'title' (string) and 'url' (string, the full URL to the "
        "job detail page)."
    )

    config = {
        "llm": {"client": llm_client, "model": extraction_model},
        "verbose": True,
        "headers": {"User-Agent": random_user_agent()},
    }

    # Add proxy configuration if enabled
    if settings.use_proxies and settings.proxy_pool:
        proxy_url = get_proxy()
        if proxy_url:
            config["loader_kwargs"] = {
                "proxy": {
                    "server": proxy_url,
                }
            }
            logger.info("Using proxy for job list extraction: %s", proxy_url)

    multi_graph = SmartScraperMultiGraph(prompt, sources, config)
    result = multi_graph.run()

    partial_jobs = []
    for source_url, extracted in result.items():
        # Validate extracted data is in expected format
        if not isinstance(extracted, dict) or "jobs" not in extracted:
            logger.warning("Failed to extract jobs from %s", source_url)
            continue

        company = company_map[source_url]
        for job in extracted["jobs"]:
            if "title" in job and "url" in job:
                # Ensure full URL by joining with base if relative
                full_url = urljoin(source_url, job["url"])
                partial_jobs.append(
                    {
                        "company": company.name,
                        "title": job["title"],
                        "url": full_url,
                    }
                )

    random_delay()
    return {"partial_jobs": partial_jobs}


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

    # Add proxy configuration if enabled
    if settings.use_proxies and settings.proxy_pool:
        proxy_url = get_proxy()
        if proxy_url:
            config["loader_kwargs"] = {
                "proxy": {
                    "server": proxy_url,
                }
            }
            logger.info("Using proxy for job details extraction: %s", proxy_url)

    multi_graph = SmartScraperMultiGraph(prompt, urls, config)
    result = multi_graph.run()

    # Map partial jobs by URL for merging
    partial_map = {j["url"]: j for j in partial_jobs}
    raw_full_jobs = []
    for url, details in result.items():
        # Skip if details not in expected dict format
        if not isinstance(details, dict):
            logger.warning("Failed to extract details from %s", url)
            continue

        partial = partial_map.get(url)
        if partial:
            full_job = {**partial, **details}
            raw_full_jobs.append(full_job)

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

            # Create content hash
            content = f"{raw['title']}{raw.get('description', '')}{raw['company']}"
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            job = JobSQL(
                company_id=company_id,
                title=raw["title"],
                description=raw.get("description", ""),
                link=raw.get("link", raw["url"]),
                location=raw.get("location", ""),
                posted_date=posted,
                salary=raw.get("salary", ""),
                content_hash=content_hash,
                application_status="New",
                last_seen=datetime.now(timezone.utc),
            )
            normalized.append(job)
        except Exception:
            logger.exception("Failed to normalize job %s", raw.get("url"))

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


def scrape_company_pages() -> list[JobSQL]:
    """Run the agentic scraping workflow for active companies.

    Returns:
        list[JobSQL]: List of normalized job objects scraped from company pages.
    """
    companies = load_active_companies()
    if not companies:
        logger.info("No active companies to scrape.")
        return []

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

    initial_state = {"companies": companies}
    try:
        final_state = graph.invoke(initial_state)
        return final_state.get("normalized_jobs", [])
    except Exception:
        logger.exception("Workflow failed")
        return []
