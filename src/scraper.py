"""Main scraper module for the AI Job Scraper application.

This module combines scraping from job boards and company career pages, applies
relevance filtering using regex on job titles, deduplicates by link, and updates
the database by upserting current jobs and deleting stale ones. It preserves
user-defined fields like favorites during updates.
"""

import logging

import sqlmodel
import typer

from langgraph.graph import END, StateGraph

from .config import Settings
from .constants import AI_REGEX, SEARCH_KEYWORDS, SEARCH_LOCATIONS
from .models import JobSQL
from .scraper_company_pages import (
    State,
    extract_details,
    extract_job_lists,
    load_active_companies,
    normalize_jobs,
)
from .scraper_job_boards import scrape_job_boards
from .utils import random_delay

settings = Settings()
engine = sqlmodel.create_engine(settings.db_url)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scrape_all() -> None:
    """Run the full scraping workflow.

    This function orchestrates scraping from company pages and job boards,
    normalizes the data, filters for relevant AI/ML jobs using regex,
    deduplicates by job link, and updates the database.

    Raises:
        Exception: If any part of the scraping or normalization fails, errors are
            logged but the function continues where possible.
    """
    # Scrape company pages using modified workflow to get jobs without saving
    companies = load_active_companies()
    company_jobs: list[JobSQL] = []
    if companies:
        workflow = StateGraph(State)
        workflow.add_node("extract_lists", extract_job_lists)
        workflow.add_node("extract_details", extract_details)
        workflow.add_node("normalize", normalize_jobs)
        workflow.set_entry_point("extract_lists")
        workflow.add_edge("extract_lists", "extract_details")
        workflow.add_edge("extract_details", "normalize")
        workflow.add_edge("normalize", END)
        graph = workflow.compile()
        initial_state = {"companies": companies}
        try:
            final_state = graph.invoke(initial_state)
            company_jobs = final_state.get("normalized_jobs", [])
        except Exception as e:
            logger.error(f"Company scraping workflow failed: {e}")
    random_delay()

    # Scrape job boards
    board_jobs_raw = scrape_job_boards(SEARCH_KEYWORDS, SEARCH_LOCATIONS)

    # Normalize board jobs to JobSQL
    board_jobs: list[JobSQL] = []
    for raw in board_jobs_raw:
        salary = ""
        min_amt = raw.get("min_amount")
        max_amt = raw.get("max_amount")
        if min_amt and max_amt:
            salary = f"${min_amt}-${max_amt}"
        elif min_amt:
            salary = f"${min_amt}+"
        elif max_amt:
            salary = f"${max_amt}"
        try:
            job = JobSQL(
                title=raw.get("title", ""),
                company=raw.get("company", ""),
                description=raw.get("description", ""),
                location=raw.get("location", ""),
                link=raw.get("job_url", ""),
                posted_date=raw.get("date_posted"),
                salary=salary,
            )
            board_jobs.append(job)
        except Exception as e:
            logger.error(f"Failed to normalize board job {raw.get('job_url')}: {e}")

    # Combine and filter relevant jobs
    all_jobs = company_jobs + board_jobs
    filtered_jobs = [job for job in all_jobs if AI_REGEX.search(job.title)]

    # Deduplicate by link, keeping the last occurrence
    job_dict = {job.link: job for job in filtered_jobs if job.link}
    dedup_jobs = list(job_dict.values())

    # Update database
    update_db(dedup_jobs)


def update_db(jobs: list[JobSQL]) -> None:
    """Update the database with scraped jobs.

    This function performs an upsert operation: adds new jobs, updates existing
    ones with fresh scraped data (preserving user fields like favorites), and
    deletes stale jobs no longer present in the current scrape.

    Args:
        jobs: List of normalized JobSQL instances to upsert.
    """
    with sqlmodel.Session(engine) as session:
        current_links = {job.link for job in jobs if job.link}
        # Upsert jobs
        for job in jobs:
            if not job.link:
                continue
            existing = session.exec(
                sqlmodel.select(JobSQL).where(JobSQL.link == job.link)
            ).first()
            if existing:
                existing.title = job.title
                existing.company = job.company
                existing.description = job.description
                existing.location = job.location
                existing.posted_date = job.posted_date
                existing.salary = job.salary
            else:
                session.add(job)
        # Delete stale jobs not in current scrape
        all_db_jobs = session.exec(sqlmodel.select(JobSQL)).all()
        for db_job in all_db_jobs:
            if db_job.link not in current_links:
                session.delete(db_job)
        session.commit()


app = typer.Typer()


@app.command()
def scrape() -> None:
    """CLI command to run the full scraping workflow."""
    scrape_all()


if __name__ == "__main__":
    app()
