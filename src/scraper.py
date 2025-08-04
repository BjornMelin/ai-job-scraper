"""Main scraper module for the AI Job Scraper application.

This module combines scraping from job boards and company career pages, applies
relevance filtering using regex on job titles, deduplicates by link, and updates
the database by upserting current jobs and deleting stale ones. It preserves
user-defined fields like favorites during updates.
"""

import hashlib
import logging

from datetime import datetime

import sqlmodel
import typer

from langgraph.graph import END, StateGraph

from .constants import AI_REGEX, SEARCH_KEYWORDS, SEARCH_LOCATIONS
from .database import SessionLocal
from .models import CompanySQL, JobSQL
from .scraper_company_pages import (
    State,
    extract_details,
    extract_job_lists,
    load_active_companies,
    normalize_jobs,
)
from .scraper_job_boards import scrape_job_boards
from .utils import random_delay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_or_create_company(session: sqlmodel.Session, company_name: str) -> int:
    """Get existing company ID or create new company.

    Args:
        session: Database session.
        company_name: Name of the company.

    Returns:
        int: Company ID.
    """
    company = session.exec(
        sqlmodel.select(CompanySQL).where(CompanySQL.name == company_name)
    ).first()

    if not company:
        # Create a new company with basic info
        company = CompanySQL(
            name=company_name,
            url="",  # Will be updated later if available
            active=True,
        )
        session.add(company)
        session.commit()
        session.refresh(company)

    return company.id


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

    # Normalize board jobs to JobSQL - need to handle new schema
    board_jobs: list[JobSQL] = []
    session = SessionLocal()
    try:
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
                # Get or create company
                company_name = raw.get("company", "Unknown")
                company_id = get_or_create_company(session, company_name)

                # Create content hash
                title = raw.get("title", "")
                description = raw.get("description", "")
                company = raw.get("company", "")
                content = f"{title}{description}{company}"
                content_hash = hashlib.md5(content.encode()).hexdigest()

                job = JobSQL(
                    title=raw.get("title", ""),
                    company_id=company_id,
                    description=raw.get("description", ""),
                    location=raw.get("location", ""),
                    link=raw.get("job_url", ""),
                    posted_date=raw.get("date_posted"),
                    salary=salary,
                    content_hash=content_hash,
                    application_status="New",
                    last_seen=datetime.now(),
                )
                board_jobs.append(job)
            except Exception as e:
                logger.error(f"Failed to normalize board job {raw.get('job_url')}: {e}")
    finally:
        session.close()

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
    archives stale jobs no longer present in the current scrape.

    Args:
        jobs: List of normalized JobSQL instances to upsert.
    """
    session = SessionLocal()
    try:
        current_links = {job.link for job in jobs if job.link}

        # Upsert jobs
        for job in jobs:
            if not job.link:
                continue

            existing = session.exec(
                sqlmodel.select(JobSQL).where(JobSQL.link == job.link)
            ).first()

            if existing:
                # Update job fields while preserving user data
                existing.title = job.title
                existing.company_id = job.company_id
                existing.description = job.description
                existing.location = job.location
                existing.posted_date = job.posted_date
                existing.salary = job.salary
                existing.content_hash = job.content_hash
                existing.last_seen = datetime.now()
                # Preserve: favorite, notes, application_status, application_date
            else:
                session.add(job)

        # Archive stale jobs instead of deleting them
        all_db_jobs = session.exec(
            sqlmodel.select(JobSQL).where(not JobSQL.archived)
        ).all()

        for db_job in all_db_jobs:
            if db_job.link not in current_links:
                # Only archive if it has user data, otherwise delete
                if (
                    db_job.favorite
                    or db_job.notes
                    or db_job.application_status != "New"
                ):
                    db_job.archived = True
                else:
                    session.delete(db_job)

        session.commit()
        logger.info(f"Updated database with {len(jobs)} jobs")

    except Exception as e:
        logger.error(f"Database update failed: {e}")
        session.rollback()
        raise
    finally:
        session.close()


app = typer.Typer()


@app.command()
def scrape() -> None:
    """CLI command to run the full scraping workflow."""
    scrape_all()


if __name__ == "__main__":
    app()
