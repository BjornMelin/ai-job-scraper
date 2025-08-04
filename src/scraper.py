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

from .constants import AI_REGEX, SEARCH_KEYWORDS, SEARCH_LOCATIONS
from .database import SessionLocal
from .models import CompanySQL, JobSQL
from .scraper_company_pages import scrape_company_pages
from .scraper_job_boards import scrape_job_boards
from .services.database_sync import SmartSyncEngine
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


def scrape_all() -> dict[str, int]:
    """Run the full scraping workflow with intelligent database synchronization.

    This function orchestrates scraping from company pages and job boards,
    normalizes the data, filters for relevant AI/ML jobs using regex,
    deduplicates by job link, and uses SmartSyncEngine for safe database updates.

    Returns:
        dict[str, int]: Synchronization statistics from SmartSyncEngine.

    Raises:
        Exception: If any part of the scraping or normalization fails, errors are
            logged but the function continues where possible.
    """
    logger.info("Starting comprehensive job scraping workflow")

    # Step 1: Scrape company pages using the decoupled workflow
    logger.info("Scraping company career pages...")
    try:
        company_jobs = scrape_company_pages()
        logger.info(f"Retrieved {len(company_jobs)} jobs from company pages")
    except Exception as e:
        logger.error(f"Company scraping failed: {e}")
        company_jobs = []

    random_delay()

    # Step 2: Scrape job boards
    logger.info("Scraping job boards...")
    try:
        board_jobs_raw = scrape_job_boards(SEARCH_KEYWORDS, SEARCH_LOCATIONS)
        logger.info(f"Retrieved {len(board_jobs_raw)} raw jobs from job boards")
    except Exception as e:
        logger.error(f"Job board scraping failed: {e}")
        board_jobs_raw = []

    # Step 3: Normalize board jobs to JobSQL objects
    board_jobs = _normalize_board_jobs(board_jobs_raw)
    logger.info(f"Normalized {len(board_jobs)} jobs from job boards")

    # Step 4: Combine and filter relevant jobs
    all_jobs = company_jobs + board_jobs
    filtered_jobs = [job for job in all_jobs if AI_REGEX.search(job.title)]
    logger.info(f"Filtered to {len(filtered_jobs)} AI/ML relevant jobs")

    # Step 5: Deduplicate by link, keeping the last occurrence
    job_dict = {job.link: job for job in filtered_jobs if job.link}
    dedup_jobs = list(job_dict.values())
    logger.info(f"Deduplicated to {len(dedup_jobs)} unique jobs")

    # Step 6: Use SmartSyncEngine for intelligent database synchronization
    logger.info("Synchronizing jobs with database using SmartSyncEngine...")
    sync_engine = SmartSyncEngine()
    sync_stats = sync_engine.sync_jobs(dedup_jobs)

    logger.info("Scraping workflow completed successfully")
    return sync_stats


def _normalize_board_jobs(board_jobs_raw: list[dict]) -> list[JobSQL]:
    """Normalize raw job board data to JobSQL objects.

    This function converts dictionaries from job board scrapers into properly
    structured JobSQL objects, handling company creation, salary formatting,
    and content hashing.

    Args:
        board_jobs_raw: List of raw job dictionaries from job board scrapers.

    Returns:
        list[JobSQL]: List of normalized JobSQL objects ready for sync.
    """
    board_jobs: list[JobSQL] = []
    session = SessionLocal()

    try:
        for raw in board_jobs_raw:
            try:
                # Format salary from min/max amounts
                salary = ""
                min_amt = raw.get("min_amount")
                max_amt = raw.get("max_amount")
                if min_amt and max_amt:
                    salary = f"${min_amt}-${max_amt}"
                elif min_amt:
                    salary = f"${min_amt}+"
                elif max_amt:
                    salary = f"${max_amt}"

                # Get or create company
                company_name = raw.get("company", "Unknown")
                company_id = get_or_create_company(session, company_name)

                # Create content hash for change detection
                title = raw.get("title", "")
                description = raw.get("description", "")
                company = raw.get("company", "")
                content = f"{title}{description}{company}"
                content_hash = hashlib.md5(content.encode()).hexdigest()

                # Create JobSQL object
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

    return board_jobs


app = typer.Typer()


@app.command()
def scrape() -> None:
    """CLI command to run the full scraping workflow."""
    sync_stats = scrape_all()
    print("\nScraping completed successfully!")
    print("📊 Sync Statistics:")
    print(f"  ✅ Inserted: {sync_stats['inserted']} new jobs")
    print(f"  🔄 Updated: {sync_stats['updated']} existing jobs")
    print(f"  📋 Archived: {sync_stats['archived']} stale jobs with user data")
    print(f"  🗑️  Deleted: {sync_stats['deleted']} stale jobs without user data")
    print(f"  ⏭️  Skipped: {sync_stats['skipped']} jobs (no changes)")


if __name__ == "__main__":
    app()
