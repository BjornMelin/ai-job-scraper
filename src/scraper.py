"""Main scraper module for the AI Job Scraper application.

This module combines scraping from job boards and company career pages, applies
relevance filtering using regex on job titles, deduplicates by link, and updates
the database by upserting current jobs and deleting stale ones. It preserves
user-defined fields like favorites during updates.
"""

import logging

from collections.abc import Sequence
from datetime import datetime, timezone

import sqlmodel
import typer

from .constants import AI_REGEX, SEARCH_KEYWORDS, SEARCH_LOCATIONS
from .database import SessionLocal
from .models import CompanySQL, JobSQL
from .scraper_company_pages import DEFAULT_MAX_JOBS_PER_COMPANY, scrape_company_pages
from .scraper_job_boards import scrape_job_boards
from .services.company_service import CompanyService
from .services.database_sync import SmartSyncEngine
from .utils import random_delay

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type aliases for better readability
type CompanyMapping = dict[str, int]
type SyncStats = dict[str, int]


def get_or_create_company(session: sqlmodel.Session, company_name: str) -> int:
    """Get existing company ID or create new company.

    Args:
        session: Database session.
        company_name: Name of the company.

    Returns:
        int: Company ID.

    Note:
        This function is kept for backward compatibility but should be avoided
        in loops. Use CompanyService.bulk_get_or_create_companies() for better
        performance.
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


def scrape_all(max_jobs_per_company: int | None = None) -> SyncStats:
    """Run the full scraping workflow with intelligent database synchronization.

    This function orchestrates scraping from company pages and job boards,
    normalizes the data, filters for relevant AI/ML jobs using regex,
    deduplicates by job link, and uses SmartSyncEngine for safe database updates.

    Args:
        max_jobs_per_company: Optional limit for jobs per company.
                            If None, defaults to 50.

    Returns:
        dict[str, int]: Synchronization statistics from SmartSyncEngine.

    Raises:
        ValueError: If max_jobs_per_company is not a positive integer.
        Exception: If any part of the scraping or normalization fails, errors are
            logged but the function continues where possible.
    """
    logger.info("=" * 60)
    logger.info("🚀 STARTING COMPREHENSIVE JOB SCRAPING WORKFLOW")
    logger.info("=" * 60)
    start_time = datetime.now(timezone.utc)

    # Validate max_jobs_per_company parameter
    if max_jobs_per_company is not None:
        if not isinstance(max_jobs_per_company, int):
            raise ValueError("max_jobs_per_company must be an integer")
        if max_jobs_per_company < 1:
            raise ValueError("max_jobs_per_company must be at least 1")

    # Log the job limit being used
    limit = max_jobs_per_company or DEFAULT_MAX_JOBS_PER_COMPANY
    logger.info("Using job limit: %d jobs per company", limit)

    # Step 1: Scrape company pages using the decoupled workflow
    logger.info("Scraping company career pages...")
    try:
        company_jobs = scrape_company_pages(max_jobs_per_company)
        logger.info("Retrieved %d jobs from company pages", len(company_jobs))
    except Exception:
        logger.exception("Company scraping failed")
        company_jobs = []

    random_delay()

    # Step 2: Scrape job boards
    logger.info("Scraping job boards...")
    try:
        board_jobs_raw = scrape_job_boards(SEARCH_KEYWORDS, SEARCH_LOCATIONS)
        logger.info("Retrieved %d raw jobs from job boards", len(board_jobs_raw))
    except Exception:
        logger.exception("Job board scraping failed")
        board_jobs_raw = []

    # Step 3: Normalize board jobs to JobSQL objects
    board_jobs = _normalize_board_jobs(board_jobs_raw)
    logger.info("Normalized %d jobs from job boards", len(board_jobs))

    # Step 4: Safety guard against mass-archiving when both scrapers fail
    if not company_jobs and not board_jobs:
        logger.warning(
            "Both company pages and job boards scrapers returned empty results. "
            "This could indicate scraping failures. Skipping sync to prevent "
            "mass-archiving of existing jobs."
        )
        return {"inserted": 0, "updated": 0, "archived": 0, "deleted": 0, "skipped": 0}

    # Additional safety check for suspiciously low job counts
    total_scraped = len(company_jobs) + len(board_jobs)
    if total_scraped < 5:  # Configurable threshold
        logger.warning(
            "Only %d jobs scraped total, which is suspiciously low. "
            "This might indicate scraping issues. Proceeding with caution...",
            total_scraped,
        )

    # Step 5: Combine jobs from both sources
    all_jobs = company_jobs + board_jobs
    logger.info("Combined %d jobs from company pages and job boards", len(all_jobs))

    # Step 5a: Filter jobs to only include those from active companies
    try:
        from .services.job_service import JobService

        active_company_names = set(JobService.get_active_companies())
        logger.info(
            "Filtering jobs to only include %d active companies",
            len(active_company_names),
        )

        # Filter jobs by company name (get company name from company_id relationship)
        company_filtered_jobs = []
        with SessionLocal() as session:
            for job in all_jobs:
                # Get company name from company_id
                if hasattr(job, "company_id") and job.company_id:
                    company = session.exec(
                        sqlmodel.select(CompanySQL).where(
                            CompanySQL.id == job.company_id
                        )
                    ).first()
                    if company and company.name in active_company_names:
                        company_filtered_jobs.append(job)
                    elif company:
                        logger.debug(
                            "Excluding job from inactive company: %s", company.name
                        )
                else:
                    logger.warning("Job missing company_id, skipping: %s", job.title)

        logger.info(
            "Filtered to %d jobs from active companies (removed %d from inactive)",
            len(company_filtered_jobs),
            len(all_jobs) - len(company_filtered_jobs),
        )

    except Exception:
        logger.exception(
            "Failed to filter by active companies, proceeding with all jobs"
        )
        company_filtered_jobs = all_jobs

    # Step 5b: Filter for AI/ML relevant jobs
    filtered_jobs = [job for job in company_filtered_jobs if AI_REGEX.search(job.title)]
    logger.info("Filtered to %d AI/ML relevant jobs", len(filtered_jobs))

    # Step 6: Deduplicate by link, keeping the last occurrence
    job_dict = {job.link: job for job in filtered_jobs if job.link}
    dedup_jobs = list(job_dict.values())
    logger.info("Deduplicated to %d unique jobs", len(dedup_jobs))

    # Step 7: Final safety check before sync
    if not dedup_jobs:
        logger.warning(
            "No valid jobs remaining after filtering and deduplication. "
            "Skipping sync to prevent archiving all existing jobs."
        )
        return {"inserted": 0, "updated": 0, "archived": 0, "deleted": 0, "skipped": 0}

    # Step 8: Use SmartSyncEngine for intelligent database synchronization
    logger.info("Synchronizing jobs with database using SmartSyncEngine...")
    sync_engine = SmartSyncEngine()
    sync_stats = sync_engine.sync_jobs(dedup_jobs)

    # Calculate and log total execution time
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 60)
    logger.info("✅ SCRAPING WORKFLOW COMPLETED SUCCESSFULLY")
    logger.info("📊 Final Statistics:")
    logger.info("  • Total jobs scraped: %d", len(all_jobs))
    logger.info("  • Jobs from active companies: %d", len(company_filtered_jobs))
    logger.info("  • AI/ML relevant jobs: %d", len(filtered_jobs))
    logger.info("  • Unique jobs after deduplication: %d", len(dedup_jobs))
    logger.info("  • Database sync results:")
    logger.info("    - Inserted: %d new jobs", sync_stats.get("inserted", 0))
    logger.info("    - Updated: %d existing jobs", sync_stats.get("updated", 0))
    logger.info("    - Archived: %d stale jobs", sync_stats.get("archived", 0))
    logger.info("    - Deleted: %d jobs", sync_stats.get("deleted", 0))
    logger.info("    - Skipped: %d unchanged jobs", sync_stats.get("skipped", 0))
    logger.info("⏱️  Total execution time: %.2f seconds", duration)
    logger.info("=" * 60)

    return sync_stats


def _normalize_board_jobs(board_jobs_raw: Sequence[dict]) -> list[JobSQL]:
    """Normalize raw job board data to JobSQL objects with optimized bulk operations.

    This function converts dictionaries from job board scrapers into properly
    structured JobSQL objects, handling company creation, salary formatting,
    and content hashing. Uses bulk operations to eliminate N+1 query patterns.

    Args:
        board_jobs_raw: List of raw job dictionaries from job board scrapers.

    Returns:
        list[JobSQL]: List of normalized JobSQL objects ready for sync.
    """
    if not board_jobs_raw:
        return []

    board_jobs: list[JobSQL] = []
    session = SessionLocal()

    try:
        # Step 1: Extract all unique company names for bulk processing
        company_names = {
            raw.get("company", "Unknown").strip()
            for raw in board_jobs_raw
            if raw.get("company", "Unknown").strip()
        }

        # Step 2: Bulk get or create companies (eliminates N+1 queries)
        company_map = CompanyService.bulk_get_or_create_companies(
            session, company_names
        )
        logger.info("Bulk processed %d unique companies", len(company_names))

        # Step 3: Process jobs with O(1) company lookups
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

                # Get company ID from pre-loaded mapping (O(1) lookup)
                company_name = raw.get("company", "Unknown").strip() or "Unknown"
                company_id = company_map.get(company_name, company_map.get("Unknown"))

                if company_id is None:
                    logger.warning(
                        "No company ID found for '%s', skipping job",
                        company_name,
                    )
                    continue

                # Use utility function for data cleaning
                from src.data_cleaning import clean_string_field

                # Use factory method to ensure proper validation and hash generation
                job = JobSQL.create_validated(
                    title=clean_string_field(raw.get("title", "")),
                    company_id=company_id,
                    description=clean_string_field(raw.get("description", "")),
                    location=clean_string_field(raw.get("location", "")),
                    link=clean_string_field(raw.get("job_url", "")),
                    posted_date=raw.get("date_posted"),
                    salary=salary,
                    application_status="New",
                    last_seen=datetime.now(timezone.utc),
                )
                board_jobs.append(job)

            except Exception:
                logger.exception("Failed to normalize board job %s", raw.get("job_url"))

        # Step 4: Commit company changes before returning jobs
        session.commit()
        logger.info("Successfully normalized %d board jobs", len(board_jobs))

    except Exception:
        logger.exception("Failed to normalize board jobs")
        session.rollback()
        raise
    finally:
        session.close()

    return board_jobs


app = typer.Typer()


@app.command()
def scrape(
    max_jobs_per_company: int = typer.Option(
        DEFAULT_MAX_JOBS_PER_COMPANY,
        "--max-jobs",
        "-j",
        help="Maximum number of jobs to scrape per company",
        min=1,
    ),
) -> None:
    """CLI command to run the full scraping workflow."""
    sync_stats = scrape_all(max_jobs_per_company)
    print("\nScraping completed successfully!")
    print("📊 Sync Statistics:")
    print(f"  ✅ Inserted: {sync_stats['inserted']} new jobs")
    print(f"  🔄 Updated: {sync_stats['updated']} existing jobs")
    print(f"  📋 Archived: {sync_stats['archived']} stale jobs with user data")
    print(f"  🗑️  Deleted: {sync_stats['deleted']} stale jobs without user data")
    print(f"  ⏭️  Skipped: {sync_stats['skipped']} jobs (no changes)")


if __name__ == "__main__":
    app()
