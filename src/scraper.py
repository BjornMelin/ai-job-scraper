"""Main scraper module for the AI Job Scraper application.

This module combines scraping from job boards and company career pages, applies
relevance filtering using regex on job titles, deduplicates by link, and updates
the database by upserting current jobs and deleting stale ones. It preserves
user-defined fields like favorites during updates.
"""

import hashlib
import logging

from datetime import datetime

import sqlalchemy.exc
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

    Note:
        This function is kept for backward compatibility but should be avoided
        in loops. Use bulk_get_or_create_companies() for better performance.
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


def bulk_get_or_create_companies(
    session: sqlmodel.Session, company_names: set[str]
) -> dict[str, int]:
    """Efficiently get or create multiple companies in bulk.

    This function eliminates N+1 query patterns by:
    1. Bulk loading existing companies in a single query
    2. Bulk creating missing companies
    3. Returning a name->ID mapping for O(1) lookups

    Args:
        session: Database session.
        company_names: Set of unique company names to process.

    Returns:
        dict[str, int]: Mapping of company names to their database IDs.
    """
    if not company_names:
        return {}

    # Step 1: Bulk load existing companies in single query
    existing_companies = session.exec(
        sqlmodel.select(CompanySQL).where(CompanySQL.name.in_(company_names))
    ).all()
    company_map = {comp.name: comp.id for comp in existing_companies}

    # Step 2: Identify missing companies
    missing_names = company_names - company_map.keys()

    # Step 3: Bulk create missing companies if any, handling race conditions
    if missing_names:
        new_companies = [
            CompanySQL(name=name, url="", active=True) for name in missing_names
        ]
        session.add_all(new_companies)

        try:
            session.flush()  # Get IDs without committing transaction
            # Add new companies to the mapping
            company_map |= {comp.name: comp.id for comp in new_companies}
            logger.info(f"Bulk created {len(missing_names)} new companies")
        except sqlalchemy.exc.IntegrityError:
            # Handle race condition: another process created some companies
            # Roll back and re-query to get the actual IDs
            session.rollback()

            # Re-query for all companies that were supposed to be missing
            retry_companies = session.exec(
                sqlmodel.select(CompanySQL).where(CompanySQL.name.in_(missing_names))
            ).all()

            # Update the mapping with companies that were created by other processes
            company_map |= {comp.name: comp.id for comp in retry_companies}

            # Create only the companies that are still truly missing
            if still_missing := missing_names - {comp.name for comp in retry_companies}:
                remaining_companies = [
                    CompanySQL(name=name, url="", active=True) for name in still_missing
                ]
                session.add_all(remaining_companies)
                session.flush()
                company_map |= {comp.name: comp.id for comp in remaining_companies}
                logger.info(
                    f"Bulk created {len(still_missing)} new companies "
                    f"(after handling race condition)"
                )
            else:
                logger.info(
                    "No new companies to create (all were created by other processes)"
                )

    logger.debug(
        f"Bulk processed {len(company_names)} companies: "
        f"{len(existing_companies)} existing, {len(missing_names)} new"
    )

    return company_map


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
            f"Only {total_scraped} jobs scraped total, which is suspiciously low. "
            "This might indicate scraping issues. Proceeding with caution..."
        )

    # Step 5: Combine and filter relevant jobs
    all_jobs = company_jobs + board_jobs
    filtered_jobs = [job for job in all_jobs if AI_REGEX.search(job.title)]
    logger.info(f"Filtered to {len(filtered_jobs)} AI/ML relevant jobs")

    # Step 6: Deduplicate by link, keeping the last occurrence
    job_dict = {job.link: job for job in filtered_jobs if job.link}
    dedup_jobs = list(job_dict.values())
    logger.info(f"Deduplicated to {len(dedup_jobs)} unique jobs")

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

    logger.info("Scraping workflow completed successfully")
    return sync_stats


def _normalize_board_jobs(board_jobs_raw: list[dict]) -> list[JobSQL]:
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
        company_map = bulk_get_or_create_companies(session, company_names)
        logger.info(f"Bulk processed {len(company_names)} unique companies")

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
                        f"No company ID found for '{company_name}', skipping job"
                    )
                    continue

                # Create content hash for change detection
                # Using MD5 for non-cryptographic fingerprinting
                # (performance over security)
                title = raw.get("title", "")
                description = raw.get("description", "")
                company = raw.get("company", "")
                content = f"{title}{description}{company}"
                content_hash = hashlib.md5(content.encode()).hexdigest()  # noqa: S324

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

        # Step 4: Commit company changes before returning jobs
        session.commit()
        logger.info(f"Successfully normalized {len(board_jobs)} board jobs")

    except Exception as e:
        logger.error(f"Failed to normalize board jobs: {e}")
        session.rollback()
        raise
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
