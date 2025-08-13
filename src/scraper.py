"""Main scraper module for the AI Job Scraper application.

This module combines scraping from job boards and company career pages, applies
relevance filtering using regex on job titles, deduplicates by link, and updates
the database by upserting current jobs and deleting stale ones. It preserves
user-defined fields like favorites during updates.
"""

import logging

from collections.abc import Sequence
from datetime import UTC, datetime

# Rich imports for beautiful CLI output
import sqlmodel
import typer

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.constants import AI_REGEX, SEARCH_KEYWORDS, SEARCH_LOCATIONS
from src.core_utils import random_delay
from src.database import SessionLocal
from src.models import CompanySQL, JobSQL
from src.scraper_company_pages import DEFAULT_MAX_JOBS_PER_COMPANY, scrape_company_pages
from src.scraper_job_boards import scrape_job_boards
from src.services.company_service import CompanyService
from src.services.database_sync import SmartSyncEngine


class ScraperParameterError(ValueError):
    """Custom exception for invalid scraper configuration parameters."""


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rich console for beautiful terminal output
console = Console()

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
        sqlmodel.select(CompanySQL).where(CompanySQL.name == company_name),
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
    # Rich panel for workflow start
    console.print(
        Panel.fit(
            "🚀 STARTING COMPREHENSIVE JOB SCRAPING WORKFLOW",
            title="[bold blue]AI Job Scraper[/bold blue]",
            style="blue",
        ),
    )
    start_time = datetime.now(UTC)

    # Validate max_jobs_per_company parameter
    if max_jobs_per_company is not None:
        if not isinstance(max_jobs_per_company, int):
            raise ScraperParameterError("max_jobs_per_company must be an integer")
        if max_jobs_per_company < 1:
            raise ScraperParameterError("max_jobs_per_company must be at least 1")

    # Log the job limit being used
    limit = max_jobs_per_company or DEFAULT_MAX_JOBS_PER_COMPANY
    console.print(f"[yellow]Using job limit: {limit} jobs per company[/yellow]")

    # Step 1: Scrape company pages using the decoupled workflow
    console.print("[bold cyan]Step 1:[/bold cyan] Scraping company career pages...")
    try:
        company_jobs = scrape_company_pages(max_jobs_per_company)
        console.print(
            f"[green]✓ Retrieved {len(company_jobs)} jobs from company pages[/green]",
        )
    except Exception:
        logger.exception("Company scraping failed")
        console.print("[red]✗ Company scraping failed[/red]")
        company_jobs = []

    random_delay()

    # Step 2: Scrape job boards
    console.print("[bold cyan]Step 2:[/bold cyan] Scraping job boards...")
    try:
        board_jobs_raw = scrape_job_boards(SEARCH_KEYWORDS, SEARCH_LOCATIONS)
        board_jobs_count = len(board_jobs_raw)
        console.print(
            f"[green]✓ Retrieved {board_jobs_count} raw jobs from job boards[/green]"
        )
    except Exception:
        logger.exception("Job board scraping failed")
        console.print("[red]✗ Job board scraping failed[/red]")
        board_jobs_raw = []

    # Step 3: Normalize board jobs to JobSQL objects
    console.print("[bold cyan]Step 3:[/bold cyan] Normalizing job data...")
    board_jobs = _normalize_board_jobs(board_jobs_raw)
    console.print(f"[green]✓ Normalized {len(board_jobs)} jobs from job boards[/green]")

    # Step 4: Safety guard against mass-archiving when both scrapers fail
    if not company_jobs and not board_jobs:
        logger.warning(
            "Both company pages and job boards scrapers returned empty results. "
            "This could indicate scraping failures. Skipping sync to prevent "
            "mass-archiving of existing jobs.",
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
                match job:
                    case job if hasattr(job, "company_id") and (
                        company := session.exec(
                            sqlmodel.select(CompanySQL).where(
                                CompanySQL.id == job.company_id,
                            )
                        ).first()
                    ):
                        match company.name:
                            case name if name in active_company_names:
                                company_filtered_jobs.append(job)
                            case _:
                                logger.debug(
                                    "Excluding job from inactive company: %s",
                                    company.name,
                                )
                    case _:
                        logger.warning(
                            "Job missing company_id, skipping: %s", job.title
                        )

        logger.info(
            "Filtered to %d jobs from active companies (removed %d from inactive)",
            len(company_filtered_jobs),
            len(all_jobs) - len(company_filtered_jobs),
        )

    except Exception:
        logger.exception(
            "Failed to filter by active companies, proceeding with all jobs",
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
            "Skipping sync to prevent archiving all existing jobs.",
        )
        return {"inserted": 0, "updated": 0, "archived": 0, "deleted": 0, "skipped": 0}

    # Step 8: Use SmartSyncEngine for intelligent database synchronization
    logger.info("Synchronizing jobs with database using SmartSyncEngine...")
    sync_engine = SmartSyncEngine()
    sync_stats = sync_engine.sync_jobs(dedup_jobs)

    # Calculate and log total execution time
    end_time = datetime.now(UTC)
    duration = (end_time - start_time).total_seconds()

    # Create a beautiful Rich table for final statistics
    console.print(
        Panel.fit(
            "✅ SCRAPING WORKFLOW COMPLETED SUCCESSFULLY",
            title="[bold green]Success[/bold green]",
            style="green",
        ),
    )

    # Create statistics table
    stats_table = Table(
        title="📊 Final Statistics",
        show_header=True,
        header_style="bold blue",
    )
    stats_table.add_column("Category", style="cyan", no_wrap=True)
    stats_table.add_column("Count", justify="right", style="magenta")

    # Add job processing stats
    stats_table.add_row("Total jobs scraped", str(len(all_jobs)))
    stats_table.add_row("Jobs from active companies", str(len(company_filtered_jobs)))
    stats_table.add_row("AI/ML relevant jobs", str(len(filtered_jobs)))
    stats_table.add_row("Unique jobs after deduplication", str(len(dedup_jobs)))
    stats_table.add_row("", "")  # Spacer row

    # Add database sync results
    stats_table.add_row(
        "🆕 New jobs inserted",
        str(sync_stats.get("inserted", 0)),
        style="green",
    )
    stats_table.add_row(
        "🔄 Existing jobs updated",
        str(sync_stats.get("updated", 0)),
        style="yellow",
    )
    stats_table.add_row(
        "📋 Jobs archived",
        str(sync_stats.get("archived", 0)),
        style="blue",
    )
    stats_table.add_row(
        "🗑️  Jobs deleted",
        str(sync_stats.get("deleted", 0)),
        style="red",
    )
    stats_table.add_row(
        "⏭️  Jobs skipped (unchanged)",
        str(sync_stats.get("skipped", 0)),
        style="dim",
    )
    stats_table.add_row("", "")  # Spacer row
    stats_table.add_row(
        "⏱️  Total execution time",
        f"{duration:.2f} seconds",
        style="bold",
    )

    console.print(stats_table)

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
            session,
            company_names,
        )
        logger.info("Bulk processed %d unique companies", len(company_names))

        # Step 3: Process jobs with O(1) company lookups
        for raw in board_jobs_raw:
            try:
                # Format salary from min/max amounts
                # Use walrus operator for concise salary formatting
                if (min_amt := raw.get("min_amount")) and (
                    max_amt := raw.get("max_amount")
                ):
                    salary = f"${min_amt}-${max_amt}"
                elif min_amt:
                    salary = f"${min_amt}+"
                elif max_amt := raw.get("max_amount"):
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
                    last_seen=datetime.now(UTC),
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

    # Create a simple summary table for CLI output
    console.print("\n[bold green]Scraping completed successfully![/bold green]")

    summary_table = Table(
        title="📊 Sync Statistics Summary",
        show_header=True,
        header_style="bold blue",
    )
    summary_table.add_column("Operation", style="cyan")
    summary_table.add_column("Count", justify="right", style="magenta")

    summary_table.add_row("✅ New jobs inserted", str(sync_stats["inserted"]))
    summary_table.add_row("🔄 Jobs updated", str(sync_stats["updated"]))
    summary_table.add_row("📋 Jobs archived", str(sync_stats["archived"]))
    summary_table.add_row("🗑️  Jobs deleted", str(sync_stats["deleted"]))
    summary_table.add_row("⏭️  Jobs skipped", str(sync_stats["skipped"]))

    console.print(summary_table)


if __name__ == "__main__":
    app()
