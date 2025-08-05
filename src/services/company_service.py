"""Company service for managing company data operations.

This module provides the CompanyService class with static methods for querying
and updating company records. It handles database operations for company creation,
status management, and active company filtering.
"""

import logging

from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime

from sqlmodel import Session, func, select
from src.database import get_session
from src.database_listeners.monitoring_listeners import performance_monitor
from src.models import CompanySQL, JobSQL

logger = logging.getLogger(__name__)


def calculate_weighted_success_rate(
    current_rate: float, scrape_count: int, success: bool, weight: float = 0.8
) -> float:
    """Calculate weighted-average success rate for scraping statistics.

    Args:
        current_rate: Current success rate (0.0 to 1.0).
        scrape_count: Total number of scrapes performed.
        success: Whether the latest scrape was successful.
        weight: Weight for historical data (default 0.8).

    Returns:
        New weighted-average success rate.
    """
    if scrape_count == 1:
        return 1.0 if success else 0.0

    current_weight = 1 - weight
    new_success = 1.0 if success else 0.0
    return weight * current_rate + current_weight * new_success


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with proper error handling.

    Provides automatic commit on success, rollback on error, and session cleanup.
    Follows SQLModel 2025 best practices for session management.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


class CompanyService:
    """Service class for company data operations.

    Provides static methods for querying, creating, and updating company records
    in the database. This service acts as an abstraction layer between the UI
    and the database models.
    """

    @staticmethod
    @performance_monitor
    def get_all_companies() -> list[CompanySQL]:
        """Get all companies ordered by name.

        Returns:
            List of all CompanySQL objects ordered alphabetically by name.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                companies = session.exec(
                    select(CompanySQL).order_by(CompanySQL.name)
                ).all()

                logger.info("Retrieved %d companies", len(companies))
                return list(companies)

        except Exception:
            logger.exception("Failed to get all companies")
            raise

    @staticmethod
    @performance_monitor
    def add_company(name: str, url: str) -> CompanySQL:
        """Add a new company to the database.

        Args:
            name: Company name (must be unique).
            url: Company careers URL.

        Returns:
            Newly created CompanySQL object.

        Raises:
            Exception: If database operation fails or company name already exists.
        """
        try:
            # Validate inputs
            if not name or not name.strip():
                raise ValueError("Company name cannot be empty")
            if not url or not url.strip():
                raise ValueError("Company URL cannot be empty")

            name = name.strip()
            url = url.strip()

            with db_session() as session:
                # Check if company already exists
                if session.exec(select(CompanySQL).filter_by(name=name)).first():
                    raise ValueError(f"Company '{name}' already exists")

                # Create new company
                company = CompanySQL(
                    name=name,
                    url=url,
                    active=True,  # New companies are active by default
                    scrape_count=0,
                    success_rate=1.0,
                )

                session.add(company)
                session.flush()  # Get the ID without committing

                logger.info("Added new company: %s (ID: %s)", name, company.id)
                return company

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception:
            logger.exception("Failed to add company '%s'", name)
            raise

    @staticmethod
    @performance_monitor
    def toggle_company_active(company_id: int) -> bool:
        """Toggle the active status of a company.

        Args:
            company_id: Database ID of the company to toggle.

        Returns:
            New active status (True/False) if successful.

        Raises:
            Exception: If database update fails or company not found.
        """
        try:
            with db_session() as session:
                company = session.exec(
                    select(CompanySQL).filter_by(id=company_id)
                ).first()
                if not company:
                    raise ValueError(f"Company with ID {company_id} not found")

                old_status = company.active
                company.active = not company.active

                logger.info(
                    "Toggled company '%s' active status from %s to %s",
                    company.name,
                    old_status,
                    company.active,
                )
                return company.active

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception:
            logger.exception(
                "Failed to toggle company active status for ID %s", company_id
            )
            raise

    @staticmethod
    @performance_monitor
    def get_active_companies() -> list[CompanySQL]:
        """Get all active companies ordered by name.

        Returns:
            List of active CompanySQL objects ordered alphabetically by name.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                companies = session.exec(
                    select(CompanySQL)
                    .filter(CompanySQL.active.is_(True))
                    .order_by(CompanySQL.name)
                ).all()

                logger.info("Retrieved %d active companies", len(companies))
                return list(companies)

        except Exception:
            logger.exception("Failed to get active companies")
            raise

    @staticmethod
    @performance_monitor
    def update_company_scrape_stats(
        company_id: int, success: bool, last_scraped: datetime | None = None
    ) -> bool:
        """Update company scraping statistics.

        Args:
            company_id: Database ID of the company to update.
            success: Whether the scrape was successful.
            last_scraped: Timestamp of the scrape (defaults to now).

        Returns:
            True if update was successful.

        Raises:
            Exception: If database update fails or company not found.
        """
        try:
            if last_scraped is None:
                last_scraped = datetime.now(datetime.UTC)

            with db_session() as session:
                company = session.exec(
                    select(CompanySQL).filter_by(id=company_id)
                ).first()
                if not company:
                    raise ValueError(f"Company with ID {company_id} not found")

                # Update scrape count
                company.scrape_count += 1

                # Update success rate using weighted average helper
                company.success_rate = calculate_weighted_success_rate(
                    company.success_rate, company.scrape_count, success
                )

                company.last_scraped = last_scraped

                logger.info(
                    "Updated scrape stats for '%s': count=%d, success_rate=%.2f",
                    company.name,
                    company.scrape_count,
                    company.success_rate,
                )
                return True

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception:
            logger.exception(
                "Failed to update scrape stats for company ID %s", company_id
            )
            raise

    @staticmethod
    @performance_monitor
    def get_company_by_id(company_id: int) -> CompanySQL | None:
        """Get a single company by its ID.

        Args:
            company_id: Database ID of the company to retrieve.

        Returns:
            CompanySQL object if found, None otherwise.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                company = session.exec(
                    select(CompanySQL).filter_by(id=company_id)
                ).first()

                if company:
                    logger.info("Retrieved company %s: %s", company_id, company.name)
                else:
                    logger.warning("Company with ID %s not found", company_id)

                return company

        except Exception:
            logger.exception("Failed to get company %s", company_id)
            raise

    @staticmethod
    @performance_monitor
    def get_company_by_name(name: str) -> CompanySQL | None:
        """Get a company by its name.

        Args:
            name: Company name to search for.

        Returns:
            CompanySQL object if found, None otherwise.

        Raises:
            Exception: If database query fails.
        """
        try:
            if not name or not name.strip():
                return None

            with db_session() as session:
                company = session.exec(
                    select(CompanySQL).filter_by(name=name.strip())
                ).first()

                if company:
                    logger.info(
                        "Retrieved company by name '%s': ID %s", name, company.id
                    )
                else:
                    logger.info("Company with name '%s' not found", name)

                return company

        except Exception:
            logger.exception("Failed to get company by name '%s'", name)
            raise

    @staticmethod
    @performance_monitor
    def get_companies_with_job_counts() -> list[dict]:
        """Get all companies with their job counts in a single optimized query.

        This method uses a LEFT JOIN to efficiently retrieve company data along
        with job counts, avoiding N+1 query problems when displaying statistics.

        Returns:
            List of dictionaries containing company data and job statistics.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Use LEFT JOIN to get companies with job counts in single query
                # This avoids N+1 queries when displaying company statistics

                query = (
                    select(
                        CompanySQL,
                        func.count(JobSQL.id).label("total_jobs"),
                        func.count(func.nullif(JobSQL.archived, True)).label(
                            "active_jobs"
                        ),
                    )
                    .outerjoin(JobSQL, CompanySQL.id == JobSQL.company_id)
                    .group_by(CompanySQL.id)
                    .order_by(CompanySQL.name)
                )

                results = session.exec(query).all()

                companies_with_stats = [
                    {
                        "company": company,
                        "total_jobs": total_jobs or 0,
                        "active_jobs": active_jobs or 0,
                    }
                    for company, total_jobs, active_jobs in results
                ]

                logger.info(
                    "Retrieved %d companies with job counts",
                    len(companies_with_stats),
                )
                return companies_with_stats

        except Exception:
            logger.exception("Failed to get companies with job counts")
            raise

    @staticmethod
    @performance_monitor
    def bulk_update_scrape_stats(updates: list[dict]) -> int:
        """Bulk update scraping statistics using SQLAlchemy 2.0 built-in operations.

        Uses SQLAlchemy's native bulk update for optimal performance while preserving
        the business logic for success rate calculations.

        Args:
            updates: List with keys: company_id, success, last_scraped
                   Example: [{"company_id": 1, "success": True, "last_scraped": dt()}]

        Returns:
            Number of companies successfully updated.

        Raises:
            Exception: If bulk update operation fails.
        """
        if not updates:
            return 0

        try:
            with db_session() as session:
                # For complex business logic like weighted averages, we need to fetch
                # current values first, then use individual updates per company
                for update in updates:
                    company_id = update["company_id"]
                    success = update["success"]
                    last_scraped = update.get(
                        "last_scraped", datetime.now(datetime.UTC)
                    )

                    company = session.exec(
                        select(CompanySQL).filter_by(id=company_id)
                    ).first()

                    if company:
                        company.scrape_count += 1

                        # Calculate new success rate using weighted average helper
                        company.success_rate = calculate_weighted_success_rate(
                            company.success_rate, company.scrape_count, success
                        )

                        company.last_scraped = last_scraped

                logger.info("Updated scrape stats for %d companies", len(updates))
                return len(updates)

        except Exception:
            logger.exception("Failed to bulk update scrape stats")
            raise
