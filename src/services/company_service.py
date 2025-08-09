"""Company service for managing company data operations.

This module provides the CompanyService class with static methods for querying
and updating company records. It handles database operations for company creation,
status management, and active company filtering.

The service includes:
- Company CRUD operations with validation and error handling
- Bulk scraping statistics updates with weighted success rates
- Active company management for scraping workflows
- Company statistics with job counts via optimized queries
- Performance monitoring decorators for observability

All methods use context-managed database sessions and comprehensive error handling
with proper logging. DTO conversion prevents DetachedInstanceError issues.
"""

import logging

from datetime import datetime, timezone

# Import Any for proper type hints
from typing import Any

from sqlmodel import func, select
from src.database import db_session
from src.database_listeners.monitoring_listeners import performance_monitor
from src.models import CompanySQL, JobSQL
from src.schemas import Company

logger = logging.getLogger(__name__)

# Type aliases for better readability
type CompanyStatsList = list[dict[str, Any]]
type ScrapeUpdateBatch = list[dict[str, Any]]


def calculate_weighted_success_rate(
    current_rate: float, scrape_count: int, success: bool, weight: float = 0.8
) -> float:
    """Calculate weighted-average success rate for scraping statistics.

    Computes a running average that gives more weight to historical performance
    while still responding to recent results. Uses exponential moving average.

    Args:
        current_rate: Current success rate between 0.0 and 1.0.
        scrape_count: Total number of scrapes performed (includes current).
        success: Whether the latest scrape attempt was successful.
        weight: Weight for historical data, default 0.8 (20% weight to new result).

    Returns:
        Updated success rate between 0.0 and 1.0.

    Examples:
        >>> rate = calculate_weighted_success_rate(0.9, 10, True, 0.8)
        >>> print(f"{rate:.2f}")  # Weighted average closer to 0.9
        >>> rate = calculate_weighted_success_rate(0.0, 1, True, 0.8)
        >>> print(f"{rate:.2f}")  # 1.0 (first scrape)
    """
    if scrape_count == 1:
        return 1.0 if success else 0.0

    current_weight = 1 - weight
    new_success = 1.0 if success else 0.0
    return weight * current_rate + current_weight * new_success


class CompanyService:
    """Service class for company data operations.

    Provides static methods for querying, creating, and updating company records
    in the database. This service acts as an abstraction layer between the UI
    and the database models.

    All methods are static and decorated with performance monitoring for observability.
    The service uses context-managed database sessions and converts SQLModel objects
    to Pydantic DTOs to prevent DetachedInstanceError issues.

    Key features:
    - Input validation for company names and URLs
    - Weighted success rate calculations for scraping statistics
    - Bulk operations for performance optimization
    - Optimized queries with job counts to prevent N+1 problems
    - Comprehensive error handling with detailed logging
    """

    @staticmethod
    def _to_dto(company_sql: CompanySQL) -> Company:
        """Convert a single SQLModel object to its Pydantic DTO.

        Helper method for consistent DTO conversion that eliminates
        DetachedInstanceError by creating clean Pydantic objects without
        database session dependencies.

        Args:
            company_sql: SQLModel CompanySQL object to convert with all
                fields populated.

        Returns:
            Company DTO object with all fields copied from the SQLModel instance.

        Raises:
            ValidationError: If SQLModel data doesn't match DTO schema.
        """
        return Company.model_validate(company_sql)

    @classmethod
    def _to_dtos(cls, companies_sql: list[CompanySQL]) -> list[Company]:
        """Convert a list of SQLModel objects to Pydantic DTOs efficiently.

        Batch conversion helper that processes multiple SQLModel objects using
        the single-object conversion method for consistency.

        Args:
            companies_sql: List of SQLModel CompanySQL objects to convert.

        Returns:
            List of Company DTO objects in the same order as input.

        Raises:
            ValidationError: If any SQLModel data doesn't match DTO schema.
        """
        return [cls._to_dto(c) for c in companies_sql]

    @staticmethod
    @performance_monitor
    def get_all_companies() -> list[Company]:
        """Get all companies ordered by name.

        Returns:
            List of all Company DTO objects ordered alphabetically by name.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                companies_sql = session.exec(
                    select(CompanySQL).order_by(CompanySQL.name)
                ).all()

                # Convert SQLModel objects to Pydantic DTOs
                companies = CompanyService._to_dtos(companies_sql)

                logger.info("Retrieved %d companies", len(companies))
                return companies

        except Exception:
            logger.exception("Failed to get all companies")
            raise

    @staticmethod
    @performance_monitor
    def add_company(name: str, url: str) -> Company:
        """Add a new company to the database.

        Args:
            name: Company name (must be unique).
            url: Company careers URL.

        Returns:
            Newly created Company DTO object.

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
                    error_msg = f"Company '{name}' already exists"
                    raise ValueError(error_msg)

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
                session.refresh(company)  # Ensure all fields are populated

                # Convert to DTO before returning
                company_dto = CompanyService._to_dto(company)
                logger.info("Added new company: %s (ID: %s)", name, company.id)
                return company_dto

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
                    error_msg = f"Company with ID {company_id} not found"
                    raise ValueError(error_msg)

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
    def get_active_companies() -> list[Company]:
        """Get all active companies ordered by name.

        Returns:
            List of active Company DTO objects ordered alphabetically by name.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                companies_sql = session.exec(
                    select(CompanySQL)
                    .filter(CompanySQL.active.is_(True))
                    .order_by(CompanySQL.name)
                ).all()

                # Convert SQLModel objects to Pydantic DTOs
                companies = CompanyService._to_dtos(companies_sql)

                logger.info("Retrieved %d active companies", len(companies))
                return companies

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
                last_scraped = datetime.now(timezone.utc)

            with db_session() as session:
                company = session.exec(
                    select(CompanySQL).filter_by(id=company_id)
                ).first()
                if not company:
                    error_msg = f"Company with ID {company_id} not found"
                    raise ValueError(error_msg)

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
    def get_company_by_id(company_id: int) -> Company | None:
        """Get a single company by its ID.

        Args:
            company_id: Database ID of the company to retrieve.

        Returns:
            Company DTO object if found, None otherwise.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                if company_sql := session.exec(
                    select(CompanySQL).filter_by(id=company_id)
                ).first():
                    company = CompanyService._to_dto(company_sql)
                    logger.info("Retrieved company %s: %s", company_id, company.name)
                    return company
                logger.warning("Company with ID %s not found", company_id)
                return None

        except Exception:
            logger.exception("Failed to get company %s", company_id)
            raise

    @staticmethod
    @performance_monitor
    def get_company_by_name(name: str) -> Company | None:
        """Get a company by its name.

        Args:
            name: Company name to search for.

        Returns:
            Company DTO object if found, None otherwise.

        Raises:
            Exception: If database query fails.
        """
        try:
            if not name or not name.strip():
                return None

            with db_session() as session:
                if company_sql := session.exec(
                    select(CompanySQL).filter_by(name=name.strip())
                ).first():
                    company = CompanyService._to_dto(company_sql)
                    logger.info(
                        "Retrieved company by name '%s': ID %s", name, company.id
                    )
                    return company
                logger.info("Company with name '%s' not found", name)
                return None

        except Exception:
            logger.exception("Failed to get company by name '%s'", name)
            raise

    @staticmethod
    @performance_monitor
    def get_companies_with_job_counts() -> CompanyStatsList:
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
    def bulk_update_scrape_stats(updates: ScrapeUpdateBatch) -> int:
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
                        "last_scraped", datetime.now(timezone.utc)
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

    @staticmethod
    @performance_monitor
    def get_companies_for_management() -> list[dict[str, Any]]:
        """Get all companies formatted for management UI display.

        Returns:
            List of dictionaries with company data for management interface.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                companies_sql = session.exec(
                    select(CompanySQL).order_by(CompanySQL.name)
                ).all()

                companies_data = [
                    {
                        "id": company.id,
                        "Name": company.name,
                        "URL": company.url,
                        "Active": company.active,
                    }
                    for company in companies_sql
                ]

                logger.info(
                    "Retrieved %d companies for management", len(companies_data)
                )
                return companies_data

        except Exception:
            logger.exception("Failed to get companies for management")
            raise

    @staticmethod
    @performance_monitor
    def update_company_active_status(company_id: int, active: bool) -> bool:
        """Update the active status of a company.

        Args:
            company_id: Database ID of the company to update.
            active: New active status.

        Returns:
            True if update was successful.

        Raises:
            Exception: If database update fails or company not found.
        """
        try:
            with db_session() as session:
                company = session.exec(
                    select(CompanySQL).filter_by(id=company_id)
                ).first()
                if not company:
                    error_msg = f"Company with ID {company_id} not found"
                    raise ValueError(error_msg)

                old_status = company.active
                company.active = active

                logger.info(
                    "Updated company '%s' active status from %s to %s",
                    company.name,
                    old_status,
                    company.active,
                )
                return True

        except ValueError:
            raise
        except Exception:
            logger.exception(
                "Failed to update company active status for ID %s", company_id
            )
            raise

    @staticmethod
    @performance_monitor
    def get_active_companies_count() -> int:
        """Get the count of active companies.

        Returns:
            Number of active companies.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                count_result = session.exec(
                    select(func.count(CompanySQL.id)).where(CompanySQL.active.is_(True))
                ).one()

                # Extract scalar value from potential tuple result
                count = (
                    count_result[0] if isinstance(count_result, tuple) else count_result
                )

                logger.info("Retrieved active companies count: %d", count)
                return count

        except Exception:
            logger.exception("Failed to get active companies count")
            raise
