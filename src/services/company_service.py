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

import sqlalchemy.exc
import sqlmodel

from sqlalchemy.orm import selectinload

# Import streamlit for caching decorators
try:
    import streamlit as st
except ImportError:
    # Create dummy decorator for non-Streamlit environments
    class _DummyStreamlit:
        @staticmethod
        def cache_data(**_kwargs):
            def decorator(func):
                return func

            return decorator

    st = _DummyStreamlit()

from sqlmodel import func, select
from src.database import db_session
from src.models import CompanySQL, JobSQL
from src.schemas import Company
from src.ui.utils.ui_helpers import (
    calculate_active_jobs_count,
    calculate_total_jobs_count,
)

logger = logging.getLogger(__name__)

# Type aliases for better readability
type CompanyMapping = dict[str, int]
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
    @st.cache_data(ttl=60)  # Cache for 1 minute
    def get_all_companies() -> list[Company]:
        """Get all companies ordered by name.

        Returns:
            List of all Company DTO objects ordered alphabetically by name.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Use selectinload to eagerly load job relationships
                companies_sql = session.exec(
                    select(CompanySQL)
                    .options(selectinload(CompanySQL.jobs))
                    .order_by(CompanySQL.name)
                ).all()

                # Convert SQLModel objects to Pydantic DTOs
                companies = CompanyService._to_dtos(companies_sql)

                logger.info("Retrieved %d companies", len(companies))
                return companies

        except Exception:
            logger.exception("Failed to get all companies")
            raise

    @staticmethod
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
    @st.cache_data(ttl=30)  # Cache for 30 seconds (more dynamic)
    def get_active_companies() -> list[Company]:
        """Get all active companies ordered by name.

        Returns:
            List of active Company DTO objects ordered alphabetically by name.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Use selectinload to eagerly load job relationships
                companies_sql = session.exec(
                    select(CompanySQL)
                    .options(selectinload(CompanySQL.jobs))
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
    def delete_company(company_id: int) -> bool:
        """Delete a company and all associated jobs.

        Removes company record and cascades deletion to all related job records.
        Provides comprehensive logging of deletion operation.

        Args:
            company_id: Database ID of the company to delete.

        Returns:
            True if company was successfully deleted, False if not found.

        Raises:
            Exception: If database operation fails.
        """
        try:
            with db_session() as session:
                # First check if company exists
                company = session.exec(
                    select(CompanySQL).filter_by(id=company_id)
                ).first()

                if not company:
                    logger.warning(
                        "Company with ID %s not found for deletion", company_id
                    )
                    return False

                company_name = company.name

                # Delete associated jobs first (explicit cascade)
                job_count = session.exec(
                    select(func.count(JobSQL.id)).where(JobSQL.company_id == company_id)
                ).first()

                if job_count:
                    session.exec(
                        sqlmodel.delete(JobSQL).where(JobSQL.company_id == company_id)
                    )
                    logger.info(
                        "Deleting %d jobs associated with company '%s'",
                        job_count,
                        company_name,
                    )

                # Delete the company
                session.delete(company)
                session.commit()

                logger.info(
                    "Successfully deleted company '%s' (ID: %s) and %d associated jobs",
                    company_name,
                    company_id,
                    job_count or 0,
                )
                return True

        except Exception:
            logger.exception("Failed to delete company with ID %s", company_id)
            raise

    @staticmethod
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
    @st.cache_data(ttl=180)  # Cache for 3 minutes
    def get_companies_with_job_counts() -> CompanyStatsList:
        """Get all companies with their job counts in a single optimized query.

        This method uses a LEFT JOIN to efficiently retrieve company data along
        with job counts, avoiding N+1 query problems when displaying statistics.

        Uses simple Streamlit caching for improved performance.

        Returns:
            List of dictionaries containing company data and job statistics.

        Raises:
            Exception: If database query fails.
        """
        try:
            with db_session() as session:
                # Use selectinload to load job relationships, then calculate in Python
                # This leverages computed properties efficiently
                companies_sql = session.exec(
                    select(CompanySQL)
                    .options(selectinload(CompanySQL.jobs))
                    .order_by(CompanySQL.name)
                ).all()

                companies_with_stats = [
                    {
                        "company": company,
                        "total_jobs": calculate_total_jobs_count(company.jobs),
                        "active_jobs": calculate_active_jobs_count(company.jobs),
                    }
                    for company in companies_sql
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
                # Step 1: Bulk load all companies to update in a single query
                company_ids = [update["company_id"] for update in updates]
                companies = session.exec(
                    select(CompanySQL).where(CompanySQL.id.in_(company_ids))
                ).all()

                # Step 2: Create lookup dict for efficient updates
                companies_by_id = {comp.id: comp for comp in companies}

                # Step 3: Apply updates using business logic
                updated_count = 0
                for update in updates:
                    company_id = update["company_id"]
                    success = update["success"]
                    last_scraped = update.get(
                        "last_scraped", datetime.now(timezone.utc)
                    )

                    if company := companies_by_id.get(company_id):
                        company.scrape_count += 1

                        # Calculate new success rate using weighted average helper
                        company.success_rate = calculate_weighted_success_rate(
                            company.success_rate, company.scrape_count, success
                        )

                        company.last_scraped = last_scraped
                        updated_count += 1

                logger.info("Updated scrape stats for %d companies", updated_count)
                return updated_count

        except Exception:
            logger.exception("Failed to bulk update scrape stats")
            raise

    @staticmethod
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
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_active_companies_count() -> int:
        """Get the count of active companies.

        Uses simple Streamlit caching for improved performance.

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

    @staticmethod
    def bulk_get_or_create_companies(
        session: sqlmodel.Session, company_names: set[str]
    ) -> CompanyMapping:
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

        # Step 1: Bulk load existing companies in single query with job relationships
        existing_companies = session.exec(
            sqlmodel.select(CompanySQL)
            .options(selectinload(CompanySQL.jobs))
            .where(CompanySQL.name.in_(company_names))
        ).all()
        company_map = {comp.name: comp.id for comp in existing_companies}

        # Step 2: Identify missing companies
        missing_names = company_names - set(company_map.keys())

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
                logger.info("Bulk created %d new companies", len(missing_names))
            except sqlalchemy.exc.IntegrityError:
                # Handle race condition: another process created some companies
                # Roll back and re-query to get the actual IDs
                session.rollback()

                # Re-query for all companies that were supposed to be missing
                retry_companies = session.exec(
                    sqlmodel.select(CompanySQL).where(
                        CompanySQL.name.in_(missing_names)
                    )
                ).all()

                # Update the mapping with companies that were created by other processes
                company_map |= {comp.name: comp.id for comp in retry_companies}

                # Create only the companies that are still truly missing
                if still_missing := missing_names - {
                    comp.name for comp in retry_companies
                }:
                    remaining_companies = [
                        CompanySQL(name=name, url="", active=True)
                        for name in still_missing
                    ]
                    session.add_all(remaining_companies)
                    session.flush()
                    company_map |= {comp.name: comp.id for comp in remaining_companies}
                    logger.info(
                        "Bulk created %d new companies (after handling race condition)",
                        len(still_missing),
                    )
                else:
                    logger.info(
                        "No new companies to create "
                        "(all were created by other processes)"
                    )

        logger.debug(
            "Bulk processed %d companies: %d existing, %d new",
            len(company_names),
            len(existing_companies),
            len(missing_names),
        )

        return company_map

    @staticmethod
    def bulk_delete_companies(company_ids: list[int]) -> int:
        """Delete multiple companies and their associated jobs in a single operation.

        Uses efficient bulk operations to delete companies and cascade to jobs,
        avoiding N+1 query problems for large datasets.

        Args:
            company_ids: List of company database IDs to delete.

        Returns:
            Number of companies successfully deleted.

        Raises:
            Exception: If bulk delete operation fails.
        """
        if not company_ids:
            return 0

        try:
            with db_session() as session:
                # First get company names for logging before deletion
                companies_to_delete = session.exec(
                    select(CompanySQL.name, CompanySQL.id).where(
                        CompanySQL.id.in_(company_ids)
                    )
                ).all()

                if not companies_to_delete:
                    logger.warning("No companies found for bulk deletion")
                    return 0

                company_names = [comp.name for comp in companies_to_delete]
                found_ids = [comp.id for comp in companies_to_delete]

                # Count associated jobs for logging
                job_count = (
                    session.exec(
                        select(func.count(JobSQL.id)).where(
                            JobSQL.company_id.in_(found_ids)
                        )
                    ).first()
                    or 0
                )

                # Delete associated jobs first (foreign key constraint)
                if job_count > 0:
                    session.exec(
                        sqlmodel.delete(JobSQL).where(JobSQL.company_id.in_(found_ids))
                    )
                    logger.info(
                        "Bulk deleted %d jobs associated with companies", job_count
                    )

                # Delete companies using bulk operation
                deleted_count = session.exec(
                    sqlmodel.delete(CompanySQL).where(CompanySQL.id.in_(found_ids))
                ).rowcount

                logger.info(
                    "Bulk deleted %d companies: %s (and %d associated jobs)",
                    deleted_count,
                    ", ".join(company_names),
                    job_count,
                )
                return deleted_count

        except Exception:
            logger.exception("Failed to bulk delete companies")
            raise

    @staticmethod
    def bulk_update_status(company_ids: list[int], active: bool) -> int:
        """Update the active status of multiple companies in a single operation.

        Uses efficient bulk update to modify company active status,
        avoiding N+1 query problems for large datasets.

        Args:
            company_ids: List of company database IDs to update.
            active: New active status to set.

        Returns:
            Number of companies successfully updated.

        Raises:
            Exception: If bulk update operation fails.
        """
        if not company_ids:
            return 0

        try:
            with db_session() as session:
                # Get company names for logging
                companies_to_update = session.exec(
                    select(CompanySQL.name, CompanySQL.id).where(
                        CompanySQL.id.in_(company_ids)
                    )
                ).all()

                if not companies_to_update:
                    logger.warning("No companies found for bulk status update")
                    return 0

                company_names = [comp.name for comp in companies_to_update]
                found_ids = [comp.id for comp in companies_to_update]

                # Update companies using bulk operation
                updated_count = session.exec(
                    sqlmodel.update(CompanySQL)
                    .where(CompanySQL.id.in_(found_ids))
                    .values(active=active)
                ).rowcount

                status_text = "activated" if active else "deactivated"
                logger.info(
                    "Bulk %s %d companies: %s",
                    status_text,
                    updated_count,
                    ", ".join(company_names),
                )
                return updated_count

        except Exception:
            logger.exception("Failed to bulk update company status")
            raise
