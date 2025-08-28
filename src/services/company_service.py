"""Minimal CompanyService replacement for legacy code compatibility.

This module provides a thin wrapper around CompanySQL database operations
to maintain compatibility with existing UI code during the JobSpy migration.
This is a temporary compatibility layer that should eventually be phased out.
"""

import logging

from typing import Any

from sqlmodel import select

from src.database import db_session
from src.models import CompanySQL

logger = logging.getLogger(__name__)


class CompanyService:
    """Minimal company service for legacy compatibility.

    This service provides basic CRUD operations for companies using
    direct database queries. It maintains API compatibility with
    the legacy CompanyService while using library-first patterns.
    """

    @staticmethod
    def get_all_companies() -> list[dict[str, Any]]:
        """Get all companies as dictionaries for UI compatibility.

        Returns:
            List of company dictionaries with id, name, url, and active fields.
        """
        try:
            with db_session() as session:
                result = session.exec(select(CompanySQL))
                companies = result.all()

                return [
                    {
                        "id": company.id,
                        "name": company.name,
                        "url": company.url,
                        "active": company.active,
                    }
                    for company in companies
                ]
        except Exception:
            logger.exception("Failed to get all companies")
            return []

    @staticmethod
    def get_companies_for_management() -> list[dict[str, Any]]:
        """Get companies formatted for management interface.

        Returns:
            List of company dictionaries for management UI.
        """
        # For compatibility, return same as get_all_companies
        return CompanyService.get_all_companies()

    @staticmethod
    def get_active_companies_count() -> int:
        """Get count of active companies.

        Returns:
            Number of active companies.
        """
        try:
            with db_session() as session:
                result = session.exec(
                    select(CompanySQL).where(CompanySQL.active == True)  # noqa: E712
                )
                return len(result.all())
        except Exception:
            logger.exception("Failed to get active companies count")
            return 0

    @staticmethod
    def add_company(name: str, url: str | None = None) -> dict[str, Any]:
        """Add a new company.

        Args:
            name: Company name.
            url: Company URL (optional).

        Returns:
            Dictionary with company data.
        """
        try:
            with db_session() as session:
                company = CompanySQL(
                    name=name,
                    url=url,
                    active=True,
                )
                session.add(company)
                session.commit()
                session.refresh(company)

                return {
                    "id": company.id,
                    "name": company.name,
                    "url": company.url,
                    "active": company.active,
                }
        except Exception:
            logger.exception("Failed to add company: %s", name)
            return {}

    @staticmethod
    def delete_company(company_id: int) -> bool:
        """Delete a company by ID.

        Args:
            company_id: Company ID to delete.

        Returns:
            True if deletion was successful.
        """
        try:
            with db_session() as session:
                company = session.get(CompanySQL, company_id)
                if company:
                    session.delete(company)
                    session.commit()
                    return True
                return False
        except Exception:
            logger.exception("Failed to delete company ID: %s", company_id)
            return False

    @staticmethod
    def toggle_company_active(company_id: int) -> bool | None:
        """Toggle company active status.

        Args:
            company_id: Company ID to toggle.

        Returns:
            New active status, or None if failed.
        """
        try:
            with db_session() as session:
                company = session.get(CompanySQL, company_id)
                if company:
                    company.active = not company.active
                    session.commit()
                    return company.active
                return None
        except Exception:
            logger.exception("Failed to toggle company ID: %s", company_id)
            return None

    @staticmethod
    def bulk_delete_companies(company_ids: list[int]) -> int:
        """Bulk delete companies.

        Args:
            company_ids: List of company IDs to delete.

        Returns:
            Number of companies deleted.
        """
        deleted_count = 0
        for company_id in company_ids:
            if CompanyService.delete_company(company_id):
                deleted_count += 1
        return deleted_count

    @staticmethod
    def bulk_update_status(company_ids: list[int], active: bool) -> int:
        """Bulk update company status.

        Args:
            company_ids: List of company IDs to update.
            active: New active status.

        Returns:
            Number of companies updated.
        """
        updated_count = 0
        try:
            with db_session() as session:
                for company_id in company_ids:
                    company = session.get(CompanySQL, company_id)
                    if company:
                        company.active = active
                        updated_count += 1
                session.commit()
        except Exception:
            logger.exception("Failed to bulk update company status")
        return updated_count

    @staticmethod
    def update_company_active_status(company_id: int, active: bool) -> bool:
        """Update company active status.

        Args:
            company_id: Company ID to update.
            active: New active status.

        Returns:
            True if update was successful.
        """
        try:
            with db_session() as session:
                company = session.get(CompanySQL, company_id)
                if company:
                    company.active = active
                    session.commit()
                    return True
                return False
        except Exception:
            logger.exception("Failed to update company %s status", company_id)
            return False
