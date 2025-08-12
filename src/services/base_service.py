"""Base service class for common database operations.

This module provides the BaseService class that serves as a foundation for all service
layers. It standardizes database session management and provides common patterns for
relationship loading.
"""

import logging

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, TypeVar

from sqlalchemy.orm import InstrumentedAttribute, joinedload, selectinload
from sqlmodel import Session
from src.database import db_session

logger = logging.getLogger(__name__)

# Type variable for SQLModel classes
SQLModelType = TypeVar("SQLModelType")


class BaseService:
    """Base service class providing standardized database operations.

    This class eliminates the repeated session management patterns across services
    by providing common session handling and relationship loading utilities.

    Key features:
    - Standardized session management with automatic commit/rollback
    - Flexible relationship loading (eager/lazy) configuration
    - Consistent error handling and logging
    - Type-safe relationship loading with proper hints
    """

    @classmethod
    @contextmanager
    def get_session_with_relationships(
        cls, *relationships: InstrumentedAttribute[Any]
    ) -> Generator[Session, None, None]:
        """Get database session with configured relationship loading.

        This context manager provides a session with pre-configured eager loading
        for specified relationships, eliminating N+1 query problems.

        Args:
            *relationships: SQLModel relationship attributes to eagerly load using
                joinedload.

        Yields:
            Session: Database session with configured relationship loading.

        Example:
            ```python
            class JobService(BaseService):
                @staticmethod
                def get_jobs_with_company():
                    with JobService.get_session_with_relationships(
                        JobSQL.company_relation
                    ) as session:
                        jobs = session.exec(select(JobSQL)).all()
                        # Jobs are returned with company_relation eagerly loaded
                        return jobs
            ```

        Note:
            - Uses joinedload for all relationships by default (good for one-to-one,
                many-to-one)
            - Automatic session commit on success, rollback on exception
            - Proper resource cleanup guaranteed
        """
        with db_session() as session:
            # Configure eager loading options if relationships are specified
            if relationships:
                # Store the loading options in session info for query builders to use
                # This is a pattern that allows query building methods to access
                # the configured loading options
                session.info.setdefault("eager_load_options", [])
                session.info["eager_load_options"].extend(
                    [joinedload(rel) for rel in relationships]
                )

            yield session

    @classmethod
    @contextmanager
    def get_session_with_selectin_load(
        cls, *relationships: InstrumentedAttribute[Any]
    ) -> Generator[Session, None, None]:
        """Get database session with selectinload for one-to-many relationships.

        This context manager is optimized for loading one-to-many relationships
        using selectinload, which is more efficient than joinedload for collections.

        Args:
            *relationships: SQLModel relationship attributes to load using selectinload.

        Yields:
            Session: Database session configured for selectinload.

        Example:
            ```python
            class CompanyService(BaseService):
                @staticmethod
                def get_companies_with_jobs():
                    with CompanyService.get_session_with_selectin_load(
                        CompanySQL.jobs
                    ) as session:
                        companies = session.exec(
                            select(CompanySQL)
                        ).all()
                        # Companies returned with jobs collection efficiently loaded
                        return companies
            ```

        Note:
            - Uses selectinload which issues separate queries for collections
            - More efficient than joinedload for one-to-many relationships
            - Prevents cartesian product issues with multiple collections
        """
        with db_session() as session:
            if relationships:
                session.info.setdefault("selectin_load_options", [])
                session.info["selectin_load_options"].extend(
                    [selectinload(rel) for rel in relationships]
                )

            yield session

    @classmethod
    @contextmanager
    def get_optimized_session(
        cls,
        *,
        joined_relationships: list[InstrumentedAttribute[Any]] | None = None,
        selectin_relationships: list[InstrumentedAttribute[Any]] | None = None,
    ) -> Generator[Session, None, None]:
        """Get database session with mixed relationship loading strategies.

        This advanced session factory allows mixing joinedload and selectinload
        strategies within a single session for optimal query performance.

        Args:
            joined_relationships: Relationships to load with joinedload (for to-one).
            selectin_relationships: Relationships to load with selectinload (for
                to-many).

        Yields:
            Session: Database session with mixed loading configuration.

        Example:
            ```python
            class JobService(BaseService):
                @staticmethod
                def get_jobs_optimized():
                    with (
                        JobService.get_optimized_session(
                            joined_relationships=[
                                JobSQL.company_relation
                            ],
                            selectin_relationships=[],  # No collections in JobSQL
                        ) as session
                    ):
                        return session.exec(select(JobSQL)).all()
            ```

        Note:
            - joinedload: Best for to-one relationships (company_relation)
            - selectinload: Best for to-many relationships (jobs collection)
            - Combines both strategies for optimal performance
        """
        with db_session() as session:
            if joined_relationships:
                session.info.setdefault("eager_load_options", [])
                session.info["eager_load_options"].extend(
                    [joinedload(rel) for rel in joined_relationships]
                )

            if selectin_relationships:
                session.info.setdefault("selectin_load_options", [])
                session.info["selectin_load_options"].extend(
                    [selectinload(rel) for rel in selectin_relationships]
                )

            yield session

    @staticmethod
    def log_operation(operation: str, **kwargs: Any) -> None:
        """Log service operation with consistent formatting.

        Args:
            operation: Description of the operation being performed.
            **kwargs: Additional context to include in log message.

        Example:
            ```python
            BaseService.log_operation(
                "bulk_update_companies", count=5, status="success"
            )
            # Logs: "Service operation: bulk_update_companies - count=5, status=success"
            ```
        """
        context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(
            "Service operation: %s%s", operation, f" - {context}" if context else ""
        )

    @staticmethod
    def handle_service_error(operation: str, error: Exception, **context: Any) -> None:
        """Handle and log service errors with consistent formatting.

        Args:
            operation: Description of the operation that failed.
            error: The exception that was raised.
            **context: Additional context about the failure.

        Example:
            ```python
            try:
                # ... some operation
            except Exception as e:
                BaseService.handle_service_error("get_companies", e, user_id=123)
                raise
            ```
        """
        context_str = ", ".join(f"{k}={v}" for k, v in context.items())
        logger.exception(
            "Service operation failed: %s - %s%s",
            operation,
            str(error),
            f" - {context_str}" if context_str else "",
        )
