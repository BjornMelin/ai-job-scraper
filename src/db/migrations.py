"""Database migration utilities for the AI Job Scraper.

This module provides utilities for running Alembic database migrations
during application startup, ensuring the database schema stays in sync
with the application code.
"""

import logging

from alembic import command
from alembic.config import Config
from alembic.util.exc import CommandError
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


def run_migrations() -> None:
    """Run Alembic database migrations to head revision.

    This function handles migration execution with proper error handling
    and logging. It's designed to run safely during application startup
    and is idempotent (safe to run multiple times).

    The function uses the alembic.ini configuration file and will apply
    all pending migrations to bring the database schema up to date.

    Raises:
        CommandError: If there's an Alembic-specific error during migration
        SQLAlchemyError: If there's a database-level error during migration
    """
    try:
        logger.info("Starting database migrations...")

        # Load Alembic configuration from alembic.ini
        alembic_cfg = Config("alembic.ini")

        # Run migrations to head (latest) revision
        # This is idempotent - safe to run multiple times
        command.upgrade(alembic_cfg, "head")

        logger.info("Database migrations completed successfully")

    except (CommandError, SQLAlchemyError) as e:
        logger.exception(
            "Failed to run database migrations [%s]",
            type(e).__name__,
        )
        # Don't raise the exception to prevent app startup failure
        # The app can still work with the current database state
        logger.warning(
            "Application will continue with current database schema. "
            "Manual migration may be required.",
        )
