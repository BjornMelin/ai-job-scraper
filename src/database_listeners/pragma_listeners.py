"""SQLite pragma event listeners for database optimization.

This module contains event listeners that apply SQLite pragmas
on each new database connection for optimal performance and safety.
"""

import logging

from src.config import Settings

settings = Settings()
logger = logging.getLogger(__name__)


def apply_pragmas(conn, _):
    """Apply SQLite pragmas on each new connection.

    This function is called automatically by SQLAlchemy on each new
    database connection to ensure optimal SQLite configuration.

    Args:
        conn: SQLAlchemy database connection object
        _: Connection record (unused)

    Note:
        Pragmas are applied from the settings.sqlite_pragmas list,
        allowing for flexible configuration of SQLite optimization.
    """
    cursor = conn.cursor()
    for pragma in settings.sqlite_pragmas:
        try:
            cursor.execute(pragma)
            logger.debug("Applied SQLite pragma: %s", pragma)
        except Exception:
            logger.warning("Failed to apply pragma '%s'", pragma)
    cursor.close()
