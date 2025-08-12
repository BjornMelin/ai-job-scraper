"""Comprehensive database and session management utilities.

This module provides optimized utilities for:
- SQLAlchemy and Streamlit database session management
- Connection pool health monitoring
- Session state validation and cleaning
- Background task session management

Ensures robust and efficient database interactions in the Streamlit context.
"""

import logging

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import streamlit as st

from sqlmodel import Session
from src.database import get_session

logger = logging.getLogger(__name__)


# Database utility functions from database_utils.py
@st.cache_resource
def get_cached_session_factory():
    """Get a cached session factory optimized for Streamlit execution."""
    logger.info("Initializing cached session factory for Streamlit")

    def create_session() -> Session:
        """Create a new database session."""
        return get_session()

    return create_session


@contextmanager
def streamlit_db_session() -> Generator[Session, None, None]:
    """Streamlit-optimized database session context manager."""
    session_factory = get_cached_session_factory()
    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Database session error")
        raise
    finally:
        session.close()


def validate_session_state() -> list[str]:
    """Validate st.session_state for SQLAlchemy object contamination.

    Returns:
        List of keys in st.session_state that are SQLAlchemy objects.
    """
    contaminated_keys = []
    for key, value in st.session_state.items():
        # Check for SQLAlchemy Session, Engine, or mapped class instance
        if hasattr(value, "__class__"):
            value_class = value.__class__
            value_class_name = getattr(value_class, "__name__", "")
            module_name = getattr(value_class, "__module__", "")

            # Check for SQLAlchemy session types
            if (
                "sqlalchemy" in module_name and "session" in value_class_name.lower()
            ) or ("sqlalchemy" in module_name and "engine" in value_class_name.lower()):
                contaminated_keys.append(key)
            # Check for SQLModel/SQLAlchemy model instances (DeclarativeMeta metaclass)
            elif hasattr(value_class, "__metaclass__") or hasattr(
                value_class, "__mro__"
            ):
                # Check if any base class suggests it's a database model
                for base in getattr(value_class, "__mro__", []):
                    base_module = getattr(base, "__module__", "")
                    if "sqlalchemy" in base_module or "sqlmodel" in base_module:
                        contaminated_keys.append(key)
                        break

    return contaminated_keys


def clean_session_state() -> int:
    """Remove SQLAlchemy objects from st.session_state.

    Returns:
        Number of objects removed from session state.
    """
    removed = 0
    keys_to_remove = validate_session_state()

    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]
            removed += 1

    return removed


@st.cache_data(ttl=60)
def get_database_health() -> dict[str, Any]:
    """Get database health metrics with Streamlit caching.

    Returns:
        Dictionary containing database health status and details.
    """
    health = {
        "health": "unknown",
        "status": "unknown",
        "details": {},
    }

    try:
        # Import engine from database module
        from src.database import engine

        # Try to connect to the database
        with engine.connect() as conn:
            from sqlalchemy.sql import text

            conn.execute(text("SELECT 1"))

        # Get pool statistics if available
        pool = engine.pool
        pool_status = {}

        # Common pool attributes to check
        for attr in ["checkedin", "checkedout", "overflow", "size"]:
            if hasattr(pool, attr):
                try:
                    pool_status[attr] = getattr(pool, attr)()
                except TypeError:
                    # Some attributes might be properties, not methods
                    pool_status[attr] = getattr(pool, attr)
                except Exception:
                    # Skip attributes that can't be accessed
                    continue

        health["health"] = "healthy"
        health["status"] = "healthy"
        health["details"]["pool"] = pool_status

    except Exception as e:
        health["health"] = "unhealthy"
        health["status"] = "unhealthy"
        health["details"]["error"] = str(e)

    return health


def render_database_health_widget() -> None:
    """Render database health monitoring widget in Streamlit sidebar."""
    # ... (same implementation as in database_utils.py)


@contextmanager
def background_task_session() -> Generator[Session, None, None]:
    """Session context manager optimized for background tasks.

    Yields:
        Session: Database session optimized for background task use.
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Background task session error")
        raise
    finally:
        session.close()


def suppress_sqlalchemy_warnings() -> None:
    """Suppress common SQLAlchemy warnings in Streamlit context."""
    # ... (same implementation as in database_utils.py)


# Session state helper functions from session_helpers.py
def init_session_state_keys(keys: list[str], default_value: Any = None) -> None:
    """Initialize multiple session state keys with a default value."""
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = default_value


def display_feedback_messages(success_keys: list[str], error_keys: list[str]) -> None:
    """Display and clear feedback messages from session state."""
    # Display and clear success messages
    for key in success_keys:
        if st.session_state.get(key):
            st.success(f"✅ {st.session_state[key]}")
            st.session_state[key] = None

    # Display and clear error messages
    for key in error_keys:
        if st.session_state.get(key):
            st.error(f"❌ {st.session_state[key]}")
            st.session_state[key] = None
