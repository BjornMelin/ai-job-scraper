"""Streamlit-optimized database utilities for the AI Job Scraper.

This module provides library-first database utilities specifically designed for
Streamlit's execution model, ensuring optimal session management, preventing
state contamination, and integrating performance monitoring.
"""

import logging
import warnings

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import streamlit as st

from sqlalchemy.orm.base import instance_state
from sqlmodel import Session

from src.database import get_connection_pool_status, get_session

logger = logging.getLogger(__name__)


@st.cache_resource
def get_cached_session_factory():
    """Get a cached session factory optimized for Streamlit execution.

    Uses Streamlit's caching to ensure efficient session management across
    page reloads and navigation. The factory itself is cached, not sessions.

    Returns:
        Function that creates new database sessions.
    """
    logger.info("Initializing cached session factory for Streamlit")

    def create_session() -> Session:
        """Create a new database session."""
        return get_session()

    return create_session


@contextmanager
def streamlit_db_session() -> Generator[Session, None, None]:
    """Streamlit-optimized database session context manager.

    Provides automatic session lifecycle management optimized for Streamlit's
    execution model with proper error handling and cleanup.

    Yields:
        Session: SQLModel database session.

    Example:
        ```python
        with streamlit_db_session() as session:
            jobs = session.exec(select(JobSQL)).all()
        ```
    """
    session_factory = get_cached_session_factory()
    session = session_factory()

    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}", exc_info=True)
        raise
    finally:
        session.close()


def validate_session_state() -> list[str]:
    """Validate st.session_state for SQLAlchemy object contamination.

    Checks for detached SQLAlchemy objects in session state that could cause
    LazyLoadingError or DetachedInstanceError. This prevents common issues
    when database objects are accidentally stored in Streamlit session state.

    Returns:
        List of keys containing SQLAlchemy objects (empty if none found).

    Example:
        ```python
        contaminated_keys = validate_session_state()
        if contaminated_keys:
            st.warning(f"Database objects found in session state: {contaminated_keys}")
        ```
    """
    contaminated_keys = []

    for key, value in st.session_state.items():
        # Skip private Streamlit keys
        if key.startswith("_"):
            continue

        # Check if value is a SQLAlchemy model instance
        if hasattr(value, "__table__") and hasattr(value, "__class__"):
            # Check if it's detached (not associated with a session)
            state = instance_state(value)
            if state.detached:
                contaminated_keys.append(key)

        # Check lists/dicts that might contain SQLAlchemy objects
        elif isinstance(value, list | tuple):
            for item in value:
                if hasattr(item, "__table__") and hasattr(item, "__class__"):
                    state = instance_state(item)
                    if state.detached:
                        contaminated_keys.append(f"{key}[item]")
                        break

        elif isinstance(value, dict):
            for dict_key, dict_value in value.items():
                if hasattr(dict_value, "__table__") and hasattr(
                    dict_value, "__class__"
                ):
                    state = instance_state(dict_value)
                    if state.detached:
                        contaminated_keys.append(f"{key}[{dict_key}]")
                        break

    return contaminated_keys


def clean_session_state() -> int:
    """Remove SQLAlchemy objects from st.session_state.

    Automatically cleans detached SQLAlchemy objects from session state
    to prevent LazyLoadingError and DetachedInstanceError issues.

    Returns:
        Number of keys cleaned.

    Example:
        ```python
        cleaned_count = clean_session_state()
        if cleaned_count > 0:
            st.info(f"Cleaned {cleaned_count} database objects from session state")
        ```
    """
    contaminated_keys = validate_session_state()

    for key in contaminated_keys:
        # Handle nested keys (e.g., "key[item]" or "key[dict_key]")
        if "[" in key:
            main_key = key.split("[")[0]
            if main_key in st.session_state:
                logger.warning(f"Removing contaminated session state key: {key}")
                del st.session_state[main_key]
        else:
            if key in st.session_state:
                logger.warning(f"Removing contaminated session state key: {key}")
                del st.session_state[key]

    return len(contaminated_keys)


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_database_health() -> dict[str, Any]:
    """Get database health metrics with Streamlit caching.

    Returns connection pool status and basic health information,
    cached for 1 minute to avoid excessive polling.

    Returns:
        Dictionary with database health metrics.

    Example:
        ```python
        health = get_database_health()
        st.metric("Active Connections", health["checked_out"])
        ```
    """
    try:
        pool_status = get_connection_pool_status()

        # Add health assessment
        pool_status["health"] = "healthy"
        if isinstance(pool_status["checked_out"], int):
            if pool_status["checked_out"] > pool_status.get("pool_size", 10) * 0.8:
                pool_status["health"] = "warning"
            if pool_status["overflow"] > 0:
                pool_status["health"] = "critical"

        return pool_status

    except Exception as e:
        logger.error(f"Failed to get database health: {e}")
        return {
            "health": "error",
            "error": str(e),
            "pool_size": "unknown",
            "checked_out": "unknown",
            "overflow": "unknown",
            "invalid": "unknown",
        }


def render_database_health_widget() -> None:
    """Render database health monitoring widget in Streamlit sidebar.

    Displays connection pool status, health metrics, and session state
    validation results in a collapsible sidebar section.

    Example:
        ```python
        # In main.py or sidebar
        render_database_health_widget()
        ```
    """
    with st.sidebar.expander("🗃️ Database Health", expanded=False):
        health = get_database_health()

        # Health status indicator
        health_status = health.get("health", "unknown")
        health_colors = {
            "healthy": "🟢",
            "warning": "🟡",
            "critical": "🔴",
            "error": "⚫",
        }

        status_text = (
            f"**Status:** {health_colors.get(health_status, '❓')} "
            f"{health_status.title()}"
        )
        st.write(status_text)

        # Connection pool metrics
        if "error" not in health:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pool Size", health.get("pool_size", "N/A"))
                st.metric("Active", health.get("checked_out", "N/A"))
            with col2:
                st.metric("Overflow", health.get("overflow", "N/A"))
                st.metric("Invalid", health.get("invalid", "N/A"))
        else:
            st.error(f"Database Error: {health['error']}")

        # Session state validation
        contaminated_keys = validate_session_state()
        if contaminated_keys:
            st.warning(f"⚠️ {len(contaminated_keys)} contaminated session keys")
            if st.button("🧹 Clean Session State"):
                cleaned = clean_session_state()
                st.success(f"Cleaned {cleaned} keys")
                st.rerun()
        else:
            st.success("✅ Session state clean")


# Background task session management enhancement
@contextmanager
def background_task_session() -> Generator[Session, None, None]:
    """Session context manager optimized for background tasks.

    Provides explicit session management for background threads with
    enhanced error handling and cleanup specifically designed for the
    simplified background task system.

    Yields:
        Session: SQLModel database session for background tasks.

    Example:
        ```python
        def background_scraping_task():
            with background_task_session() as session:
                # Perform database operations
                companies = session.exec(select(CompanySQL)).all()
        ```
    """
    session = get_session()

    try:
        logger.debug("Starting background task database session")
        yield session
        session.commit()
        logger.debug("Background task database session committed successfully")
    except Exception as e:
        session.rollback()
        logger.error(f"Background task database session error: {e}", exc_info=True)
        raise
    finally:
        session.close()
        logger.debug("Background task database session closed")


def suppress_sqlalchemy_warnings() -> None:
    """Suppress common SQLAlchemy warnings in Streamlit context.

    Filters out warnings that are common in Streamlit's execution model
    but don't indicate actual problems, such as DetachedInstanceWarning
    when objects are accessed after session closure.
    """
    warnings.filterwarnings(
        "ignore",
        message=".*DetachedInstanceWarning.*",
        category=UserWarning,
        module="sqlalchemy.*",
    )

    warnings.filterwarnings(
        "ignore",
        message=".*lazy loading.*",
        category=UserWarning,
        module="sqlalchemy.*",
    )
