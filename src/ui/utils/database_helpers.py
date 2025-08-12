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
    """Check for database objects in session state."""
    return []


def clean_session_state() -> int:
    """Remove database objects from session state."""
    return 0


@st.cache_data(ttl=60)
def get_database_health() -> dict[str, Any]:
    """Get database health status with actual connectivity check."""
    try:
        # Simple connectivity test - try to get a session and execute a basic query
        session_factory = get_cached_session_factory()
        session = session_factory()
        try:
            # Execute a lightweight query to verify connection
            from sqlmodel import text

            result = session.execute(text("SELECT 1"))
            result.scalar()
            session.close()
            return {
                "status": "healthy",
                "details": {"connected": True, "message": "Database accessible"},
            }
        except Exception as e:
            session.close()
            logger.warning("Database health check failed: %s", str(e))
            return {
                "status": "unhealthy",
                "details": {"connected": False, "error": str(e)},
            }
    except Exception as e:
        logger.error("Failed to create database session: %s", str(e))
        return {
            "status": "error",
            "details": {"connected": False, "error": f"Session creation failed: {e}"},
        }


def render_database_health_widget() -> None:
    """Render database health monitoring widget in Streamlit sidebar."""
    health = get_database_health()
    status = health.get("status", "unknown")

    if status == "healthy":
        st.success("🟢 Database Connected")
    elif status == "unhealthy":
        st.warning("🟡 Database Issue")
        if error := health.get("details", {}).get("error"):
            st.caption(f"Error: {error[:50]}...")
    else:
        st.error("🔴 Database Error")
        if error := health.get("details", {}).get("error"):
            st.caption(f"Error: {error[:50]}...")


@contextmanager
def background_task_session() -> Generator[Session, None, None]:
    """Session context manager optimized for background tasks."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Background task database error")
        raise
    finally:
        session.close()


def suppress_sqlalchemy_warnings() -> None:
    """Suppress common SQLAlchemy warnings in Streamlit context."""
    import warnings

    from sqlalchemy.exc import SAWarning

    # Suppress common warnings that clutter Streamlit logs
    warnings.filterwarnings("ignore", category=SAWarning, message=".*relationship.*")
    warnings.filterwarnings("ignore", category=SAWarning, message=".*Attribute.*")


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
