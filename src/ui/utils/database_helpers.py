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
    """Validate st.session_state for SQLAlchemy object contamination."""
    # ... (same implementation as in database_utils.py)


def clean_session_state() -> int:
    """Remove SQLAlchemy objects from st.session_state."""
    # ... (same implementation as in database_utils.py)


@st.cache_data(ttl=60)
def get_database_health() -> dict[str, Any]:
    """Get database health metrics with Streamlit caching."""
    # ... (same implementation as in database_utils.py)


def render_database_health_widget() -> None:
    """Render database health monitoring widget in Streamlit sidebar."""
    # ... (same implementation as in database_utils.py)


@contextmanager
def background_task_session() -> Generator[Session, None, None]:
    """Session context manager optimized for background tasks."""
    # ... (same implementation as in database_utils.py)


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
