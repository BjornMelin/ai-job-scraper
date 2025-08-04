"""Database connection and session management for the AI Job Scraper.

This module provides synchronous database connectivity using SQLAlchemy
and SQLModel. It handles database engine creation, session management,
and table creation for the AI Job Scraper application.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import Session, SQLModel

from .config import Settings

settings = Settings()

# Create synchronous SQLAlchemy engine
engine = create_engine(settings.db_url, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_db_and_tables() -> None:
    """Create database tables from SQLModel definitions.

    This function creates all tables defined in the SQLModel metadata.
    It should be called once during application initialization to ensure
    all required database tables exist.
    """
    SQLModel.metadata.create_all(engine)


def get_session() -> Session:
    """Create a new database session.

    Returns:
        Session: A new SQLAlchemy session for database operations.

    Note:
        The caller is responsible for closing the session when done.
        Consider using a context manager or try/finally block.
    """
    return SessionLocal()
