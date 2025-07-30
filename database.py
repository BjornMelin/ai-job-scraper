"""Database configuration and session management for the AI Job Scraper application.

This module provides the database engine, session factory, and database initialization
functionality used throughout the application. It follows SQLAlchemy best practices
by centralizing database connection management.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config import Settings
from models import Base

# Initialize settings
settings = Settings()

# Create engine - single instance for the entire application
engine = create_engine(settings.db_url)

# Create all tables
Base.metadata.create_all(engine)

# Create session factory - bound to the engine
SessionLocal = sessionmaker(bind=engine)
