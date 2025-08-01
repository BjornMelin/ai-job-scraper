"""Database connection and session management for the AI Job Scraper."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel.sql.expression import Select, SelectOfScalar

from .config import Settings

settings = Settings()

# Convert regular SQLite URL to async if needed
db_url = settings.db_url
if db_url.startswith("sqlite:///"):
    db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///")
elif db_url == "sqlite:///:memory:":
    db_url = "sqlite+aiosqlite:///:memory:"

engine: AsyncEngine = create_async_engine(db_url, echo=False, future=True)

# Create async session factory
async_session_factory = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False
)

# Patch SQLModel to support async
Select.inherit_cache = True  # type: ignore[attr-defined]
SelectOfScalar.inherit_cache = True  # type: ignore[attr-defined]


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Async context manager for database sessions.

    Yields:
        AsyncSession: An asynchronous database session.
    """
    async with async_session_factory() as session:
        yield session
