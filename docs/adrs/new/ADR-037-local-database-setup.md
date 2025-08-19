# ADR-037: Local Database Setup

## Title

SQLModel with SQLite for Local Development

## Version/Date

2.0 / August 19, 2025

## Status

**Accepted** - Focused on local development simplicity

## Description

Simple database setup using SQLModel with SQLite for local development. Provides basic job and company models with straightforward relationships, leveraging SQLModel's native capabilities without production complexity.

## Context

### Local Development Focus

This database architecture is designed for local development workflows:

1. **SQLite File-Based**: Simple file-based database requiring no server setup
2. **SQLModel Native**: Leverage built-in capabilities without custom sync engines
3. **Development Patterns**: Simple models focused on core functionality
4. **Data Persistence**: Basic persistence between development sessions
5. **Easy Debugging**: Direct SQLite file access for inspection

### Framework Integration

- **ORM**: SQLModel (combines SQLAlchemy + Pydantic)
- **Database**: SQLite with WAL mode for development
- **Validation**: Pydantic v2 for automatic validation
- **Relationships**: Simple foreign key relationships
- **Migration**: Basic table creation (no complex migrations needed)

## Decision

**Use Simple SQLModel + SQLite Setup** for local development:

### Database Models

```python
# src/models/database.py
from sqlmodel import SQLModel, Field, Relationship, create_engine, Session
from typing import Optional, List
from datetime import datetime
from pathlib import Path

# Simple SQLite database for development
DATABASE_URL = "sqlite:///./data/jobs.db"

engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL debugging in development
    connect_args={"check_same_thread": False}
)

def init_database():
    """Initialize database with simple table creation."""
    # Ensure data directory exists
    Path("./data").mkdir(exist_ok=True)
    
    # Create all tables
    SQLModel.metadata.create_all(engine)
    
    # Enable WAL mode for better concurrency in development
    with Session(engine) as session:
        session.exec("PRAGMA journal_mode=WAL")
        session.exec("PRAGMA synchronous=NORMAL")
        session.commit()

class JobModel(SQLModel, table=True):
    """Simple job model for local development."""
    __tablename__ = "jobs"
    
    # Basic fields
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    company: str
    location: Optional[str] = None
    description: Optional[str] = None
    
    # Simple salary fields
    salary_text: Optional[str] = None
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    
    # URLs and dates
    url: str = Field(unique=True)
    posted_date: Optional[str] = None
    scraped_at: datetime = Field(default_factory=datetime.now)
    
    # Simple flags
    is_active: bool = Field(default=True)
    is_favorited: bool = Field(default=False)
    
    # Relationship to company
    company_id: Optional[int] = Field(default=None, foreign_key="companies.id")
    company_info: Optional["CompanyModel"] = Relationship(back_populates="jobs")

class CompanyModel(SQLModel, table=True):
    """Simple company model for local development."""
    __tablename__ = "companies"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(unique=True)
    domain: Optional[str] = None
    description: Optional[str] = None
    size: Optional[str] = None
    industry: Optional[str] = None
    
    # Relationship to jobs
    jobs: List[JobModel] = Relationship(back_populates="company_info")

# Database session helper
def get_session():
    """Get database session for operations."""
    with Session(engine) as session:
        yield session
```

### Simple Database Operations

```python
# src/services/database_service.py
from sqlmodel import Session, select
from src.models.database import engine, JobModel, CompanyModel
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class DatabaseService:
    """Simple database operations for local development."""
    
    def create_job(self, job_data: dict) -> JobModel:
        """Create a new job record."""
        with Session(engine) as session:
            # Check if job already exists
            existing = session.exec(
                select(JobModel).where(JobModel.url == job_data["url"])
            ).first()
            
            if existing:
                return self.update_job(existing.id, job_data)
            
            # Create new job
            job = JobModel(**job_data)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
    
    def update_job(self, job_id: int, job_data: dict) -> Optional[JobModel]:
        """Update existing job record."""
        with Session(engine) as session:
            job = session.get(JobModel, job_id)
            if not job:
                return None
            
            # Update fields
            for key, value in job_data.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            session.add(job)
            session.commit()
            session.refresh(job)
            return job
    
    def get_jobs(self, limit: int = 100, offset: int = 0) -> List[JobModel]:
        """Get jobs with simple pagination."""
        with Session(engine) as session:
            statement = select(JobModel).where(
                JobModel.is_active == True
            ).offset(offset).limit(limit)
            
            return list(session.exec(statement))
    
    def get_job_by_id(self, job_id: int) -> Optional[JobModel]:
        """Get job by ID."""
        with Session(engine) as session:
            return session.get(JobModel, job_id)
    
    def search_jobs(self, query: str) -> List[JobModel]:
        """Simple text search in jobs."""
        with Session(engine) as session:
            statement = select(JobModel).where(
                JobModel.title.contains(query) |
                JobModel.company.contains(query) |
                JobModel.description.contains(query)
            )
            return list(session.exec(statement))
    
    def create_company(self, company_data: dict) -> CompanyModel:
        """Create or get existing company."""
        with Session(engine) as session:
            # Check if company exists
            existing = session.exec(
                select(CompanyModel).where(CompanyModel.name == company_data["name"])
            ).first()
            
            if existing:
                return existing
            
            # Create new company
            company = CompanyModel(**company_data)
            session.add(company)
            session.commit()
            session.refresh(company)
            return company
    
    def get_companies(self) -> List[CompanyModel]:
        """Get all companies."""
        with Session(engine) as session:
            return list(session.exec(select(CompanyModel)))
    
    def get_stats(self) -> dict:
        """Get simple database statistics."""
        with Session(engine) as session:
            total_jobs = len(list(session.exec(select(JobModel))))
            active_jobs = len(list(session.exec(
                select(JobModel).where(JobModel.is_active == True)
            )))
            total_companies = len(list(session.exec(select(CompanyModel))))
            
            return {
                "total_jobs": total_jobs,
                "active_jobs": active_jobs,
                "total_companies": total_companies
            }

# Global database service instance
db_service = DatabaseService()
```

### Integration with Reflex

```python
# src/state/database_state.py
import reflex as rx
from src.services.database_service import db_service
from src.models.database import JobModel, CompanyModel
from typing import List

class DatabaseState(rx.State):
    """Database state for Reflex UI."""
    
    # Simple state variables
    jobs: List[dict] = []
    companies: List[dict] = []
    current_job: dict = {}
    search_query: str = ""
    
    # Stats
    total_jobs: int = 0
    active_jobs: int = 0
    total_companies: int = 0
    
    def load_jobs(self):
        """Load jobs for display."""
        jobs = db_service.get_jobs(limit=50)
        self.jobs = [
            {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "salary_text": job.salary_text,
                "scraped_at": job.scraped_at.strftime("%Y-%m-%d"),
                "is_favorited": job.is_favorited
            }
            for job in jobs
        ]
    
    def search_jobs(self):
        """Search jobs by query."""
        if not self.search_query.strip():
            self.load_jobs()
            return
        
        jobs = db_service.search_jobs(self.search_query)
        self.jobs = [
            {
                "id": job.id,
                "title": job.title,
                "company": job.company,
                "location": job.location,
                "salary_text": job.salary_text,
                "scraped_at": job.scraped_at.strftime("%Y-%m-%d"),
                "is_favorited": job.is_favorited
            }
            for job in jobs
        ]
    
    def toggle_favorite(self, job_id: int):
        """Toggle job favorite status."""
        job = db_service.get_job_by_id(job_id)
        if job:
            db_service.update_job(job_id, {"is_favorited": not job.is_favorited})
            self.load_jobs()  # Refresh list
    
    def load_companies(self):
        """Load companies for display."""
        companies = db_service.get_companies()
        self.companies = [
            {
                "id": company.id,
                "name": company.name,
                "domain": company.domain,
                "industry": company.industry,
                "job_count": len(company.jobs) if company.jobs else 0
            }
            for company in companies
        ]
    
    def load_stats(self):
        """Load database statistics."""
        stats = db_service.get_stats()
        self.total_jobs = stats["total_jobs"]
        self.active_jobs = stats["active_jobs"]
        self.total_companies = stats["total_companies"]
    
    def on_load(self):
        """Load initial data when page loads."""
        self.load_jobs()
        self.load_companies()
        self.load_stats()
```

### Development Utilities

```python
# src/utils/db_utils.py
from src.models.database import init_database, engine
from src.services.database_service import db_service
import sqlite3
from pathlib import Path

def reset_database():
    """Reset database for development (careful - destroys data!)."""
    db_path = Path("./data/jobs.db")
    if db_path.exists():
        db_path.unlink()
    
    init_database()
    print("Database reset complete.")

def seed_sample_data():
    """Add sample data for development."""
    sample_jobs = [
        {
            "title": "Python Developer",
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "description": "Looking for a Python developer...",
            "url": "https://example.com/job1",
            "salary_text": "$80,000 - $120,000"
        },
        {
            "title": "Data Scientist",
            "company": "Data Inc",
            "location": "New York, NY",
            "description": "Data science position...",
            "url": "https://example.com/job2",
            "salary_text": "$100,000 - $150,000"
        }
    ]
    
    for job_data in sample_jobs:
        db_service.create_job(job_data)
    
    print(f"Added {len(sample_jobs)} sample jobs.")

def inspect_database():
    """Simple database inspection for development."""
    stats = db_service.get_stats()
    print(f"Database Statistics:")
    print(f"  Total Jobs: {stats['total_jobs']}")
    print(f"  Active Jobs: {stats['active_jobs']}")
    print(f"  Companies: {stats['total_companies']}")
    
    # Show recent jobs
    recent_jobs = db_service.get_jobs(limit=5)
    print(f"\nRecent Jobs:")
    for job in recent_jobs:
        print(f"  {job.title} at {job.company}")

if __name__ == "__main__":
    # Development commands
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "reset":
            reset_database()
        elif command == "seed":
            seed_sample_data()
        elif command == "inspect":
            inspect_database()
        else:
            print("Available commands: reset, seed, inspect")
    else:
        print("Usage: python -m src.utils.db_utils [reset|seed|inspect]")
```

## Consequences

### Positive Outcomes

- **Simple Setup**: Single SQLite file, no server required
- **Local Development**: Perfect for development and testing
- **SQLModel Benefits**: Type safety and validation out of the box
- **Direct Access**: Can inspect database with SQLite browser tools
- **Fast Iteration**: Quick schema changes during development
- **Portable**: Database file can be easily moved or backed up

### Negative Consequences

- **Development Only**: Not suitable for production multi-user scenarios
- **Limited Concurrency**: SQLite limitations for high-concurrency use
- **No Advanced Features**: Missing production database features
- **File-Based**: Potential for file corruption if not handled properly

### Risk Mitigation

- **WAL Mode**: Enables better concurrency for development
- **Regular Backups**: Simple file copy for backup
- **Clear Documentation**: Guidelines for development usage
- **Migration Path**: Clear upgrade path to production database

## Development Guidelines

### Database File Management

- Database stored in `./data/jobs.db`
- Use WAL mode for better development experience
- Regular backups during development: `cp ./data/jobs.db ./data/backup-$(date +%Y%m%d).db`

### Schema Changes

- For development: Simply delete database file and restart
- Use `python -m src.utils.db_utils reset` command
- Add sample data with `python -m src.utils.db_utils seed`

### Debugging

- Set `echo=True` in engine creation for SQL logging
- Use SQLite browser for direct database inspection
- Use `python -m src.utils.db_utils inspect` for quick stats

## Related ADRs

- **Supports ADR-035**: Local Development Architecture (database component)
- **Replaces Archived ADR-037**: Integrated Database Architecture (production-focused)
- **Supports ADR-040**: UI Component Architecture (data integration)

## Success Criteria

- [ ] Database initializes correctly on first run
- [ ] Basic CRUD operations work for jobs and companies
- [ ] Simple search functionality available
- [ ] Database persists between application restarts
- [ ] Development utilities work for reset/seed/inspect
- [ ] Integration with Reflex state management successful

---

*This ADR provides a simple, practical database setup for local development without production database complexity.*
