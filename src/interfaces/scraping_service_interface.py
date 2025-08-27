"""Interface definition for unified scraping services.

This module defines the IScrapingService protocol that standardizes interactions
with both job board scrapers (JobSpy) and company page scrapers (ScrapeGraphAI).
This interface enables the implementation of the unified 2-tier scraping
architecture specified in Phase 3A.
"""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel

from src.schemas import Job


class SourceType(Enum):
    """Types of job sources supported by the unified scraper."""

    JOB_BOARDS = "job_boards"  # LinkedIn, Indeed, Glassdoor via JobSpy
    COMPANY_PAGES = "company_pages"  # Company career pages via ScrapeGraphAI
    UNIFIED = "unified"  # Both tiers combined


class ScrapingStatus(BaseModel):
    """Status information for scraping operations."""

    task_id: str
    status: str  # "running", "completed", "failed", "queued"
    progress_percentage: float
    jobs_found: int
    jobs_processed: int
    source_type: SourceType
    start_time: datetime
    end_time: datetime | None = None
    error_message: str | None = None
    success_rate: float = 0.0


class JobQuery(BaseModel):
    """Query parameters for unified job scraping."""

    keywords: list[str]
    locations: list[str] = ["Remote"]
    source_types: list[SourceType] = [SourceType.UNIFIED]
    max_results: int = 100
    hours_old: int = 72
    enable_ai_enhancement: bool = True
    concurrent_requests: int = 10


class IScrapingService(Protocol):
    """Protocol for unified job scraping services.

    This interface defines the contract for implementing the 2-tier scraping
    architecture that combines JobSpy (Tier 1) and ScrapeGraphAI (Tier 2)
    for comprehensive job data extraction.

    Key features:
    - Async operations with 15x performance improvement target
    - 95%+ scraping success rate with proxy integration
    - Comprehensive error handling with tenacity retry logic
    - Real-time status monitoring and progress updates
    - Source type routing (job boards vs company pages)
    """

    @abstractmethod
    async def scrape_unified(self, query: JobQuery) -> list[Job]:
        """Execute unified scraping across multiple job sources.

        Combines Tier 1 (JobSpy) and Tier 2 (ScrapeGraphAI) scraping to provide
        comprehensive job data with AI-powered enhancement.

        Args:
            query: Job search parameters and configuration.

        Returns:
            List of structured Job objects with enhanced data.

        Raises:
            ScrapingServiceError: When scraping operations fail.
        """
        ...

    @abstractmethod
    async def scrape_job_boards_async(self, query: JobQuery) -> list[Job]:
        """Scrape job boards using JobSpy (Tier 1).

        High-performance async scraping of structured job boards like
        LinkedIn, Indeed, and Glassdoor.

        Args:
            query: Job search parameters.

        Returns:
            List of jobs from job board sources.
        """
        ...

    @abstractmethod
    async def scrape_company_pages_async(self, query: JobQuery) -> list[Job]:
        """Scrape company career pages using ScrapeGraphAI (Tier 2).

        AI-powered extraction of job listings from company career pages
        with intelligent content parsing.

        Args:
            query: Job search parameters.

        Returns:
            List of jobs from company page sources.
        """
        ...

    @abstractmethod
    async def enhance_job_data(self, jobs: list[Job]) -> list[Job]:
        """Enhance job data using AI-powered analysis.

        Apply ScrapeGraphAI's AI capabilities to enrich job data with
        additional insights, better descriptions, and structured information.

        Args:
            jobs: List of jobs to enhance.

        Returns:
            List of enhanced Job objects.
        """
        ...

    @abstractmethod
    async def start_background_scraping(self, query: JobQuery) -> str:
        """Start background scraping operation.

        Initiates asynchronous scraping task for long-running operations
        with progress monitoring capabilities.

        Args:
            query: Job search parameters.

        Returns:
            Task ID for monitoring progress.
        """
        ...

    @abstractmethod
    async def get_scraping_status(self, task_id: str) -> ScrapingStatus:
        """Get status of background scraping operation.

        Args:
            task_id: ID of the scraping task.

        Returns:
            Current status information.
        """
        ...

    @abstractmethod
    async def monitor_scraping_progress(
        self, task_id: str
    ) -> AsyncGenerator[ScrapingStatus, None]:
        """Monitor scraping progress with real-time updates.

        Yields status updates as scraping progresses, enabling
        real-time UI updates and progress tracking.

        Args:
            task_id: ID of the scraping task.

        Yields:
            ScrapingStatus updates as operation progresses.
        """
        ...

    @abstractmethod
    async def get_success_rate_metrics(self) -> dict[str, Any]:
        """Get scraping success rate and performance metrics.

        Returns:
            Dictionary with success rates, performance metrics,
            and health information.
        """
        ...


class ScrapingServiceError(Exception):
    """Base exception for scraping service errors."""

    def __init__(self, message: str, source_type: SourceType | None = None):
        super().__init__(message)
        self.source_type = source_type


class JobBoardScrapingError(ScrapingServiceError):
    """Error during job board scraping (Tier 1)."""

    def __init__(self, message: str):
        super().__init__(message, SourceType.JOB_BOARDS)


class CompanyPageScrapingError(ScrapingServiceError):
    """Error during company page scraping (Tier 2)."""

    def __init__(self, message: str):
        super().__init__(message, SourceType.COMPANY_PAGES)


class AIEnhancementError(ScrapingServiceError):
    """Error during AI-powered job data enhancement."""

    def __init__(self, message: str):
        super().__init__(message, SourceType.UNIFIED)
