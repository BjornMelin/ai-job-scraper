"""Clean JobSpy wrapper service with Pydantic integration.

This module provides a library-first approach to job scraping using JobSpy
with direct integration to our Pydantic models. Replaces the complex unified_scraper
with a minimal, maintainable implementation focused on core functionality.
"""

import asyncio
import logging

from typing import Any

import pandas as pd

from jobspy import scrape_jobs

from src.models.job_models import (
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
)

logger = logging.getLogger(__name__)


class JobSpyScraper:
    """Clean JobSpy wrapper with Pydantic model integration.

    Provides async/sync methods for job scraping with professional
    error handling and automatic DataFrame to Pydantic conversion.
    """

    def __init__(self) -> None:
        """Initialize scraper with optimal default settings."""
        self.default_settings = {
            "results_wanted": 100,
            "country_indeed": "USA",
            "linkedin_fetch_description": True,
            "linkedin_company_fetch_description": True,
            "description_format": "markdown",
        }

    async def scrape_jobs_async(self, request: JobScrapeRequest) -> JobScrapeResult:
        """Scrape jobs asynchronously using JobSpy with Pydantic integration.

        Args:
            request: Pydantic model with scraping parameters.

        Returns:
            JobScrapeResult with structured job data or empty result on failure.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.scrape_jobs_sync, request)

    def scrape_jobs_sync(self, request: JobScrapeRequest) -> JobScrapeResult:
        """Scrape jobs synchronously using JobSpy with Pydantic integration.

        Args:
            request: Pydantic model with scraping parameters.

        Returns:
            JobScrapeResult with structured job data or empty result on failure.
        """
        try:
            # Convert Pydantic request to JobSpy parameters
            scrape_params = self._build_scrape_params(request)

            logger.info("Starting JobSpy scraping with params: %s", scrape_params)

            # Execute JobSpy scraping
            jobs_df = scrape_jobs(**scrape_params)

            if jobs_df is None or jobs_df.empty:
                logger.warning("JobSpy returned empty or None DataFrame")
                return self._empty_result(request)

            logger.info("JobSpy found %d jobs", len(jobs_df))

            # Convert DataFrame to Pydantic models
            jobs = self._dataframe_to_models(jobs_df, request.site_name)

            return JobScrapeResult(
                jobs=jobs,
                total_found=len(jobs),
                request_params=request,
                metadata={"scraping_method": "jobspy", "success": True},
            )

        except Exception:
            logger.exception("JobSpy scraping failed")
            return self._empty_result(request, error="Scraping operation failed")

    def _build_scrape_params(self, request: JobScrapeRequest) -> dict[str, Any]:
        """Build JobSpy parameters from Pydantic request model."""
        params = self.default_settings.copy()

        # Map site_name to JobSpy format
        if isinstance(request.site_name, list):
            params["site_name"] = [site.value for site in request.site_name]
        else:
            params["site_name"] = [request.site_name.value]

        # Map core parameters
        params.update(
            {
                "search_term": request.search_term,
                "google_search_term": request.google_search_term,
                "location": request.location,
                "distance": request.distance,
                "is_remote": request.is_remote,
                "results_wanted": request.results_wanted,
                "country_indeed": request.country_indeed,
                "offset": request.offset,
                "hours_old": request.hours_old,
                "enforce_annual_salary": request.enforce_annual_salary,
                "linkedin_fetch_description": request.linkedin_fetch_description,
                "description_format": request.description_format,
            }
        )

        # Add job_type and easy_apply if provided
        if request.job_type:
            params["job_type"] = request.job_type.value
        if request.easy_apply is not None:
            params["easy_apply"] = request.easy_apply

        # Filter None values
        return {k: v for k, v in params.items() if v is not None}

    def _dataframe_to_models(
        self, jobs_df: pd.DataFrame, _requested_sites: list[JobSite] | JobSite
    ) -> list[JobPosting]:
        """Convert JobSpy DataFrame to list of JobPosting models.

        Args:
            jobs_df: Pandas DataFrame from JobSpy.
            _requested_sites: Sites requested for scraping (unused, kept for future).

        Returns:
            List of validated JobPosting models.
        """
        jobs = []

        for _, row in jobs_df.iterrows():
            try:
                # Convert pandas row to dict with safe value handling
                job_data = {}
                for col, value in row.items():
                    if pd.isna(value) or (isinstance(value, str) and not value.strip()):
                        job_data[col] = None
                    elif isinstance(value, (pd.Timestamp, pd.DatetimeIndex)):
                        job_data[col] = value.date() if hasattr(value, "date") else None
                    else:
                        job_data[col] = value

                # Ensure required fields with safe defaults
                if "id" not in job_data or not job_data["id"]:
                    job_data["id"] = f"job_{len(jobs)}_{hash(str(job_data))}"

                # Safe float conversion for salary fields
                job_data["min_amount"] = self._safe_float(job_data.get("min_amount"))
                job_data["max_amount"] = self._safe_float(job_data.get("max_amount"))
                job_data["company_rating"] = self._safe_float(
                    job_data.get("company_rating")
                )

                # Create and validate JobPosting
                job_posting = JobPosting.model_validate(job_data)
                jobs.append(job_posting)

            except Exception:
                logger.warning("Failed to convert job row to model")
                continue

        logger.info("Successfully converted %d jobs to Pydantic models", len(jobs))
        return jobs

    def _safe_float(self, value: Any) -> float | None:
        """Safely convert value to float, returning None on failure."""
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def _empty_result(
        self, request: JobScrapeRequest, error: str | None = None
    ) -> JobScrapeResult:
        """Create empty JobScrapeResult for error cases."""
        metadata = {"scraping_method": "jobspy", "success": False}
        if error:
            metadata["error"] = error

        return JobScrapeResult(
            jobs=[],
            total_found=0,
            request_params=request,
            metadata=metadata,
        )


# Global instance for easy import and usage
job_scraper = JobSpyScraper()


# Backward compatibility function
async def scrape_jobs_by_query(
    query: str,
    location: str | None = None,
    sites: list[str] | None = None,
    count: int = 100,
) -> list[dict[str, Any]]:
    """Backward compatibility function matching legacy interface.

    Args:
        query: Search terms for jobs.
        location: Location to search in.
        sites: List of site names to search.
        count: Number of results to return.

    Returns:
        List of job dictionaries in legacy format.
    """
    try:
        # Convert legacy parameters to new Pydantic model
        site_enums = []
        if sites:
            for site in sites:
                site_enum = JobSite.normalize(site)
                if site_enum:
                    site_enums.append(site_enum)

        if not site_enums:
            site_enums = [JobSite.LINKEDIN]  # Default fallback

        request = JobScrapeRequest(
            site_name=site_enums,
            search_term=query,
            location=location,
            results_wanted=count,
        )

        result = await job_scraper.scrape_jobs_async(request)

        # Convert JobPosting models back to dictionaries for backward compatibility
        return [job.model_dump() for job in result.jobs]

    except Exception:
        logger.exception("Legacy scrape_jobs_by_query failed")
        return []
