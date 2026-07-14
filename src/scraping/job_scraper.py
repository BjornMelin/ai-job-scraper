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
        return await asyncio.to_thread(self.scrape_jobs_sync, request)

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
            jobs, invalid_rows = self._dataframe_to_models(jobs_df, request.site_name)
            raw_found = len(jobs_df)
            metadata: dict[str, Any] = {
                "scraping_method": "jobspy",
                "success": bool(jobs),
                "raw_found": raw_found,
                "valid_rows": len(jobs),
                "invalid_rows": invalid_rows,
            }
            if invalid_rows:
                row_label = "row" if raw_found == 1 else "rows"
                message = f"{invalid_rows} of {raw_found} provider {row_label} failed validation"
                metadata["warning" if jobs else "error"] = message

            return JobScrapeResult(
                jobs=jobs,
                total_found=len(jobs),
                request_params=request,
                metadata=metadata,
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
        self, jobs_df: pd.DataFrame, requested_sites: list[JobSite] | JobSite
    ) -> tuple[list[JobPosting], int]:
        """Convert JobSpy DataFrame to list of JobPosting models.

        Args:
            jobs_df: Pandas DataFrame from JobSpy.
            requested_sites: Sites requested for scraping.

        Returns:
            Validated jobs and the number of rejected provider rows.
        """
        jobs = []

        # Determine default site for jobs without explicit site info
        default_site = (
            requested_sites
            if isinstance(requested_sites, JobSite)
            else requested_sites[0]
        )

        for _, row in jobs_df.iterrows():
            try:
                # Convert pandas row to dict with safe value handling
                job_data = {}
                for col, value in row.items():
                    if (pd.api.types.is_scalar(value) and pd.isna(value)) or (
                        isinstance(value, str) and not value.strip()
                    ):
                        job_data[col] = None
                    elif isinstance(value, pd.Timestamp):
                        job_data[col] = value.date() if hasattr(value, "date") else None
                    else:
                        job_data[col] = value

                # Ensure required fields with safe defaults
                if "id" not in job_data or not job_data["id"]:
                    job_data["id"] = (
                        job_data.get("job_url_direct") or job_data.get("job_url") or ""
                    )

                # Set site field if not present in data
                if "site" not in job_data or not job_data["site"]:
                    job_data["site"] = default_site

                # Safe float conversion for salary fields
                job_data["min_amount"] = self._safe_float(job_data.get("min_amount"))
                job_data["max_amount"] = self._safe_float(job_data.get("max_amount"))
                job_data["company_rating"] = self._safe_float(
                    job_data.get("company_rating")
                )

                # Create and validate JobPosting
                job_posting = JobPosting.model_validate(job_data)
                jobs.append(job_posting)

            except (TypeError, ValueError) as error:
                logger.warning("Skipped invalid job row: %s", error)
                continue

        invalid_rows = len(jobs_df) - len(jobs)
        logger.info(
            "Converted %d provider rows; %d failed validation",
            len(jobs),
            invalid_rows,
        )
        return jobs, invalid_rows

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
        metadata = {
            "scraping_method": "jobspy",
            "success": error is None,
            "raw_found": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
        }
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
