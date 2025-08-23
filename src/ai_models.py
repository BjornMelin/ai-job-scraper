"""Pydantic models for AI structured output using Instructor.

This module defines structured output models for various AI tasks:
- Job posting extraction from web content
- Company information extraction
- Job listing analysis and normalization

These models replace custom JSON parsing with validated, typed responses
from LLM completions using the Instructor library.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class JobPosting(BaseModel):
    """Structured model for job posting extraction from web content."""

    title: str = Field(
        description="The job title or position name", min_length=1, max_length=200
    )
    company: str = Field(
        description="Company name offering the position", min_length=1, max_length=100
    )
    location: str = Field(
        description="Job location (city, state, country, or 'Remote')", max_length=100
    )
    salary_text: str | None = Field(
        default=None,
        description=(
            "Raw salary information as text (e.g., '$80,000 - $120,000', 'Competitive')"
        ),
        max_length=100,
    )
    employment_type: str | None = Field(
        default=None,
        description="Type of employment (Full-time, Part-time, Contract, etc.)",
        max_length=50,
    )
    description: str = Field(
        description="Job description and requirements", min_length=50, max_length=5000
    )
    url: str | None = Field(
        default=None, description="Direct URL to the job posting", max_length=500
    )
    posted_date: str | None = Field(
        default=None,
        description="When the job was posted (in ISO format if possible)",
        max_length=50,
    )

    @field_validator("title", "company", "location", "description")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Ensure required string fields are not just whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or just whitespace")
        return v.strip()

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str | None) -> str | None:
        """Validate URL format if provided."""
        if v and not v.startswith(("http://", "https://")):
            # Attempt to fix relative URLs by prepending https://
            return f"https://{v}"
        return v


class CompanyInfo(BaseModel):
    """Structured model for company information extraction."""

    name: str = Field(description="Official company name", min_length=1, max_length=100)
    industry: str | None = Field(
        default=None, description="Primary industry or sector", max_length=100
    )
    size: str | None = Field(
        default=None,
        description="Company size (e.g., '1-50 employees', 'Large enterprise')",
        max_length=50,
    )
    headquarters: str | None = Field(
        default=None, description="Company headquarters location", max_length=100
    )
    description: str | None = Field(
        default=None,
        description="Brief company description or mission",
        max_length=1000,
    )
    website: str | None = Field(
        default=None, description="Main company website URL", max_length=200
    )


class JobListExtraction(BaseModel):
    """Model for extracting multiple job listings from a single page."""

    jobs: list[JobPosting] = Field(
        description="List of job postings found on the page", default_factory=list
    )
    total_found: int = Field(description="Total number of job postings found", ge=0)
    extraction_confidence: float = Field(
        description="Confidence level of the extraction (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=1.0,
    )
    notes: str | None = Field(
        default=None,
        description="Any additional notes about the extraction process",
        max_length=500,
    )

    @field_validator("total_found")
    @classmethod
    def validate_total_matches_jobs(cls, v: int, info) -> int:
        """Ensure total_found matches the actual number of jobs extracted."""
        if info.data and "jobs" in info.data:
            jobs = info.data["jobs"]
            if v != len(jobs):
                # Auto-correct to match actual count
                return len(jobs)
        return v


class ContentAnalysis(BaseModel):
    """Model for analyzing web content to determine if it contains job listings."""

    contains_jobs: bool = Field(description="Whether the page contains job listings")
    job_count_estimate: int = Field(
        description="Estimated number of job listings on the page", ge=0
    )
    page_type: str = Field(
        description=(
            "Type of page (e.g., 'job_board', 'career_page', 'individual_job', 'other')"
        ),
        max_length=50,
    )
    confidence: float = Field(
        description="Confidence in the analysis (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=1.0,
    )
    reasoning: str | None = Field(
        default=None,
        description="Brief explanation of the analysis decision",
        max_length=500,
    )


class SalaryExtraction(BaseModel):
    """Model for extracting structured salary information from text."""

    min_salary: float | None = Field(
        default=None,
        description="Minimum salary amount (annual, in local currency)",
        ge=0,
    )
    max_salary: float | None = Field(
        default=None,
        description="Maximum salary amount (annual, in local currency)",
        ge=0,
    )
    currency: str | None = Field(
        default="USD",
        description="Currency code (e.g., 'USD', 'EUR', 'GBP')",
        max_length=10,
    )
    period: str | None = Field(
        default="yearly",
        description="Salary period (yearly, monthly, hourly, etc.)",
        max_length=20,
    )
    is_range: bool = Field(
        description="Whether this represents a salary range or single value",
        default=False,
    )
    confidence: float = Field(
        description="Confidence in salary extraction (0.0 to 1.0)",
        ge=0.0,
        le=1.0,
        default=1.0,
    )

    @field_validator("max_salary")
    @classmethod
    def validate_salary_range(cls, v: float | None, info) -> float | None:
        """Ensure max salary is greater than or equal to min salary."""
        if info.data and "min_salary" in info.data:
            min_salary = info.data["min_salary"]
            if v is not None and min_salary is not None and v < min_salary:
                # Swap values if they're reversed
                return min_salary
        return v
