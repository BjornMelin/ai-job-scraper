"""Test fixtures package for AI Job Scraper.

This package provides comprehensive test fixtures for JobSpy integration testing,
including type-safe factories using polyfactory and extensive edge case coverage.

Available fixtures:
- jobspy_fixtures: Comprehensive JobSpy DataFrame and model fixtures
- Basic fixtures: empty, valid, malformed, edge cases
- Parametrized fixtures: Different response types and sizes
- Request/Result fixtures: JobScrapeRequest and JobScrapeResult instances
- Specialized fixtures: Unicode, salary edge cases, date handling

Usage:
    from tests.fixtures.jobspy_fixtures import (
        valid_jobspy_response,
        JobPostingFactory,
        sample_scrape_request
    )
"""
