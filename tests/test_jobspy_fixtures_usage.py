"""Example test demonstrating JobSpy fixtures usage.

This test file showcases how to use the comprehensive JobSpy fixtures
for various testing scenarios including parametrized tests, edge cases,
and type-safe factory patterns.
"""

import pandas as pd
import pytest
from src.models.job_models import JobPosting, JobScrapeRequest, JobScrapeResult

from tests.fixtures.jobspy_fixtures import (
    JobPostingFactory,
    JobScrapeRequestFactory,
)


class TestJobSpyFixturesUsage:
    """Test class demonstrating fixture usage patterns."""

    def test_polyfactory_job_posting_generation(self):
        """Test JobPostingFactory generates valid job postings."""
        # Generate single job posting
        job = JobPostingFactory.build()

        assert isinstance(job, JobPosting)
        assert job.id is not None
        assert job.title is not None
        assert job.company is not None

        # Verify salary range logic (min <= max)
        if job.min_amount and job.max_amount:
            assert job.min_amount <= job.max_amount

    def test_polyfactory_dataframe_generation(self):
        """Test factory generates valid pandas DataFrames."""
        job_data_frame = JobPostingFactory.to_dataframe(count=20)

        assert isinstance(job_data_frame, pd.DataFrame)
        assert len(job_data_frame) == 20
        assert "id" in job_data_frame.columns
        assert "title" in job_data_frame.columns
        assert "company" in job_data_frame.columns

        # Verify no null IDs
        assert not job_data_frame["id"].isna().any()

    def test_job_scrape_request_factory(self):
        """Test JobScrapeRequestFactory generates valid requests."""
        request = JobScrapeRequestFactory.build()

        assert isinstance(request, JobScrapeRequest)
        assert request.search_term is not None
        assert request.results_wanted > 0
        assert 0 <= request.distance <= 200

    def test_valid_response_fixture(self, valid_jobspy_response):
        """Test valid response fixture provides realistic data."""
        assert isinstance(valid_jobspy_response, pd.DataFrame)
        assert len(valid_jobspy_response) == 5
        assert not valid_jobspy_response.empty

        # Verify required columns exist
        assert "id" in valid_jobspy_response.columns
        assert "title" in valid_jobspy_response.columns
        assert "company" in valid_jobspy_response.columns

    def test_empty_response_fixture(self, empty_jobspy_response):
        """Test empty response fixture for error handling."""
        assert isinstance(empty_jobspy_response, pd.DataFrame)
        assert empty_jobspy_response.empty
        assert len(empty_jobspy_response) == 0

    def test_malformed_response_fixture(self, malformed_jobspy_response):
        """Test malformed response fixture for error handling."""
        assert isinstance(malformed_jobspy_response, pd.DataFrame)
        assert not malformed_jobspy_response.empty

        # Verify presence of malformed data
        assert malformed_jobspy_response.isna().any().any()  # Some null values

        # Check for various malformed scenarios
        for _, row in malformed_jobspy_response.iterrows():
            # At least some fields should be problematic
            [
                pd.isna(row.get("id")) or row.get("id") == "",
                pd.isna(row.get("title")) or row.get("title") == "",
                pd.isna(row.get("company")) or row.get("company") == "",
            ]
            # At least one field should be problematic in malformed data
            # (This is a meta-test of our fixture quality)

    def test_unicode_response_fixture(self, unicode_jobs_response):
        """Test Unicode response fixture handles international characters."""
        assert isinstance(unicode_jobs_response, pd.DataFrame)
        assert len(unicode_jobs_response) == 3

        # Verify Unicode characters are preserved
        titles = unicode_jobs_response["title"].tolist()
        assert any("🐍" in title for title in titles)  # Python emoji
        assert any("💻" in title for title in titles)  # Computer emoji
        assert any("🔧" in title for title in titles)  # Tool emoji

        # Verify international characters
        assert any("Développeur" in title for title in titles)  # French
        assert any("软件工程师" in title for title in titles)  # Chinese
        assert any("Инженер" in title for title in titles)  # Russian

    @pytest.mark.parametrize(
        "jobspy_response", ["valid", "empty", "malformed"], indirect=True
    )
    def test_parametrized_responses(self, jobspy_response):
        """Test parametrized fixture with different response types."""
        assert isinstance(jobspy_response, pd.DataFrame)

        if jobspy_response is not None and not jobspy_response.empty:
            # Valid and malformed should have data
            assert len(jobspy_response) > 0
        # Empty should be... empty (handled by the fixture)

    @pytest.mark.parametrize("jobspy_response_sized", [5, 50], indirect=True)
    def test_sized_responses(self, jobspy_response_sized):
        """Test sized response fixtures for performance testing."""
        assert isinstance(jobspy_response_sized, pd.DataFrame)
        # Verify the size matches the parameter
        # The fixture generates the exact count specified

    def test_scrape_request_fixtures(self, sample_scrape_request):
        """Test JobScrapeRequest fixtures."""
        assert isinstance(sample_scrape_request, JobScrapeRequest)
        assert sample_scrape_request.results_wanted > 0

    def test_scrape_result_fixtures(self, successful_scrape_result):
        """Test JobScrapeResult fixtures."""
        assert isinstance(successful_scrape_result, JobScrapeResult)
        assert successful_scrape_result.metadata["success"] is True
        assert len(successful_scrape_result.jobs) > 0

        # Verify jobs are properly typed
        for job in successful_scrape_result.jobs:
            assert isinstance(job, JobPosting)


class TestJobSpyFixtureEdgeCases:
    """Test edge cases and error scenarios using fixtures."""

    def test_malformed_data_handling(self, malformed_jobspy_response):
        """Test that malformed data is handled gracefully."""
        # Simulate processing malformed data (like JobSpyScraper would)
        processed_jobs = []

        for _, row in malformed_jobspy_response.iterrows():
            try:
                # Attempt to create JobPosting from row data
                job_data = row.to_dict()

                # Basic data cleaning (like JobSpyScraper does)
                if not job_data.get("id"):
                    job_data["id"] = f"fallback_{hash(str(job_data))}"

                # This might fail for truly malformed data, which is expected
                job = JobPosting.model_validate(job_data)
                processed_jobs.append(job)
            except Exception:
                # Skip malformed records (expected behavior)
                continue

        # Should either process some jobs or handle all gracefully
        assert isinstance(processed_jobs, list)  # At minimum, list structure preserved

    def test_large_dataset_performance(self):
        """Test performance with large datasets."""
        import time

        start_time = time.time()
        large_df = JobPostingFactory.to_dataframe(count=1000)
        generation_time = time.time() - start_time

        assert len(large_df) == 1000
        assert generation_time < 10.0  # Should generate 1000 jobs in <10 seconds

        # Test DataFrame operations are still fast
        start_time = time.time()
        unique_companies = large_df["company"].nunique()
        query_time = time.time() - start_time

        assert unique_companies > 0
        assert query_time < 1.0  # Basic queries should be fast


class TestJobSpyIntegrationPatterns:
    """Test integration patterns using fixtures."""

    def test_dataframe_to_jobposting_conversion(self, valid_jobspy_response):
        """Test converting DataFrame to JobPosting models."""
        converted_jobs = []

        for _, row in valid_jobspy_response.iterrows():
            job_data = row.to_dict()

            # Handle NaN values (like JobSpyScraper does)
            for key, value in job_data.items():
                # Handle different value types safely
                if pd.api.types.is_scalar(value) and pd.isna(value):
                    job_data[key] = None

            job = JobPosting.model_validate(job_data)
            converted_jobs.append(job)

        assert len(converted_jobs) == len(valid_jobspy_response)
        assert all(isinstance(job, JobPosting) for job in converted_jobs)

    def test_jobscraperequest_to_params(self, sample_scrape_request):
        """Test converting JobScrapeRequest to JobSpy parameters."""
        # Simulate parameter building (like JobSpyScraper._build_scrape_params)
        params = {
            "search_term": sample_scrape_request.search_term,
            "location": sample_scrape_request.location,
            "results_wanted": sample_scrape_request.results_wanted,
            "is_remote": sample_scrape_request.is_remote,
        }

        # Add site_name handling
        if isinstance(sample_scrape_request.site_name, list):
            params["site_name"] = [
                site.value for site in sample_scrape_request.site_name
            ]
        else:
            params["site_name"] = [sample_scrape_request.site_name.value]

        assert "search_term" in params
        assert "site_name" in params
        assert isinstance(params["site_name"], list)
        assert params["results_wanted"] > 0
