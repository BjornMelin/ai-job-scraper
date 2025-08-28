"""Comprehensive tests for JobSpy Pydantic models.

This module tests all JobSpy model validation, including:
- JobScrapeRequest validation and field normalization
- JobPosting validation with edge cases
- JobScrapeResult structure and DataFrame conversion
- Enum validation and normalization (JobSite, JobType, LocationType)
- Field validators for safe type conversion
- Error handling for malformed data
"""

import pandas as pd
import pytest

from pydantic import ValidationError

from src.models.job_models import (
    JobPosting,
    JobScrapeRequest,
    JobScrapeResult,
    JobSite,
    JobType,
    LocationType,
)


class TestJobSiteEnum:
    """Test JobSite enum validation and normalization."""

    def test_jobsite_enum_values(self, jobsite_enum_values):
        """Test all JobSite enum values are valid."""
        expected_sites = {"linkedin", "indeed", "glassdoor", "zip_recruiter", "google"}
        actual_sites = {site.value for site in jobsite_enum_values}
        assert actual_sites == expected_sites

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        (
            ("linkedin", JobSite.LINKEDIN),
            ("LINKEDIN", JobSite.LINKEDIN),
            ("LinkedIn", JobSite.LINKEDIN),
            ("indeed", JobSite.INDEED),
            ("glassdoor", JobSite.GLASSDOOR),
            ("zip_recruiter", JobSite.ZIP_RECRUITER),
            ("zip-recruiter", JobSite.ZIP_RECRUITER),
            ("ziprecruiter", JobSite.ZIP_RECRUITER),
            ("google", JobSite.GOOGLE),
        ),
    )
    def test_jobsite_normalize_valid(self, input_value, expected):
        """Test JobSite normalization with valid inputs."""
        result = JobSite.normalize(input_value)
        assert result == expected

    @pytest.mark.parametrize(
        "input_value",
        (
            None,
            "",
            "unknown",
            "facebook",
            "twitter",
            "   ",
            "123",
        ),
    )
    def test_jobsite_normalize_invalid(self, input_value):
        """Test JobSite normalization with invalid inputs."""
        result = JobSite.normalize(input_value)
        assert result is None

    def test_jobsite_normalize_whitespace(self):
        """Test JobSite normalization handles whitespace."""
        assert JobSite.normalize("  linkedin  ") == JobSite.LINKEDIN
        assert JobSite.normalize("\tindeed\n") == JobSite.INDEED


class TestJobTypeEnum:
    """Test JobType enum validation and normalization."""

    def test_jobtype_enum_values(self, jobtype_enum_values):
        """Test all JobType enum values are valid."""
        expected_types = {"fulltime", "parttime", "contract", "internship", "temporary"}
        actual_types = {jtype.value for jtype in jobtype_enum_values}
        assert actual_types == expected_types

    @pytest.mark.parametrize(
        ("input_value", "expected"),
        (
            ("fulltime", JobType.FULLTIME),
            ("full-time", JobType.FULLTIME),
            ("full_time", JobType.FULLTIME),
            ("full", JobType.FULLTIME),
            ("permanent", JobType.FULLTIME),
            ("parttime", JobType.PARTTIME),
            ("part-time", JobType.PARTTIME),
            ("part", JobType.PARTTIME),
            ("contract", JobType.CONTRACT),
            ("contractor", JobType.CONTRACT),
            ("internship", JobType.INTERNSHIP),
            ("intern", JobType.INTERNSHIP),
            ("temporary", JobType.TEMPORARY),
            ("temp", JobType.TEMPORARY),
        ),
    )
    def test_jobtype_normalize_valid(self, input_value, expected):
        """Test JobType normalization with valid inputs."""
        result = JobType.normalize(input_value)
        assert result == expected

    @pytest.mark.parametrize(
        "input_value",
        (
            None,
            "",
            "unknown",
            "freelance",
            "volunteer",
            "   ",
            "123",
        ),
    )
    def test_jobtype_normalize_invalid(self, input_value):
        """Test JobType normalization with invalid inputs."""
        result = JobType.normalize(input_value)
        assert result is None


class TestLocationTypeEnum:
    """Test LocationType enum and location logic."""

    def test_locationtype_enum_values(self, locationtype_enum_values):
        """Test all LocationType enum values are valid."""
        expected_types = {"remote", "onsite", "hybrid"}
        actual_types = {ltype.value for ltype in locationtype_enum_values}
        assert actual_types == expected_types

    @pytest.mark.parametrize(
        ("is_remote", "location", "expected"),
        (
            (True, None, LocationType.REMOTE),
            (True, "San Francisco", LocationType.REMOTE),
            (False, "Remote", LocationType.REMOTE),
            (False, "Hybrid - San Francisco", LocationType.HYBRID),
            (False, "San Francisco, CA", LocationType.ONSITE),
            (False, None, LocationType.ONSITE),
            (None, "Remote Work", LocationType.REMOTE),
            (None, "Hybrid Office", LocationType.HYBRID),
            (None, "New York, NY", LocationType.ONSITE),
        ),
    )
    def test_locationtype_from_remote_flag(self, is_remote, location, expected):
        """Test LocationType determination from remote flag and location string."""
        result = LocationType.from_remote_flag(is_remote, location)
        assert result == expected

    def test_locationtype_case_insensitive(self):
        """Test LocationType detection is case insensitive."""
        assert LocationType.from_remote_flag(False, "REMOTE") == LocationType.REMOTE
        assert (
            LocationType.from_remote_flag(False, "Hybrid Work") == LocationType.HYBRID
        )


class TestJobScrapeRequest:
    """Test JobScrapeRequest validation and field normalization."""

    def test_jobscrape_request_defaults(self):
        """Test JobScrapeRequest default values."""
        request = JobScrapeRequest()
        assert request.site_name == JobSite.LINKEDIN
        assert request.search_term is None
        assert request.distance == 50
        assert request.is_remote is False
        assert request.results_wanted == 15
        assert request.offset == 0
        assert request.country_indeed == "usa"

    def test_jobscrape_request_valid_creation(self):
        """Test creating valid JobScrapeRequest."""
        request = JobScrapeRequest(
            site_name=[JobSite.LINKEDIN, JobSite.INDEED],
            search_term="Python developer",
            location="San Francisco, CA",
            distance=25,
            is_remote=True,
            job_type=JobType.FULLTIME,
            results_wanted=100,
        )
        assert isinstance(request.site_name, list)
        assert len(request.site_name) == 2
        assert JobSite.LINKEDIN in request.site_name
        assert request.search_term == "Python developer"
        assert request.distance == 25
        assert request.is_remote is True
        assert request.job_type == JobType.FULLTIME
        assert request.results_wanted == 100

    def test_jobscrape_request_site_name_normalization(self):
        """Test site_name field normalization."""
        # String normalization
        request = JobScrapeRequest(site_name="linkedin")
        assert request.site_name == JobSite.LINKEDIN

        # List normalization
        request = JobScrapeRequest(site_name=["linkedin", "indeed"])
        assert request.site_name == [JobSite.LINKEDIN, JobSite.INDEED]

    def test_jobscrape_request_jobtype_normalization(self):
        """Test job_type field normalization."""
        request = JobScrapeRequest(job_type="full-time")
        assert request.job_type == JobType.FULLTIME

    @pytest.mark.parametrize("distance", (-1, 201, 1000))
    def test_jobscrape_request_distance_validation(self, distance):
        """Test distance field validation bounds."""
        with pytest.raises(ValidationError) as exc_info:
            JobScrapeRequest(distance=distance)
        assert "greater than or equal to 0" in str(
            exc_info.value
        ) or "less than or equal to 200" in str(exc_info.value)

    @pytest.mark.parametrize("results_wanted", (0, 1001, -5))
    def test_jobscrape_request_results_validation(self, results_wanted):
        """Test results_wanted field validation bounds."""
        with pytest.raises(ValidationError) as exc_info:
            JobScrapeRequest(results_wanted=results_wanted)
        assert "greater than or equal to 1" in str(
            exc_info.value
        ) or "less than or equal to 1000" in str(exc_info.value)

    def test_jobscrape_request_offset_validation(self):
        """Test offset field validation."""
        with pytest.raises(ValidationError):
            JobScrapeRequest(offset=-1)

    def test_jobscrape_request_hours_old_validation(self):
        """Test hours_old field validation."""
        with pytest.raises(ValidationError):
            JobScrapeRequest(hours_old=0)


class TestJobPosting:
    """Test JobPosting model validation and field processing."""

    def test_jobposting_valid_creation(self, sample_jobspy_raw_data):
        """Test creating valid JobPosting from raw data."""
        job_data = sample_jobspy_raw_data[0]
        posting = JobPosting.model_validate(job_data)

        assert posting.id == "job_001_linkedin"
        assert posting.site == JobSite.LINKEDIN
        assert posting.title == "Senior Python Developer"
        assert posting.company == "TechCorp Inc"
        assert posting.location == "San Francisco, CA"
        assert posting.job_type == JobType.FULLTIME
        assert posting.min_amount == 120000.0
        assert posting.max_amount == 180000.0
        assert posting.is_remote is False
        assert posting.location_type == LocationType.ONSITE

    def test_jobposting_required_fields(self):
        """Test JobPosting with minimal required fields."""
        minimal_data = {
            "id": "test_001",
            "site": "linkedin",
            "title": "Test Job",
            "company": "Test Company",
        }
        posting = JobPosting.model_validate(minimal_data)
        assert posting.id == "test_001"
        assert posting.site == JobSite.LINKEDIN
        assert posting.title == "Test Job"
        assert posting.company == "Test Company"
        assert posting.location is None
        assert posting.is_remote is False
        assert posting.location_type == LocationType.ONSITE

    def test_jobposting_safe_float_conversion(self):
        """Test safe float conversion for salary and rating fields."""
        test_cases = [
            # Valid float conversions
            ({"min_amount": "120000.50"}, 120000.50),
            ({"min_amount": 150000}, 150000.0),
            ({"company_rating": "4.5"}, 4.5),
            # Invalid conversions should return None
            ({"min_amount": "not_a_number"}, None),
            ({"min_amount": ""}, None),
            ({"min_amount": None}, None),
            ({"company_rating": "five_stars"}, None),
        ]

        base_data = {
            "id": "test_001",
            "site": "linkedin",
            "title": "Test Job",
            "company": "Test Company",
        }

        for field_data, expected in test_cases:
            job_data = {**base_data, **field_data}
            posting = JobPosting.model_validate(job_data)
            field_name = next(iter(field_data.keys()))
            assert getattr(posting, field_name) == expected

    def test_jobposting_site_normalization(self):
        """Test site field normalization."""
        job_data = {
            "id": "test_001",
            "site": "LINKEDIN",
            "title": "Test Job",
            "company": "Test Company",
        }
        posting = JobPosting.model_validate(job_data)
        assert posting.site == JobSite.LINKEDIN

    def test_jobposting_jobtype_normalization(self):
        """Test job_type field normalization."""
        job_data = {
            "id": "test_001",
            "site": "linkedin",
            "title": "Test Job",
            "company": "Test Company",
            "job_type": "full-time",
        }
        posting = JobPosting.model_validate(job_data)
        assert posting.job_type == JobType.FULLTIME

    def test_jobposting_location_type_logic(self):
        """Test location_type determination logic."""
        base_data = {
            "id": "test_001",
            "site": "linkedin",
            "title": "Test Job",
            "company": "Test Company",
        }

        # Remote job
        remote_data = {**base_data, "is_remote": True}
        posting = JobPosting.model_validate(remote_data)
        assert posting.location_type == LocationType.REMOTE

        # Hybrid from location string
        hybrid_data = {**base_data, "location": "Hybrid - San Francisco"}
        posting = JobPosting.model_validate(hybrid_data)
        assert posting.location_type == LocationType.HYBRID

        # Onsite default
        onsite_data = {**base_data, "location": "San Francisco, CA"}
        posting = JobPosting.model_validate(onsite_data)
        assert posting.location_type == LocationType.ONSITE

    def test_jobposting_edge_cases(self, edge_case_test_data):
        """Test JobPosting handles edge cases gracefully."""
        base_data = {
            "id": "edge_test",
            "site": "linkedin",
        }

        # Test each edge case category
        for case_data in edge_case_test_data.values():
            job_data = {**base_data, **case_data}
            # Should not raise validation error
            posting = JobPosting.model_validate(job_data)
            assert posting.id == "edge_test"

    def test_jobposting_malformed_data_handling(self):
        """Test JobPosting handles malformed data appropriately."""
        malformed_data = {
            "id": "malformed_test",
            "site": "unknown_site",  # Should be handled by normalization
            "title": "",  # Empty but valid
            "company": None,  # None but should be handled
            "date_posted": "invalid-date",  # Invalid date format
            "is_remote": "maybe",  # Invalid boolean
        }

        # This should either succeed with normalized data or raise ValidationError
        try:
            posting = JobPosting.model_validate(malformed_data)
            # If successful, check normalization worked
            assert posting.id == "malformed_test"
        except ValidationError:
            # If validation fails, that's also acceptable for malformed data
            pass


class TestJobScrapeResult:
    """Test JobScrapeResult model and DataFrame conversion."""

    def test_jobscrape_result_creation(self, sample_job_scrape_result):
        """Test creating JobScrapeResult."""
        result = sample_job_scrape_result
        assert len(result.jobs) == 3
        assert result.total_found == 3
        assert result.job_count == 3
        assert isinstance(result.request_params, JobScrapeRequest)
        assert isinstance(result.metadata, dict)

    def test_jobscrape_result_from_pandas(
        self, sample_jobspy_dataframe, sample_job_scrape_request
    ):
        """Test creating JobScrapeResult from pandas DataFrame."""
        result = JobScrapeResult.from_pandas(
            sample_jobspy_dataframe, sample_job_scrape_request, metadata={"test": True}
        )

        assert len(result.jobs) == len(sample_jobspy_dataframe)
        assert result.total_found == len(sample_jobspy_dataframe)
        assert result.request_params == sample_job_scrape_request
        assert result.metadata["test"] is True

    def test_jobscrape_result_to_pandas(self, sample_job_scrape_result):
        """Test converting JobScrapeResult back to pandas DataFrame."""
        df = sample_job_scrape_result.to_pandas()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_job_scrape_result.jobs)
        assert "id" in df.columns
        assert "title" in df.columns
        assert "company" in df.columns
        assert "site" in df.columns

    def test_jobscrape_result_empty_jobs(self, sample_job_scrape_request):
        """Test JobScrapeResult with empty job list."""
        result = JobScrapeResult(
            jobs=[],
            total_found=0,
            request_params=sample_job_scrape_request,
        )

        assert result.job_count == 0
        assert len(result.jobs) == 0

        # to_pandas should return empty DataFrame
        df = result.to_pandas()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_jobscrape_result_filter_by_location_type(self, sample_job_scrape_result):
        """Test filtering jobs by location type."""
        # Filter for remote jobs (should find the Indeed job)
        remote_result = sample_job_scrape_result.filter_by_location_type(
            LocationType.REMOTE
        )
        assert len(remote_result.jobs) == 1
        assert remote_result.jobs[0].location_type == LocationType.REMOTE
        assert remote_result.total_found == 1

    def test_jobscrape_result_filter_by_job_type(self, sample_job_scrape_result):
        """Test filtering jobs by job type."""
        # Filter for fulltime jobs (should find LinkedIn and Glassdoor jobs)
        fulltime_result = sample_job_scrape_result.filter_by_job_type(JobType.FULLTIME)
        assert len(fulltime_result.jobs) == 2
        assert all(job.job_type == JobType.FULLTIME for job in fulltime_result.jobs)
        assert fulltime_result.total_found == 2

    def test_jobscrape_result_pandas_nan_handling(self, sample_job_scrape_request):
        """Test JobScrapeResult handles pandas NaN values correctly."""
        # Create DataFrame with NaN values
        df_with_nans = pd.DataFrame(
            [
                {
                    "id": "nan_test",
                    "site": "linkedin",
                    "title": "Test Job",
                    "company": "Test Company",
                    "min_amount": float("nan"),
                    "max_amount": pd.NA,
                    "company_rating": None,
                }
            ]
        )

        result = JobScrapeResult.from_pandas(df_with_nans, sample_job_scrape_request)
        job = result.jobs[0]

        assert job.min_amount is None
        assert job.max_amount is None
        assert job.company_rating is None

    def test_jobscrape_result_performance_large_dataset(
        self, performance_test_data, sample_job_scrape_request
    ):
        """Test JobScrapeResult performance with large dataset."""
        import time

        # Create large DataFrame
        df = pd.DataFrame(performance_test_data)

        # Test conversion performance
        start_time = time.time()
        result = JobScrapeResult.from_pandas(df, sample_job_scrape_request)
        conversion_time = time.time() - start_time

        # Should handle 1000 jobs quickly (< 1 second)
        assert conversion_time < 1.0
        assert len(result.jobs) == 1000
        assert result.job_count == 1000

        # Test back-conversion performance
        start_time = time.time()
        converted_df = result.to_pandas()
        back_conversion_time = time.time() - start_time

        assert back_conversion_time < 1.0
        assert len(converted_df) == 1000


class TestJobSpyIntegrationValidation:
    """Test integration validation scenarios for complete JobSpy workflows."""

    def test_end_to_end_dataflow_validation(
        self, mock_jobspy_scrape_success, sample_job_scrape_request
    ):
        """Test complete data flow from request to result."""
        import jobspy

        # Mock JobSpy call
        df = jobspy.scrape_jobs(
            site_name=sample_job_scrape_request.site_name,
            search_term=sample_job_scrape_request.search_term,
            location=sample_job_scrape_request.location,
            results_wanted=sample_job_scrape_request.results_wanted,
        )

        # Convert to JobScrapeResult
        result = JobScrapeResult.from_pandas(df, sample_job_scrape_request)

        # Validate complete workflow
        assert isinstance(result, JobScrapeResult)
        assert len(result.jobs) > 0
        assert all(isinstance(job, JobPosting) for job in result.jobs)
        assert result.request_params == sample_job_scrape_request

    def test_error_handling_validation(
        self, mock_jobspy_scrape_error, sample_job_scrape_request
    ):
        """Test error handling in JobSpy integration."""
        import jobspy

        with pytest.raises(ConnectionError) as exc_info:
            jobspy.scrape_jobs(
                site_name=sample_job_scrape_request.site_name,
                search_term=sample_job_scrape_request.search_term,
            )

        assert "Failed to connect to job site" in str(exc_info.value)

    def test_empty_results_validation(
        self, mock_jobspy_scrape_empty, sample_job_scrape_request
    ):
        """Test handling empty results from JobSpy."""
        import jobspy

        df = jobspy.scrape_jobs(
            site_name=sample_job_scrape_request.site_name,
            search_term=sample_job_scrape_request.search_term,
        )

        result = JobScrapeResult.from_pandas(df, sample_job_scrape_request)

        assert len(result.jobs) == 0
        assert result.job_count == 0
        assert result.total_found == 0

    def test_malformed_data_resilience(
        self, mock_jobspy_scrape_malformed, sample_job_scrape_request
    ):
        """Test resilience to malformed data from JobSpy."""
        import jobspy

        df = jobspy.scrape_jobs(
            site_name=sample_job_scrape_request.site_name,
            search_term=sample_job_scrape_request.search_term,
        )

        # Should handle malformed data gracefully or raise appropriate validation errors
        try:
            result = JobScrapeResult.from_pandas(df, sample_job_scrape_request)
            # If successful, data should be cleaned/normalized
            assert isinstance(result, JobScrapeResult)
        except ValidationError as e:
            # If validation fails, error should be informative
            assert "validation error" in str(e).lower()

    @pytest.mark.parametrize(
        "site_param",
        (
            JobSite.LINKEDIN,
            [JobSite.LINKEDIN, JobSite.INDEED],
            "linkedin",
            ["linkedin", "indeed"],
        ),
    )
    def test_site_parameter_handling(self, mock_jobspy_scrape_success, site_param):
        """Test different site parameter formats are handled correctly."""
        request = JobScrapeRequest(
            site_name=site_param,
            search_term="Python developer",
        )

        # Should normalize site_name properly
        if isinstance(site_param, str):
            assert request.site_name == JobSite.LINKEDIN
        elif isinstance(site_param, list):
            if all(isinstance(s, str) for s in site_param):
                expected_sites = [JobSite.LINKEDIN, JobSite.INDEED]
                assert request.site_name == expected_sites
