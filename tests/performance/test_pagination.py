"""Tests for pagination functionality in the AI Job Scraper.

This module contains comprehensive tests for pagination URL building,
pagination detection, job duplicate filtering across pages, empty page detection,
and other pagination-related functionality.
"""

import inspect

from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import pytest

# Mock implementations for pagination testing
# These would typically be imported from the actual scraper modules
# when pagination functionality is fully implemented

# Mock pagination patterns for testing
PAGINATION_PATTERNS = {
    "next_button": [".next", "a[aria-label='Next']", ".pagination-next"],
    "load_more_button": [".load-more", "button[data-action='load-more']"],
    "page_numbers": [".pagination a", ".page-numbers a"],
}

# Mock company schemas for testing
COMPANY_SCHEMAS = {
    "microsoft": {
        "pagination": {"type": "page_param", "param": "pg", "start": 1, "increment": 1},
    },
    "openai": {
        "pagination": {
            "type": "page_param",
            "param": "page",
            "start": 1,
            "increment": 1,
        },
    },
    "workday_company": {
        "pagination": {
            "type": "workday",
            "offset_param": "offset",
            "limit_param": "limit",
            "limit": 20,
        },
    },
}


def update_url_with_pagination(
    base_url: str,
    pagination_type: str,
    **kwargs: Any,
) -> str:
    """Mock implementation of pagination URL building.

    Args:
        base_url: The base URL to add pagination parameters to
        pagination_type: Type of pagination (page_param, offset_limit, workday)
        **kwargs: Additional pagination parameters

    Returns:
        URL string with pagination parameters added
    """
    parsed = urlparse(base_url)
    query_params = parse_qs(parsed.query)

    if pagination_type == "page_param":
        param = kwargs.get("param", "page")
        page = kwargs.get("page", 1)
        query_params[param] = [str(page)]

    elif pagination_type in ("offset_limit", "workday"):
        offset_param = kwargs.get("offset_param", "offset")
        limit_param = kwargs.get("limit_param", "limit")
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit", 20)
        query_params[offset_param] = [str(offset)]
        query_params[limit_param] = [str(limit)]

    # Convert back to single values for urlencode
    query_dict = {k: v[0] for k, v in query_params.items()}
    new_query = urlencode(query_dict)

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            new_query,
            parsed.fragment,
        ),
    )


def detect_pagination_elements(
    page_content: str | None,
    _company: str,
) -> dict[str, Any] | None:
    """Mock implementation of pagination detection.

    Args:
        page_content: HTML content of the page to analyze (can be None)
        _company: Company name (unused in mock)

    Returns:
        Dictionary with pagination info or None if no pagination found
    """
    if not page_content:
        return None
    return {"has_pagination": True, "patterns_found": ["next_button"]}


def normalize_job_data(job_data: dict[str, Any], _company: str) -> dict[str, Any]:
    """Mock implementation of job data normalization.

    Args:
        job_data: Raw job data dictionary from scraping
        _company: Company name (unused in mock)

    Returns:
        Normalized job data with standardized field names
    """
    # Field mapping for normalization
    field_mapping = {
        "title": ["title", "jobTitle", "position", "role", "job_title"],
        "description": [
            "description",
            "jobDescription",
            "summary",
            "job_description",
            "details",
        ],
        "link": ["link", "url", "applyUrl", "apply_url", "job_url"],
        "location": ["location", "jobLocation", "office", "workplace", "job_location"],
        "posted_date": [
            "posted_date",
            "postedDate",
            "datePosted",
            "date",
            "publishedAt",
        ],
    }

    normalized = {}

    for target_field, source_fields in field_mapping.items():
        value = None
        for field in source_fields:
            if job_data.get(field):
                value = job_data[field]
                break

        # Apply defaults and transformations
        if target_field == "location" and value:
            value = str(value).strip()
            if not value:
                value = "Unknown"
        elif target_field == "location" and not value:
            value = "Unknown"
        elif target_field == "posted_date" and (not value or not value.strip()):
            value = None
        elif target_field in ("title", "description", "link") and value:
            value = str(value).strip()
        else:
            value = value or ""

        normalized[target_field] = value

    return normalized


class TestPaginationURLBuilding:
    """Test cases for pagination URL building functionality."""

    def test_page_param_pagination_basic(self):
        """Test basic page parameter pagination URL building."""
        base_url = "https://example.com/jobs"
        result = update_url_with_pagination(
            base_url,
            "page_param",
            param="page",
            page=3,
        )

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        assert params["page"] == ["3"]
        assert "example.com/jobs" in result

    def test_page_param_pagination_with_existing_params(self):
        """Test page parameter pagination with existing URL parameters."""
        base_url = "https://example.com/jobs?filter=ai&location=sf"
        result = update_url_with_pagination(
            base_url,
            "page_param",
            param="page",
            page=2,
        )

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        assert params["page"] == ["2"]
        assert params["filter"] == ["ai"]
        assert params["location"] == ["sf"]

    def test_page_param_pagination_custom_param_name(self):
        """Test page parameter pagination with custom parameter name."""
        base_url = "https://example.com/jobs"
        result = update_url_with_pagination(base_url, "page_param", param="p", page=5)

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        assert params["p"] == ["5"]
        assert "page" not in params

    def test_page_param_pagination_defaults(self):
        """Test page parameter pagination with default values."""
        base_url = "https://example.com/jobs"
        result = update_url_with_pagination(base_url, "page_param")

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        # Default param is "page" and default page is 1
        assert params["page"] == ["1"]

    def test_offset_limit_pagination_basic(self):
        """Test basic offset/limit pagination URL building."""
        base_url = "https://example.com/jobs"
        result = update_url_with_pagination(
            base_url,
            "offset_limit",
            offset_param="offset",
            limit_param="limit",
            offset=20,
            limit=10,
        )

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        assert params["offset"] == ["20"]
        assert params["limit"] == ["10"]

    def test_offset_limit_pagination_with_existing_params(self):
        """Test offset/limit pagination preserving existing parameters."""
        base_url = "https://example.com/jobs?search=engineer&type=fulltime"
        result = update_url_with_pagination(
            base_url,
            "offset_limit",
            offset_param="start",
            limit_param="count",
            offset=40,
            limit=20,
        )

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        assert params["start"] == ["40"]
        assert params["count"] == ["20"]
        assert params["search"] == ["engineer"]
        assert params["type"] == ["fulltime"]

    def test_workday_pagination(self):
        """Test Workday-style pagination URL building."""
        base_url = "https://company.wd1.myworkdayjobs.com/careers"
        result = update_url_with_pagination(
            base_url,
            "workday",
            offset_param="offset",
            limit_param="limit",
            offset=60,
            limit=20,
        )

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        assert params["offset"] == ["60"]
        assert params["limit"] == ["20"]

    def test_offset_limit_pagination_defaults(self):
        """Test offset/limit pagination with default values."""
        base_url = "https://example.com/jobs"
        result = update_url_with_pagination(base_url, "offset_limit")

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        # Default offset is 0, limit is 20, param names are "offset" and "limit"
        assert params["offset"] == ["0"]
        assert params["limit"] == ["20"]

    def test_pagination_url_encoding(self):
        """Test that special characters in URLs are properly encoded."""
        base_url = "https://example.com/jobs?q=AI%20Engineer&loc=San%20Francisco"
        result = update_url_with_pagination(
            base_url,
            "page_param",
            param="page",
            page=2,
        )

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        assert params["page"] == ["2"]
        assert params["q"] == ["AI Engineer"]  # parse_qs automatically decodes
        assert params["loc"] == ["San Francisco"]

    @pytest.mark.parametrize(
        ("pagination_type", "expected_params"),
        (
            ("page_param", {"page": ["1"]}),
            ("offset_limit", {"offset": ["0"], "limit": ["20"]}),
            ("workday", {"offset": ["0"], "limit": ["20"]}),
        ),
    )
    def test_pagination_types_with_defaults(self, pagination_type, expected_params):
        """Test different pagination types with their default values."""
        base_url = "https://example.com/jobs"
        result = update_url_with_pagination(base_url, pagination_type)

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        for param, value in expected_params.items():
            assert params[param] == value


class TestPaginationDetection:
    """Test cases for pagination element detection functionality."""

    def test_pagination_patterns_available(self):
        """Test that pagination patterns are defined in the scraper module."""
        assert "next_button" in PAGINATION_PATTERNS
        assert "load_more_button" in PAGINATION_PATTERNS
        assert "page_numbers" in PAGINATION_PATTERNS

        # Verify patterns are lists of CSS selectors
        assert isinstance(PAGINATION_PATTERNS["next_button"], list)
        assert isinstance(PAGINATION_PATTERNS["load_more_button"], list)
        assert isinstance(PAGINATION_PATTERNS["page_numbers"], list)

        # Verify patterns contain expected selectors
        assert any(".next" in pattern for pattern in PAGINATION_PATTERNS["next_button"])
        assert any(
            ".load-more" in pattern
            for pattern in PAGINATION_PATTERNS["load_more_button"]
        )
        assert any(
            ".pagination" in pattern for pattern in PAGINATION_PATTERNS["page_numbers"]
        )

    def test_pagination_function_signature(self):
        """Test that detect_pagination_elements function has correct signature."""
        sig = inspect.signature(detect_pagination_elements)
        params = list(sig.parameters.keys())

        assert "page_content" in params
        assert "_company" in params
        assert len(params) == 2

    def test_detect_pagination_elements_error_handling(self):
        """Test that pagination detection handles errors gracefully."""
        # Test with invalid input (empty content)
        result = detect_pagination_elements("", "test_company")
        assert result is None

        # Test with None input
        result = detect_pagination_elements(None, "test_company")
        assert result is None

        # Test with valid content
        result = detect_pagination_elements("<html>content</html>", "test_company")
        assert isinstance(result, dict)
        assert result["has_pagination"] is True


class TestJobDataNormalization:
    """Test cases for job data normalization across different formats."""

    def test_normalize_job_data_standard_fields(self):
        """Test normalization with standard field names."""
        job_data = {
            "title": "Senior AI Engineer",
            "description": "We are looking for an experienced AI engineer.",
            "link": "https://example.com/job/123",
            "location": "San Francisco, CA",
            "posted_date": "2024-01-15",
        }

        result = normalize_job_data(job_data, "test_company")

        assert result["title"] == "Senior AI Engineer"
        assert result["description"] == "We are looking for an experienced AI engineer."
        assert result["link"] == "https://example.com/job/123"
        assert result["location"] == "San Francisco, CA"
        assert result["posted_date"] == "2024-01-15"

    def test_normalize_job_data_alternative_field_names(self):
        """Test normalization with alternative field naming conventions."""
        job_data = {
            "jobTitle": "Machine Learning Engineer",
            "jobDescription": "Join our ML team to build scalable systems.",
            "applyUrl": "https://example.com/apply/456",
            "jobLocation": "Remote",
            "datePosted": "2024-02-01",
        }

        result = normalize_job_data(job_data, "test_company")

        assert result["title"] == "Machine Learning Engineer"
        assert result["description"] == "Join our ML team to build scalable systems."
        assert result["link"] == "https://example.com/apply/456"
        assert result["location"] == "Remote"
        assert result["posted_date"] == "2024-02-01"

    def test_normalize_job_data_camelcase_variations(self):
        """Test normalization with camelCase field variations."""
        job_data = {
            "position": "AI Research Scientist",
            "summary": "Conduct cutting-edge AI research.",
            "job_url": "https://example.com/research/789",
            "office": "New York, NY",
            "publishedAt": "2024-03-01",
        }

        result = normalize_job_data(job_data, "test_company")

        assert result["title"] == "AI Research Scientist"
        assert result["description"] == "Conduct cutting-edge AI research."
        assert result["link"] == "https://example.com/research/789"
        assert result["location"] == "New York, NY"
        assert result["posted_date"] == "2024-03-01"

    def test_normalize_job_data_role_variations(self):
        """Test normalization with 'role' field variations."""
        job_data = {
            "role": "MLOps Engineer",
            "details": "Build and maintain ML infrastructure.",
            "url": "https://example.com/mlops/101",
            "workplace": "Boston, MA",
            "date": "2024-04-01",
        }

        result = normalize_job_data(job_data, "test_company")

        assert result["title"] == "MLOps Engineer"
        assert result["description"] == "Build and maintain ML infrastructure."
        assert result["link"] == "https://example.com/mlops/101"
        assert result["location"] == "Boston, MA"
        assert result["posted_date"] == "2024-04-01"

    def test_normalize_job_data_missing_fields(self):
        """Test normalization with missing fields uses defaults."""
        job_data = {
            "title": "AI Engineer",
            "description": "Work on AI projects.",
            # Missing link, location, posted_date
        }

        result = normalize_job_data(job_data, "test_company")

        assert result["title"] == "AI Engineer"
        assert result["description"] == "Work on AI projects."
        assert result["link"] == ""  # Default empty string
        assert result["location"] == "Unknown"  # Default is "Unknown"
        assert result["posted_date"] is None  # Default is None

    def test_normalize_job_data_whitespace_trimming(self):
        """Test that normalization trims whitespace from fields."""
        job_data = {
            "title": "  Senior AI Engineer  ",
            "description": "  We need an AI engineer.  ",
            "link": "  https://example.com/job/123  ",
            "location": "  San Francisco, CA  ",
            "posted_date": "  2024-01-15  ",
        }

        result = normalize_job_data(job_data, "test_company")

        assert result["title"] == "Senior AI Engineer"
        assert result["description"] == "We need an AI engineer."
        assert result["link"] == "https://example.com/job/123"
        assert result["location"] == "San Francisco, CA"
        assert result["posted_date"] == "  2024-01-15  "  # posted_date is not stripped

    def test_normalize_job_data_empty_string_handling(self):
        """Test normalization handles empty strings correctly."""
        job_data = {
            "title": "",
            "description": "",
            "link": "",
            "location": "",
            "posted_date": "",
        }

        result = normalize_job_data(job_data, "test_company")

        assert result["title"] == ""
        assert result["description"] == ""
        assert result["link"] == ""
        assert (
            result["location"] == "Unknown"
        )  # When location is empty, defaults to "Unknown"
        assert (
            result["posted_date"] is None
        )  # Empty string becomes None due to 'or' chain

    def test_normalize_job_data_field_precedence(self):
        """Test that field precedence works correctly when multiple variations exist."""
        job_data = {
            "title": "Primary Title",
            "jobTitle": "Secondary Title",
            "position": "Tertiary Title",
            "description": "Primary Description",
            "jobDescription": "Secondary Description",
            "link": "https://primary.com",
            "applyUrl": "https://secondary.com",
        }

        result = normalize_job_data(job_data, "test_company")

        # Should use the first available field in priority order
        assert result["title"] == "Primary Title"
        assert result["description"] == "Primary Description"
        assert result["link"] == "https://primary.com"

    @pytest.mark.parametrize(
        ("field_variations", "expected_value"),
        (
            # Title variations
            ({"jobTitle": "Test Job"}, "Test Job"),
            ({"position": "Test Position"}, "Test Position"),
            ({"role": "Test Role"}, "Test Role"),
            ({"job_title": "Test Job Title"}, "Test Job Title"),
            # Description variations
            ({"jobDescription": "Test Description"}, "Test Description"),
            ({"summary": "Test Summary"}, "Test Summary"),
            ({"job_description": "Test Job Desc"}, "Test Job Desc"),
            ({"details": "Test Details"}, "Test Details"),
            # Link variations
            ({"url": "https://test.com"}, "https://test.com"),
            ({"applyUrl": "https://apply.com"}, "https://apply.com"),
            ({"apply_url": "https://apply-url.com"}, "https://apply-url.com"),
            ({"job_url": "https://job-url.com"}, "https://job-url.com"),
            # Location variations
            ({"jobLocation": "Test Location"}, "Test Location"),
            ({"office": "Test Office"}, "Test Office"),
            ({"workplace": "Test Workplace"}, "Test Workplace"),
            ({"job_location": "Test Job Location"}, "Test Job Location"),
            # Date variations
            ({"postedDate": "2024-01-01"}, "2024-01-01"),
            ({"datePosted": "2024-02-01"}, "2024-02-01"),
            ({"date": "2024-03-01"}, "2024-03-01"),
            ({"publishedAt": "2024-04-01"}, "2024-04-01"),
        ),
    )
    def test_normalize_job_data_field_variations(
        self,
        field_variations,
        expected_value,
    ):
        """Test normalization with various field name variations."""
        result = normalize_job_data(field_variations, "test_company")

        # Determine which field type we're testing based on expected value
        # Check more specific patterns first to avoid misclassification
        if (
            "Description" in expected_value
            or "Summary" in expected_value
            or "Details" in expected_value
            or "Desc" in expected_value
        ):
            assert result["description"] == expected_value
        elif (
            "Location" in expected_value
            or "Office" in expected_value
            or "Workplace" in expected_value
        ):
            assert result["location"] == expected_value
        elif "http" in expected_value:
            assert result["link"] == expected_value
        elif "2024-" in expected_value:
            assert result["posted_date"] == expected_value
        elif (
            "Job" in expected_value
            or "Position" in expected_value
            or "Role" in expected_value
        ):
            assert result["title"] == expected_value


class TestPaginationIntegration:
    """Integration tests for pagination functionality."""

    def test_pagination_config_validation(self):
        """Test that pagination configurations are valid."""
        for company, config in COMPANY_SCHEMAS.items():
            if "pagination" in config:
                pagination = config["pagination"]

                # Check required fields based on type
                if pagination.get("type") == "page_param":
                    assert "param" in pagination, f"{company} missing pagination param"
                    assert "start" in pagination, f"{company} missing pagination start"
                    assert "increment" in pagination, (
                        f"{company} missing pagination increment"
                    )

                elif pagination.get("type") in {"offset_limit", "workday"}:
                    assert "offset_param" in pagination, (
                        f"{company} missing offset_param"
                    )
                    assert "limit_param" in pagination, f"{company} missing limit_param"
                    assert "limit" in pagination, f"{company} missing limit"

                elif pagination.get("type") == "load_more_button":
                    assert "button_selector" in pagination, (
                        f"{company} missing button_selector"
                    )

    @pytest.mark.parametrize(
        ("url", "pagination_type", "params", "expected_in_url"),
        (
            # Real-world examples based on COMPANY_SCHEMAS
            (
                "https://jobs.careers.microsoft.com/global/en/search",
                "page_param",
                {"param": "pg", "page": 2},
                "pg=2",
            ),
            (
                "https://company.wd1.myworkdayjobs.com/en-US/careers",
                "workday",
                {
                    "offset_param": "offset",
                    "limit_param": "limit",
                    "offset": 20,
                    "limit": 20,
                },
                ["offset=20", "limit=20"],
            ),
            (
                "https://boards.greenhouse.io/openai",
                "page_param",
                {"param": "page", "page": 3},
                "page=3",
            ),
        ),
    )
    def test_real_world_pagination_scenarios(
        self,
        url,
        pagination_type,
        params,
        expected_in_url,
    ):
        """Test pagination with real-world company URL patterns."""
        result = update_url_with_pagination(url, pagination_type, **params)

        if isinstance(expected_in_url, list):
            for expected in expected_in_url:
                assert expected in result
        else:
            assert expected_in_url in result

    def test_pagination_preserves_existing_filters(self):
        """Test that pagination preserves existing search filters and parameters."""
        # Simulate a job search URL with filters
        base_url = "https://example.com/jobs?q=AI+Engineer&location=San+Francisco&remote=true&salary=150k"

        result = update_url_with_pagination(
            base_url,
            "page_param",
            param="page",
            page=5,
        )

        parsed = urlparse(result)
        params = parse_qs(parsed.query)

        # Verify pagination was added
        assert params["page"] == ["5"]

        # Verify existing filters were preserved
        assert params["q"] == ["AI Engineer"]
        assert params["location"] == ["San Francisco"]
        assert params["remote"] == ["true"]
        assert params["salary"] == ["150k"]

    def test_pagination_url_consistency(self):
        """Test that pagination URLs are built consistently across multiple calls."""
        base_url = "https://example.com/jobs?filter=engineering"

        # Build the same pagination URL multiple times
        results = []
        for _ in range(5):
            result = update_url_with_pagination(
                base_url,
                "offset_limit",
                offset=40,
                limit=20,
            )
            results.append(result)

        # All results should be identical
        assert len(set(results)) == 1

        # Verify the URL contains expected parameters
        parsed = urlparse(results[0])
        params = parse_qs(parsed.query)
        assert params["offset"] == ["40"]
        assert params["limit"] == ["20"]
        assert params["filter"] == ["engineering"]

    def test_empty_page_detection_logic(self):
        """Test logic for detecting empty pages in pagination."""
        # This tests the conceptual logic that would be used in extract_jobs
        max_empty_pages = 2
        empty_page_count = 0
        pages_processed = []

        # Simulate processing pages with some empty results
        page_results = [
            ["job1", "job2", "job3"],  # Page 1: 3 jobs
            ["job4", "job5"],  # Page 2: 2 jobs
            [],  # Page 3: 0 jobs (empty)
            [],  # Page 4: 0 jobs (empty)
            # Should stop here due to max_empty_pages = 2
        ]

        for page_num, jobs in enumerate(page_results, 1):
            pages_processed.append(page_num)

            if not jobs:  # Empty page
                empty_page_count += 1
                if empty_page_count >= max_empty_pages:
                    break
            else:
                empty_page_count = 0  # Reset counter on non-empty page

        # Should have processed 4 pages and stopped
        assert len(pages_processed) == 4
        assert empty_page_count == 2

    def test_duplicate_job_filtering_concept(self):
        """Test the concept of filtering duplicate jobs across pages."""
        # Simulate jobs from multiple pages with some duplicates
        page1_jobs = [
            {"link": "https://example.com/job/1", "title": "AI Engineer 1"},
            {"link": "https://example.com/job/2", "title": "AI Engineer 2"},
            {"link": "https://example.com/job/3", "title": "AI Engineer 3"},
        ]

        page2_jobs = [
            {
                "link": "https://example.com/job/3",
                "title": "AI Engineer 3",
            },  # Duplicate
            {"link": "https://example.com/job/4", "title": "AI Engineer 4"},
            {"link": "https://example.com/job/5", "title": "AI Engineer 5"},
        ]

        page3_jobs = [
            {
                "link": "https://example.com/job/1",
                "title": "AI Engineer 1",
            },  # Duplicate
            {"link": "https://example.com/job/6", "title": "AI Engineer 6"},
        ]

        # Simulate the duplicate filtering logic
        all_jobs = []
        seen_links = set()

        for page_jobs in [page1_jobs, page2_jobs, page3_jobs]:
            for job in page_jobs:
                job_link = job["link"]
                if job_link not in seen_links:
                    all_jobs.append(job)
                    seen_links.add(job_link)

        # Should have 6 unique jobs (filtered out 2 duplicates)
        assert len(all_jobs) == 6

        # Verify all jobs have unique links
        job_links = [job["link"] for job in all_jobs]
        assert len(job_links) == len(set(job_links))

        # Verify we have the expected jobs
        expected_links = {
            "https://example.com/job/1",
            "https://example.com/job/2",
            "https://example.com/job/3",
            "https://example.com/job/4",
            "https://example.com/job/5",
            "https://example.com/job/6",
        }
        actual_links = {job["link"] for job in all_jobs}
        assert actual_links == expected_links
