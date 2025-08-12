"""Comprehensive tests for scraper_company_pages.py module.

This module provides extensive test coverage for the company page scraping workflow,
including LangGraph orchestration, SmartScraperMultiGraph integration, state management,
proxy configuration, and error handling scenarios.
"""

import hashlib

from datetime import datetime, timezone
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.models import CompanySQL, JobSQL
from src.scraper_company_pages import (
    State,
    _add_proxy_config,
    extract_details,
    extract_job_lists,
    get_normalized_jobs,
    load_active_companies,
    normalize_jobs,
    scrape_company_pages,
)


class TestLoadActiveCompanies:
    """Test suite for database company loading."""

    @patch("src.scraper_company_pages.SessionLocal")
    def test_load_active_companies_success(self, mock_session_local: Mock) -> None:
        """Test successful loading of active companies."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock database results
        mock_companies = [
            CompanySQL(
                id=1, name="Active Corp 1", url="https://corp1.com/careers", active=True
            ),
            CompanySQL(
                id=2, name="Active Corp 2", url="https://corp2.com/careers", active=True
            ),
        ]
        mock_session.exec.return_value.all.return_value = mock_companies

        result = load_active_companies()

        assert len(result) == 2
        assert result[0].name == "Active Corp 1"
        assert result[1].name == "Active Corp 2"
        assert all(company.active for company in result)

        # Verify session was properly closed
        mock_session.close.assert_called_once()

    @patch("src.scraper_company_pages.SessionLocal")
    def test_load_active_companies_empty_result(self, mock_session_local: Mock) -> None:
        """Test handling when no active companies exist."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_session.exec.return_value.all.return_value = []

        result = load_active_companies()

        assert result == []

    @patch("src.scraper_company_pages.SessionLocal")
    @patch("src.scraper_company_pages.logger")
    def test_load_active_companies_database_error(
        self, mock_logger: Mock, mock_session_local: Mock
    ) -> None:
        """Test handling database errors during company loading."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock database error
        mock_session.exec.side_effect = Exception("Database connection failed")

        with pytest.raises(Exception, match="Database connection failed"):
            load_active_companies()


class TestProxyConfiguration:
    """Test suite for proxy configuration functionality."""

    @patch("src.scraper_company_pages.settings")
    @patch("src.scraper_company_pages.get_proxy")
    def test_add_proxy_config_enabled(
        self, mock_get_proxy: Mock, mock_settings: Mock
    ) -> None:
        """Test adding proxy configuration when proxies are enabled."""
        mock_settings.use_proxies = True
        mock_settings.proxy_pool = ["proxy1:8080", "proxy2:8080"]
        mock_get_proxy.return_value = "http://proxy1:8080"

        base_config = {
            "llm": {"model": "test-model"},
            "verbose": True,
        }

        result = _add_proxy_config(base_config, "job list")

        assert "loader_kwargs" in result
        assert "proxy" in result["loader_kwargs"]
        assert result["loader_kwargs"]["proxy"]["server"] == "http://proxy1:8080"

        # Original config should be preserved
        assert result["llm"]["model"] == "test-model"
        assert result["verbose"] is True

    @patch("src.scraper_company_pages.settings")
    def test_add_proxy_config_disabled(self, mock_settings: Mock) -> None:
        """Test proxy configuration when proxies are disabled."""
        mock_settings.use_proxies = False

        base_config = {
            "llm": {"model": "test-model"},
            "verbose": True,
        }

        result = _add_proxy_config(base_config, "job list")

        # Config should remain unchanged
        assert result == base_config
        assert "loader_kwargs" not in result

    @patch("src.scraper_company_pages.settings")
    @patch("src.scraper_company_pages.get_proxy")
    def test_add_proxy_config_no_available_proxy(
        self, mock_get_proxy: Mock, mock_settings: Mock
    ) -> None:
        """Test handling when no proxy is available."""
        mock_settings.use_proxies = True
        mock_settings.proxy_pool = []
        mock_get_proxy.return_value = None

        base_config = {"llm": {"model": "test-model"}}

        result = _add_proxy_config(base_config, "job list")

        # Config should remain unchanged when no proxy available
        assert result == base_config
        assert "loader_kwargs" not in result


class TestExtractJobLists:
    """Test suite for job list extraction functionality."""

    @patch("src.scraper_company_pages.SmartScraperMultiGraph")
    @patch("src.scraper_company_pages._add_proxy_config")
    @patch("src.scraper_company_pages.random_user_agent")
    @patch("src.scraper_company_pages.random_delay")
    def test_extract_job_lists_success(
        self,
        mock_delay: Mock,
        mock_user_agent: Mock,
        mock_proxy_config: Mock,
        mock_scraper_class: Mock,
    ) -> None:
        """Test successful job list extraction."""
        mock_user_agent.return_value = "TestAgent/1.0"
        mock_proxy_config.return_value = {"test": "config"}

        # Mock state
        state: State = {
            "companies": [
                CompanySQL(
                    id=1, name="Test Corp", url="https://test.com/careers", active=True
                ),
                CompanySQL(
                    id=2, name="Demo Inc", url="https://demo.com/jobs", active=True
                ),
            ],
            "max_jobs_per_company": 10,
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        # Mock scraper results
        mock_scraper_instance = Mock()
        mock_scraper_class.return_value = mock_scraper_instance
        mock_scraper_instance.run.return_value = {
            "https://test.com/careers": {
                "jobs": [
                    {"title": "AI Engineer", "url": "/careers/ai-engineer"},
                    {"title": "ML Engineer", "url": "/careers/ml-engineer"},
                ]
            },
            "https://demo.com/jobs": {
                "jobs": [
                    {"title": "Data Scientist", "url": "/jobs/data-scientist"},
                ]
            },
        }

        result = extract_job_lists(state)

        # Verify result structure
        assert "partial_jobs" in result
        partial_jobs = result["partial_jobs"]
        assert len(partial_jobs) == 3

        # Verify job details
        ai_job = next(job for job in partial_jobs if job["title"] == "AI Engineer")
        assert ai_job["company"] == "Test Corp"
        assert ai_job["url"] == "https://test.com/careers/ai-engineer"

        # Verify scraper was called with correct parameters
        mock_scraper_class.assert_called_once()
        call_args = mock_scraper_class.call_args
        assert "Extract up to 10 job listings" in call_args[0][0]  # Prompt
        assert len(call_args[0][1]) == 2  # Two company URLs

    @patch("src.scraper_company_pages.SmartScraperMultiGraph")
    def test_extract_job_lists_no_sources(self, mock_scraper_class: Mock) -> None:
        """Test handling when no companies are provided."""
        state: State = {
            "companies": [],
            "max_jobs_per_company": 10,
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        result = extract_job_lists(state)

        assert result == {"partial_jobs": []}
        mock_scraper_class.assert_not_called()

    @patch("src.scraper_company_pages.SmartScraperMultiGraph")
    @patch("src.scraper_company_pages._add_proxy_config")
    @patch("src.scraper_company_pages.random_user_agent")
    @patch("src.scraper_company_pages.random_delay")
    @patch("src.scraper_company_pages.logger")
    def test_extract_job_lists_extraction_failure(
        self,
        mock_logger: Mock,
        mock_delay: Mock,
        mock_user_agent: Mock,
        mock_proxy_config: Mock,
        mock_scraper_class: Mock,
    ) -> None:
        """Test handling extraction failures and malformed responses."""
        mock_user_agent.return_value = "TestAgent/1.0"
        mock_proxy_config.return_value = {"test": "config"}

        state: State = {
            "companies": [
                CompanySQL(
                    id=1, name="Test Corp", url="https://test.com/careers", active=True
                ),
            ],
            "max_jobs_per_company": 10,
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        # Mock scraper to return malformed results
        mock_scraper_instance = Mock()
        mock_scraper_class.return_value = mock_scraper_instance
        mock_scraper_instance.run.return_value = {
            "https://test.com/careers": "invalid response format"  # Not dict
        }

        result = extract_job_lists(state)

        assert result == {"partial_jobs": []}
        mock_logger.warning.assert_called_with(
            "Failed to extract jobs from %s", "https://test.com/careers"
        )

    @patch("src.scraper_company_pages.SmartScraperMultiGraph")
    @patch("src.scraper_company_pages._add_proxy_config")
    @patch("src.scraper_company_pages.random_user_agent")
    @patch("src.scraper_company_pages.random_delay")
    def test_extract_job_lists_job_limit_enforcement(
        self,
        mock_delay: Mock,
        mock_user_agent: Mock,
        mock_proxy_config: Mock,
        mock_scraper_class: Mock,
    ) -> None:
        """Test that job limits per company are properly enforced."""
        mock_user_agent.return_value = "TestAgent/1.0"
        mock_proxy_config.return_value = {"test": "config"}

        state: State = {
            "companies": [
                CompanySQL(
                    id=1, name="Test Corp", url="https://test.com/careers", active=True
                ),
            ],
            "max_jobs_per_company": 2,  # Limit to 2 jobs
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        # Mock scraper to return more jobs than the limit
        mock_scraper_instance = Mock()
        mock_scraper_class.return_value = mock_scraper_instance
        mock_scraper_instance.run.return_value = {
            "https://test.com/careers": {
                "jobs": [
                    {"title": "Job 1", "url": "/job1"},
                    {"title": "Job 2", "url": "/job2"},
                    {"title": "Job 3", "url": "/job3"},  # Should be filtered out
                    {"title": "Job 4", "url": "/job4"},  # Should be filtered out
                ]
            }
        }

        result = extract_job_lists(state)

        partial_jobs = result["partial_jobs"]
        assert len(partial_jobs) == 2  # Limited to 2 jobs
        assert partial_jobs[0]["title"] == "Job 1"
        assert partial_jobs[1]["title"] == "Job 2"

    @patch("src.scraper_company_pages.SmartScraperMultiGraph")
    @patch("src.scraper_company_pages._add_proxy_config")
    @patch("src.scraper_company_pages.random_user_agent")
    @patch("src.scraper_company_pages.random_delay")
    def test_extract_job_lists_url_joining(
        self,
        mock_delay: Mock,
        mock_user_agent: Mock,
        mock_proxy_config: Mock,
        mock_scraper_class: Mock,
    ) -> None:
        """Test proper URL joining for relative job URLs."""
        mock_user_agent.return_value = "TestAgent/1.0"
        mock_proxy_config.return_value = {"test": "config"}

        state: State = {
            "companies": [
                CompanySQL(
                    id=1, name="Test Corp", url="https://test.com/careers/", active=True
                ),
            ],
            "max_jobs_per_company": 10,
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        mock_scraper_instance = Mock()
        mock_scraper_class.return_value = mock_scraper_instance
        mock_scraper_instance.run.return_value = {
            "https://test.com/careers/": {
                "jobs": [
                    {"title": "Job 1", "url": "/jobs/1"},  # Relative URL
                    {"title": "Job 2", "url": "jobs/2"},  # Relative URL
                    {
                        "title": "Job 3",
                        "url": "https://test.com/jobs/3",
                    },  # Absolute URL
                ]
            }
        }

        result = extract_job_lists(state)

        partial_jobs = result["partial_jobs"]
        assert len(partial_jobs) == 3

        # Verify URLs were properly joined
        assert partial_jobs[0]["url"] == "https://test.com/jobs/1"
        assert partial_jobs[1]["url"] == "https://test.com/careers/jobs/2"
        assert partial_jobs[2]["url"] == "https://test.com/jobs/3"


class TestExtractDetails:
    """Test suite for job detail extraction functionality."""

    @patch("src.scraper_company_pages.SmartScraperMultiGraph")
    @patch("src.scraper_company_pages._add_proxy_config")
    @patch("src.scraper_company_pages.random_user_agent")
    @patch("src.scraper_company_pages.random_delay")
    def test_extract_details_success(
        self,
        mock_delay: Mock,
        mock_user_agent: Mock,
        mock_proxy_config: Mock,
        mock_scraper_class: Mock,
    ) -> None:
        """Test successful job detail extraction."""
        mock_user_agent.return_value = "TestAgent/1.0"
        mock_proxy_config.return_value = {"test": "config"}

        state: State = {
            "companies": [],
            "partial_jobs": [
                {
                    "company": "Test Corp",
                    "title": "AI Engineer",
                    "url": "https://test.com/jobs/ai-engineer",
                },
                {
                    "company": "Demo Inc",
                    "title": "ML Engineer",
                    "url": "https://demo.com/jobs/ml-engineer",
                },
            ],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        # Mock scraper results
        mock_scraper_instance = Mock()
        mock_scraper_class.return_value = mock_scraper_instance
        mock_scraper_instance.run.return_value = {
            "https://test.com/jobs/ai-engineer": {
                "description": "We are looking for an experienced AI engineer...",
                "location": "San Francisco, CA",
                "posted_date": "2024-01-15",
                "salary": "$120k-180k",
                "link": "https://test.com/apply/ai-engineer",
            },
            "https://demo.com/jobs/ml-engineer": {
                "description": "Join our ML team...",
                "location": "Remote",
                "posted_date": "2024-01-16",
                "salary": "$100k-160k",
                "link": "https://demo.com/apply/ml-engineer",
            },
        }

        result = extract_details(state)

        # Verify result structure
        assert "raw_full_jobs" in result
        raw_jobs = result["raw_full_jobs"]
        assert len(raw_jobs) == 2

        # Verify merged data
        ai_job = next(job for job in raw_jobs if job["title"] == "AI Engineer")
        assert ai_job["company"] == "Test Corp"
        assert (
            ai_job["description"] == "We are looking for an experienced AI engineer..."
        )
        assert ai_job["location"] == "San Francisco, CA"
        assert ai_job["salary"] == "$120k-180k"

    @patch("src.scraper_company_pages.SmartScraperMultiGraph")
    def test_extract_details_no_jobs(self, mock_scraper_class: Mock) -> None:
        """Test handling when no partial jobs are provided."""
        state: State = {
            "companies": [],
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        result = extract_details(state)

        assert result == {"raw_full_jobs": []}
        mock_scraper_class.assert_not_called()

    @patch("src.scraper_company_pages.SmartScraperMultiGraph")
    @patch("src.scraper_company_pages._add_proxy_config")
    @patch("src.scraper_company_pages.random_user_agent")
    @patch("src.scraper_company_pages.random_delay")
    @patch("src.scraper_company_pages.logger")
    def test_extract_details_extraction_failure(
        self,
        mock_logger: Mock,
        mock_delay: Mock,
        mock_user_agent: Mock,
        mock_proxy_config: Mock,
        mock_scraper_class: Mock,
    ) -> None:
        """Test handling extraction failures for job details."""
        mock_user_agent.return_value = "TestAgent/1.0"
        mock_proxy_config.return_value = {"test": "config"}

        state: State = {
            "companies": [],
            "partial_jobs": [
                {
                    "company": "Test Corp",
                    "title": "AI Engineer",
                    "url": "https://test.com/jobs/ai-engineer",
                }
            ],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        # Mock scraper to return invalid response
        mock_scraper_instance = Mock()
        mock_scraper_class.return_value = mock_scraper_instance
        mock_scraper_instance.run.return_value = {
            "https://test.com/jobs/ai-engineer": "invalid response"  # Not a dict
        }

        result = extract_details(state)

        assert result == {"raw_full_jobs": []}
        mock_logger.warning.assert_called_with(
            "Failed to extract details from %s", "https://test.com/jobs/ai-engineer"
        )


class TestNormalizeJobs:
    """Test suite for job data normalization."""

    @patch("src.scraper_company_pages.SessionLocal")
    def test_normalize_jobs_success(self, mock_session_local: Mock) -> None:
        """Test successful job normalization."""
        # Mock session and company lookup
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_company = CompanySQL(
            id=1, name="Test Corp", url="https://test.com", active=True
        )
        mock_session.exec.return_value.first.return_value = mock_company

        state: State = {
            "companies": [],
            "partial_jobs": [],
            "raw_full_jobs": [
                {
                    "company": "Test Corp",
                    "title": "AI Engineer",
                    "description": "AI role description",
                    "url": "https://test.com/jobs/ai-engineer",
                    "location": "San Francisco, CA",
                    "posted_date": "2024-01-15",
                    "salary": "$120k-180k",
                    "link": "https://test.com/apply/ai-engineer",
                }
            ],
            "normalized_jobs": [],
        }

        result = normalize_jobs(state)

        # Verify result
        assert "normalized_jobs" in result
        normalized_jobs = result["normalized_jobs"]
        assert len(normalized_jobs) == 1

        job = normalized_jobs[0]
        assert isinstance(job, JobSQL)
        assert job.title == "AI Engineer"
        assert job.company_id == 1
        assert job.description == "AI role description"
        assert job.location == "San Francisco, CA"
        assert job.salary == "$120k-180k"
        assert job.application_status == "New"
        assert job.content_hash is not None

    @patch("src.scraper_company_pages.SessionLocal")
    def test_normalize_jobs_date_parsing_various_formats(
        self, mock_session_local: Mock
    ) -> None:
        """Test date parsing for various date formats."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_company = CompanySQL(
            id=1, name="Test Corp", url="https://test.com", active=True
        )
        mock_session.exec.return_value.first.return_value = mock_company

        test_cases = [
            ("2024-01-15", "2024-01-15T00:00:00+00:00"),  # ISO format
            ("01/15/2024", "2024-01-15T00:00:00+00:00"),  # US format
            ("January 15, 2024", "2024-01-15T00:00:00+00:00"),  # Long format
            ("15 January 2024", "2024-01-15T00:00:00+00:00"),  # International format
            ("invalid date", None),  # Invalid format
            (None, None),  # No date
        ]

        for date_input, expected_iso in test_cases:
            state: State = {
                "companies": [],
                "partial_jobs": [],
                "raw_full_jobs": [
                    {
                        "company": "Test Corp",
                        "title": f"Job for {date_input}",
                        "description": "Description",
                        "url": "https://test.com/job1",
                        "posted_date": date_input,
                    }
                ],
                "normalized_jobs": [],
            }

            result = normalize_jobs(state)
            normalized_jobs = result["normalized_jobs"]

            if expected_iso:
                assert len(normalized_jobs) == 1
                job = normalized_jobs[0]
                assert job.posted_date is not None
                assert job.posted_date.isoformat() == expected_iso
            else:
                # Should still create job even with invalid date
                assert len(normalized_jobs) == 1
                assert normalized_jobs[0].posted_date is None

    @patch("src.scraper_company_pages.SessionLocal")
    def test_normalize_jobs_company_creation(self, mock_session_local: Mock) -> None:
        """Test creating new companies during normalization."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock no existing company found
        mock_session.exec.return_value.first.return_value = None

        # Mock company creation
        mock_new_company = CompanySQL(id=2, name="New Corp", url="", active=True)
        mock_session.add.return_value = None
        mock_session.commit.return_value = None
        mock_session.refresh.return_value = None
        mock_new_company.id = 2  # Simulate database assignment

        state: State = {
            "companies": [],
            "partial_jobs": [],
            "raw_full_jobs": [
                {
                    "company": "New Corp",
                    "title": "Engineer",
                    "description": "Description",
                    "url": "https://new.com/job1",
                }
            ],
            "normalized_jobs": [],
        }

        # Mock the company creation flow
        def mock_exec_side_effect(_query):
            mock_result = Mock()
            mock_result.first.return_value = None
            return mock_result

        mock_session.exec.side_effect = mock_exec_side_effect

        def mock_add_side_effect(company):
            company.id = 2  # Simulate database assignment

        mock_session.add.side_effect = mock_add_side_effect

        result = normalize_jobs(state)

        normalized_jobs = result["normalized_jobs"]
        assert len(normalized_jobs) == 1

        # Verify company was created
        mock_session.add.assert_called()
        mock_session.commit.assert_called()

    @patch("src.scraper_company_pages.SessionLocal")
    @patch("src.scraper_company_pages.logger")
    def test_normalize_jobs_content_hash_generation(
        self, mock_logger: Mock, mock_session_local: Mock
    ) -> None:
        """Test content hash generation for duplicate detection."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        mock_company = CompanySQL(
            id=1, name="Test Corp", url="https://test.com", active=True
        )
        mock_session.exec.return_value.first.return_value = mock_company

        state: State = {
            "companies": [],
            "partial_jobs": [],
            "raw_full_jobs": [
                {
                    "company": "Test Corp",
                    "title": "AI Engineer",
                    "description": "Job description",
                    "url": "https://test.com/job1",
                }
            ],
            "normalized_jobs": [],
        }

        result = normalize_jobs(state)

        normalized_jobs = result["normalized_jobs"]
        assert len(normalized_jobs) == 1

        job = normalized_jobs[0]

        # Verify content hash
        expected_content = "AI EngineerJob descriptionTest Corp"
        expected_hash = hashlib.sha256(expected_content.encode()).hexdigest()
        assert job.content_hash == expected_hash

    @patch("src.scraper_company_pages.SessionLocal")
    @patch("src.scraper_company_pages.logger")
    def test_normalize_jobs_error_handling(
        self, mock_logger: Mock, mock_session_local: Mock
    ) -> None:
        """Test error handling during job normalization."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        # Mock database error during company lookup
        mock_session.exec.side_effect = Exception("Database error")

        state: State = {
            "companies": [],
            "partial_jobs": [],
            "raw_full_jobs": [
                {
                    "company": "Error Corp",
                    "title": "Job Title",
                    "description": "Description",
                    "url": "https://error.com/job1",
                }
            ],
            "normalized_jobs": [],
        }

        result = normalize_jobs(state)

        # Should return empty result on error
        assert result["normalized_jobs"] == []

        # Verify error was logged
        mock_logger.exception.assert_called_with(
            "Failed to normalize job %s", "https://error.com/job1"
        )


class TestGetNormalizedJobs:
    """Test suite for normalized job retrieval."""

    def test_get_normalized_jobs_success(self) -> None:
        """Test successful retrieval of normalized jobs."""
        mock_jobs = [
            Mock(spec=JobSQL),
            Mock(spec=JobSQL),
        ]

        state: State = {
            "companies": [],
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": mock_jobs,
        }

        result = get_normalized_jobs(state)

        assert result == {"normalized_jobs": mock_jobs}

    def test_get_normalized_jobs_empty_state(self) -> None:
        """Test handling empty state."""
        state: State = {
            "companies": [],
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        result = get_normalized_jobs(state)

        assert result == {"normalized_jobs": []}


class TestScrapeCompanyPages:
    """Test suite for the main company page scraping orchestration."""

    @patch("src.scraper_company_pages.StateGraph")
    @patch("src.scraper_company_pages.load_active_companies")
    @patch("src.scraper_company_pages.logger")
    def test_scrape_company_pages_success(
        self, mock_logger: Mock, mock_load_companies: Mock, mock_state_graph: Mock
    ) -> None:
        """Test successful company page scraping workflow."""
        # Mock active companies
        mock_companies = [
            CompanySQL(
                id=1, name="Corp 1", url="https://corp1.com/careers", active=True
            ),
            CompanySQL(
                id=2, name="Corp 2", url="https://corp2.com/careers", active=True
            ),
        ]
        mock_load_companies.return_value = mock_companies

        # Mock workflow execution
        mock_workflow = Mock()
        mock_graph = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = mock_graph

        mock_final_state = {
            "normalized_jobs": [
                Mock(spec=JobSQL),
                Mock(spec=JobSQL),
            ]
        }
        mock_graph.invoke.return_value = mock_final_state

        result = scrape_company_pages(max_jobs_per_company=25)

        # Verify result
        assert len(result) == 2

        # Verify workflow was set up correctly
        mock_workflow.add_node.assert_any_call("extract_lists", extract_job_lists)
        mock_workflow.add_node.assert_any_call("extract_details", extract_details)
        mock_workflow.add_node.assert_any_call("normalize", normalize_jobs)
        mock_workflow.add_node.assert_any_call("save", get_normalized_jobs)

        # Verify edges were added
        mock_workflow.add_edge.assert_any_call("extract_lists", "extract_details")
        mock_workflow.add_edge.assert_any_call("extract_details", "normalize")
        mock_workflow.add_edge.assert_any_call("normalize", "save")

        # Verify entry point was set
        mock_workflow.set_entry_point.assert_called_with("extract_lists")

        # Verify graph was invoked with correct state
        mock_graph.invoke.assert_called_once()
        invoke_args = mock_graph.invoke.call_args[0][0]
        assert invoke_args["companies"] == mock_companies
        assert invoke_args["max_jobs_per_company"] == 25

    @patch("src.scraper_company_pages.load_active_companies")
    @patch("src.scraper_company_pages.logger")
    def test_scrape_company_pages_no_active_companies(
        self, mock_logger: Mock, mock_load_companies: Mock
    ) -> None:
        """Test handling when no active companies exist."""
        mock_load_companies.return_value = []

        result = scrape_company_pages()

        assert result == []
        mock_logger.info.assert_called_with("No active companies to scrape.")

    @patch("src.scraper_company_pages.StateGraph")
    @patch("src.scraper_company_pages.load_active_companies")
    @patch("src.scraper_company_pages.logger")
    def test_scrape_company_pages_workflow_failure(
        self, mock_logger: Mock, mock_load_companies: Mock, mock_state_graph: Mock
    ) -> None:
        """Test handling workflow execution failures."""
        # Mock active companies
        mock_companies = [
            CompanySQL(id=1, name="Corp", url="https://corp.com/careers", active=True),
        ]
        mock_load_companies.return_value = mock_companies

        # Mock workflow to raise exception
        mock_workflow = Mock()
        mock_graph = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = mock_graph
        mock_graph.invoke.side_effect = Exception("Workflow failed")

        result = scrape_company_pages()

        assert result == []
        mock_logger.exception.assert_called_with("Workflow failed")

    @patch("src.scraper_company_pages.StateGraph")
    @patch("src.scraper_company_pages.load_active_companies")
    @patch("src.scraper_company_pages.settings")
    @patch("src.scraper_company_pages.SqliteSaver")
    def test_scrape_company_pages_with_checkpointing(
        self,
        mock_sqlite_saver: Mock,
        mock_settings: Mock,
        mock_load_companies: Mock,
        mock_state_graph: Mock,
    ) -> None:
        """Test workflow execution with checkpointing enabled."""
        # Enable checkpointing
        mock_settings.use_checkpointing = True

        mock_companies = [
            CompanySQL(id=1, name="Corp", url="https://corp.com/careers", active=True),
        ]
        mock_load_companies.return_value = mock_companies

        # Mock checkpointer
        mock_checkpointer = Mock()
        mock_sqlite_saver.from_conn_string.return_value = mock_checkpointer

        # Mock workflow
        mock_workflow = Mock()
        mock_graph = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = mock_graph
        mock_graph.invoke.return_value = {"normalized_jobs": []}

        scrape_company_pages()

        # Verify checkpointer was created and used
        mock_sqlite_saver.from_conn_string.assert_called_once_with("checkpoints.sqlite")
        mock_workflow.compile.assert_called_once_with(checkpointer=mock_checkpointer)

    @patch("src.scraper_company_pages.StateGraph")
    @patch("src.scraper_company_pages.load_active_companies")
    @patch("src.scraper_company_pages.settings")
    def test_scrape_company_pages_without_checkpointing(
        self, mock_settings: Mock, mock_load_companies: Mock, mock_state_graph: Mock
    ) -> None:
        """Test workflow execution without checkpointing."""
        # Disable checkpointing
        mock_settings.use_checkpointing = False

        mock_companies = [
            CompanySQL(id=1, name="Corp", url="https://corp.com/careers", active=True),
        ]
        mock_load_companies.return_value = mock_companies

        # Mock workflow
        mock_workflow = Mock()
        mock_graph = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = mock_graph
        mock_graph.invoke.return_value = {"normalized_jobs": []}

        scrape_company_pages()

        # Verify workflow was compiled without checkpointer
        mock_workflow.compile.assert_called_once_with(checkpointer=None)

    @patch("src.scraper_company_pages.StateGraph")
    @patch("src.scraper_company_pages.load_active_companies")
    def test_scrape_company_pages_default_max_jobs(
        self, mock_load_companies: Mock, mock_state_graph: Mock
    ) -> None:
        """Test default max_jobs_per_company value."""
        mock_companies = [
            CompanySQL(id=1, name="Corp", url="https://corp.com/careers", active=True),
        ]
        mock_load_companies.return_value = mock_companies

        # Mock workflow
        mock_workflow = Mock()
        mock_graph = Mock()
        mock_state_graph.return_value = mock_workflow
        mock_workflow.compile.return_value = mock_graph
        mock_graph.invoke.return_value = {"normalized_jobs": []}

        scrape_company_pages()  # No max_jobs_per_company parameter

        # Verify default value was used
        invoke_args = mock_graph.invoke.call_args[0][0]
        assert invoke_args["max_jobs_per_company"] == 50  # DEFAULT_MAX_JOBS_PER_COMPANY


class TestStateManagement:
    """Test suite for LangGraph state management."""

    def test_state_type_definition(self) -> None:
        """Test State TypedDict structure."""
        # Test that State can be instantiated with all required fields
        state: State = {
            "companies": [],
            "partial_jobs": [],
            "raw_full_jobs": [],
            "normalized_jobs": [],
        }

        assert isinstance(state["companies"], list)
        assert isinstance(state["partial_jobs"], list)
        assert isinstance(state["raw_full_jobs"], list)
        assert isinstance(state["normalized_jobs"], list)

    def test_state_with_data(self) -> None:
        """Test State with actual data structures."""
        company = CompanySQL(
            id=1, name="Test Corp", url="https://test.com", active=True
        )
        job = JobSQL(
            company_id=1,
            title="AI Engineer",
            description="AI role",
            link="https://test.com/job1",
            location="Remote",
            content_hash="hash123",
            application_status="New",
            last_seen=datetime.now(timezone.utc),
        )

        state: State = {
            "companies": [company],
            "partial_jobs": [
                {
                    "company": "Test Corp",
                    "title": "AI Engineer",
                    "url": "https://test.com/job1",
                }
            ],
            "raw_full_jobs": [
                {
                    "company": "Test Corp",
                    "title": "AI Engineer",
                    "description": "AI role",
                }
            ],
            "normalized_jobs": [job],
        }

        assert len(state["companies"]) == 1
        assert state["companies"][0].name == "Test Corp"
        assert len(state["normalized_jobs"]) == 1
        assert state["normalized_jobs"][0].title == "AI Engineer"
