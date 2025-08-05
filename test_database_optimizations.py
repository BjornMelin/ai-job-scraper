#!/usr/bin/env python3
"""Test script to validate database optimizations after library-first changes.

This script tests the enhanced database utilities, session management, and
performance optimizations to ensure they work correctly with the simplified
background task system and Streamlit integration.
"""

import logging
import time

from concurrent.futures import ThreadPoolExecutor

import pytest

from src.database import create_db_and_tables, get_connection_pool_status
from src.models import CompanySQL
from src.services.company_service import CompanyService
from src.services.job_service import JobService
from src.ui.utils.database_utils import (
    background_task_session,
    clean_session_state,
    get_database_health,
    streamlit_db_session,
    validate_session_state,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Initialize database for all tests."""
    create_db_and_tables()


def test_database_health_monitoring():
    """Test database health monitoring functionality."""
    logger.info("Testing database health monitoring...")

    # Test connection pool status
    pool_status = get_connection_pool_status()
    assert isinstance(pool_status, dict)
    assert "pool_size" in pool_status
    logger.info("Connection pool status: %s", pool_status)

    # Test health assessment
    health = get_database_health()
    assert isinstance(health, dict)
    assert "health" in health
    logger.info("Database health: %s", health)

    print("✅ Database health monitoring tests passed")


def test_streamlit_session_management():
    """Test Streamlit-optimized session management."""
    logger.info("Testing Streamlit session management...")

    # Test streamlit_db_session context manager
    with streamlit_db_session() as session:
        # Create a test company
        test_company = CompanySQL(
            name="Test Company Session", url="https://test.com", active=True
        )
        session.add(test_company)
        # Session should auto-commit on exit

    # Verify the company was created
    companies = CompanyService.get_all_companies()
    test_companies = [c for c in companies if c.name == "Test Company Session"]
    assert test_companies

    print("✅ Streamlit session management tests passed")


def test_background_task_session():
    """Test background task session management."""
    logger.info("Testing background task session management...")

    def background_db_operation():
        """Simulate a database operation in a background thread."""
        with background_task_session() as session:
            # Create a test company
            test_company = CompanySQL(
                name="Background Test Company",
                url="https://background-test.com",
                active=True,
            )
            session.add(test_company)
            # Session should auto-commit on exit
        return "Background operation completed"

    # Run background operation in a thread
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(background_db_operation) for _ in range(2)]

        results = [future.result() for future in futures]
        assert all("completed" in result for result in results)

    # Verify companies were created
    companies = CompanyService.get_all_companies()
    bg_companies = [c for c in companies if "Background Test" in c.name]
    assert len(bg_companies) >= 2

    print("✅ Background task session tests passed")


def test_session_state_validation():
    """Test session state validation functionality."""
    logger.info("Testing session state validation...")

    # Note: This would normally work with actual st.session_state
    # For testing purposes, we'll just verify the functions don't error
    contaminated_keys = validate_session_state()
    assert isinstance(contaminated_keys, list)

    cleaned_count = clean_session_state()
    assert isinstance(cleaned_count, int)

    print("✅ Session state validation tests passed")


def test_concurrent_database_access():
    """Test concurrent database access patterns."""
    logger.info("Testing concurrent database access...")

    def concurrent_job_query():
        """Simulate concurrent job queries."""
        filters = {
            "text_search": "",
            "company": "All",
            "application_status": "All",
            "date_from": None,
            "date_to": None,
            "favorites_only": False,
        }
        jobs = JobService.get_filtered_jobs(filters)
        return len(jobs)

    # Run multiple concurrent queries
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(concurrent_job_query) for _ in range(10)]

        results = [future.result() for future in futures]

    end_time = time.time()

    logger.info("10 concurrent queries completed in %.2fs", end_time - start_time)
    logger.info("Query results: %s", results)

    # All queries should succeed
    assert all(isinstance(result, int) for result in results)

    print("✅ Concurrent database access tests passed")


def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    logger.info("Testing performance monitoring...")

    # Create some test data to measure performance
    start_time = time.time()

    # Test bulk company creation through service
    for i in range(10):
        CompanyService.add_company(f"Perf Test Company {i}", f"https://test{i}.com")

    creation_time = time.time() - start_time
    logger.info("Created 10 companies in %.3fs", creation_time)

    # Test bulk job queries
    start_time = time.time()
    filters = {"text_search": "", "company": "All", "application_status": "All"}
    jobs = JobService.get_filtered_jobs(filters)
    query_time = time.time() - start_time

    logger.info("Queried %d jobs in %.3fs", len(jobs), query_time)

    # Performance should be reasonable
    assert creation_time < 5.0  # Should create 10 companies in under 5 seconds
    assert query_time < 2.0  # Should query jobs in under 2 seconds

    print("✅ Performance monitoring tests passed")
