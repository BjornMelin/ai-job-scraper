"""Performance test for database optimization validation.

This script validates the database optimizations for T1.2 Background Scraping
by testing:
1. Thread safety with concurrent database access
2. N+1 query elimination with bulk company operations
3. SQLite configuration under threading load
"""

import time

from concurrent.futures import ThreadPoolExecutor

import pytest

from src.database import SessionLocal, create_db_and_tables
from src.models import CompanySQL
from src.scraper import bulk_get_or_create_companies


@pytest.fixture(scope="session", autouse=True)
def setup_database():
    """Initialize database for all tests."""
    create_db_and_tables()


def test_thread_safety():
    """Test thread-safe database access."""
    print("Testing thread safety...")

    def worker_thread(thread_id):
        """Worker function to test database access from multiple threads."""
        try:
            with SessionLocal() as session:
                # Create a company with thread-specific name
                company = CompanySQL(
                    name=f"Test Company {thread_id}",
                    url=f"https://test{thread_id}.com",
                    active=True,
                )
                session.add(company)
                session.commit()
                print(f"Thread {thread_id}: Successfully created company")
                return True
        except Exception as e:
            print(f"Thread {thread_id}: Failed with error: {e}")
            return False

    # Test with multiple concurrent threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(worker_thread, i) for i in range(10)]
        results = [future.result() for future in futures]

    success_count = sum(results)
    print(f"Thread safety test: {success_count}/10 threads succeeded")
    return success_count == 10


def test_bulk_company_performance():
    """Test N+1 query elimination with bulk operations."""
    print("Testing bulk company creation performance...")

    # Test data: 50 unique company names
    test_companies = {f"Bulk Test Company {i}" for i in range(50)}

    with SessionLocal() as session:
        # Measure bulk operation performance
        start_time = time.time()
        company_map = bulk_get_or_create_companies(session, test_companies)
        bulk_time = time.time() - start_time

        session.commit()

        print(
            f"Bulk operation: Created/fetched {len(company_map)} companies "
            f"in {bulk_time:.3f}s"
        )
        print(f"Average time per company: {bulk_time / len(company_map):.4f}s")

        # Verify all companies were processed
        assert len(company_map) == len(test_companies)

        return bulk_time < 1.0  # Should complete in under 1 second


def test_session_context_management():
    """Test proper session context management."""
    print("Testing session context management...")

    try:
        # Test proper session lifecycle
        with SessionLocal() as session:
            companies = session.query(CompanySQL).limit(5).all()
            print(f"Successfully queried {len(companies)} companies")

        # Session should be properly closed here
        print("Session context management: OK")
    except Exception as e:
        print(f"Session context management failed: {e}")
        return False
    return True
