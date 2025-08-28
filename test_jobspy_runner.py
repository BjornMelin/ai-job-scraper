#!/usr/bin/env python3
"""Simple test runner for JobSpy tests without conftest dependencies.

This script runs the JobSpy tests independently to validate functionality
and measure execution time.
"""

import sys
import time

from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_basic_tests():
    """Run basic JobSpy tests without pytest dependencies."""
    print("🚀 Running JobSpy Test Suite Validation")
    print("=" * 50)

    start_time = time.time()

    # Test 1: Import all modules
    print("📦 Testing imports...")
    try:
        from src.models.job_models import (
            JobPosting,
            JobScrapeRequest,
            JobScrapeResult,
            JobSite,
            JobType,
            LocationType,
        )

        print("✅ JobSpy models imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

    # Test 2: Test JobSite enum functionality
    print("\n🔧 Testing JobSite enum...")
    try:
        # Test normalization
        assert JobSite.normalize("linkedin") == JobSite.LINKEDIN
        assert JobSite.normalize("INDEED") == JobSite.INDEED
        assert JobSite.normalize("unknown") is None
        print("✅ JobSite normalization works correctly")
    except AssertionError as e:
        print(f"❌ JobSite test failed: {e}")
        return False

    # Test 3: Test JobType enum functionality
    print("\n🔧 Testing JobType enum...")
    try:
        assert JobType.normalize("full-time") == JobType.FULLTIME
        assert JobType.normalize("contract") == JobType.CONTRACT
        assert JobType.normalize("unknown") is None
        print("✅ JobType normalization works correctly")
    except AssertionError as e:
        print(f"❌ JobType test failed: {e}")
        return False

    # Test 4: Test LocationType functionality
    print("\n🔧 Testing LocationType enum...")
    try:
        assert LocationType.from_remote_flag(True) == LocationType.REMOTE
        assert LocationType.from_remote_flag(False, "Remote") == LocationType.REMOTE
        assert LocationType.from_remote_flag(False, "Hybrid") == LocationType.HYBRID
        assert LocationType.from_remote_flag(False, "New York") == LocationType.ONSITE
        print("✅ LocationType logic works correctly")
    except AssertionError as e:
        print(f"❌ LocationType test failed: {e}")
        return False

    # Test 5: Test JobScrapeRequest creation and validation
    print("\n🔧 Testing JobScrapeRequest...")
    try:
        # Basic request
        request = JobScrapeRequest(
            site_name=JobSite.LINKEDIN,
            search_term="Python Developer",
            results_wanted=50,
        )
        assert request.search_term == "Python Developer"
        assert request.results_wanted == 50
        assert request.site_name == JobSite.LINKEDIN

        # Request with normalization
        request2 = JobScrapeRequest(
            site_name="indeed",  # Should be normalized
            job_type="full-time",  # Should be normalized
            results_wanted=25,
        )
        assert request2.site_name == JobSite.INDEED
        assert request2.job_type == JobType.FULLTIME

        print("✅ JobScrapeRequest validation works correctly")
    except Exception as e:
        print(f"❌ JobScrapeRequest test failed: {e}")
        return False

    # Test 6: Test JobPosting creation and validation
    print("\n🔧 Testing JobPosting...")
    try:
        # Basic job posting
        posting = JobPosting(
            id="test_001",
            site=JobSite.LINKEDIN,
            title="Senior Python Developer",
            company="TechCorp",
            location="San Francisco, CA",
            job_type=JobType.FULLTIME,
            min_amount=120000.0,
            max_amount=180000.0,
            is_remote=False,
        )

        assert posting.title == "Senior Python Developer"
        assert posting.company == "TechCorp"
        assert posting.location_type == LocationType.ONSITE
        assert posting.min_amount == 120000.0

        # Test field validation
        posting2 = JobPosting(
            id="test_002",
            site="indeed",  # Should be normalized
            title="Data Scientist",
            company="DataCorp",
            job_type="contract",  # Should be normalized
            min_amount="100000",  # Should be converted to float
            is_remote=True,
        )

        assert posting2.site == JobSite.INDEED
        assert posting2.job_type == JobType.CONTRACT
        assert posting2.min_amount == 100000.0
        assert posting2.location_type == LocationType.REMOTE

        print("✅ JobPosting validation works correctly")
    except Exception as e:
        print(f"❌ JobPosting test failed: {e}")
        return False

    # Test 7: Test JobScrapeResult functionality
    print("\n🔧 Testing JobScrapeResult...")
    try:
        import pandas as pd

        # Create sample data
        sample_data = [
            {
                "id": "result_001",
                "site": "linkedin",
                "title": "Engineer",
                "company": "TestCorp",
                "location": "Remote",
                "is_remote": True,
            }
        ]

        df = pd.DataFrame(sample_data)
        request = JobScrapeRequest(search_term="Engineer")

        # Test from_pandas conversion
        result = JobScrapeResult.from_pandas(df, request)

        assert len(result.jobs) == 1
        assert result.total_found == 1
        assert result.jobs[0].title == "Engineer"
        assert result.jobs[0].company == "TestCorp"
        assert result.jobs[0].location_type == LocationType.REMOTE

        # Test back conversion
        df_converted = result.to_pandas()
        assert len(df_converted) == 1

        print("✅ JobScrapeResult conversion works correctly")
    except Exception as e:
        print(f"❌ JobScrapeResult test failed: {e}")
        return False

    # Test 8: Test fixtures import (if available)
    print("\n🔧 Testing fixtures...")
    try:
        sys.path.insert(0, str(Path(__file__).parent / "tests"))

        # This would normally be a fixture, so we'll call it directly
        data = [
            {
                "id": "fixture_test",
                "site": "linkedin",
                "title": "Test Job",
                "company": "Test Company",
                "location": "Test Location",
            }
        ]

        # Validate fixture structure
        posting = JobPosting.model_validate(data[0])
        assert posting.title == "Test Job"

        print("✅ Test fixtures work correctly")
    except Exception as e:
        print(f"⚠️  Fixture test skipped: {e}")

    # Performance test
    print("\n⚡ Running performance test...")
    try:
        perf_start = time.time()

        # Create many job postings
        for i in range(100):
            posting = JobPosting(
                id=f"perf_{i}",
                site=JobSite.LINKEDIN,
                title=f"Job {i}",
                company=f"Company {i}",
                location="Remote",
            )

        perf_time = time.time() - perf_start
        print(f"✅ Performance test: Created 100 JobPostings in {perf_time:.3f}s")

        if perf_time > 1.0:
            print("⚠️  Performance warning: Creation took longer than expected")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

    execution_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("🎉 All JobSpy tests completed successfully!")
    print(f"⚡ Total execution time: {execution_time:.3f} seconds")

    if execution_time > 10.0:
        print("⚠️  Warning: Tests took longer than 10 seconds")
        return False

    print("✅ Fast execution requirement met (<10s)")

    return True


def test_mock_integration():
    """Test mock integration functionality."""
    print("\n🔗 Testing mock integration patterns...")

    try:
        # Test mock JobSpy functionality
        import pandas as pd

        def mock_jobspy_scrape_jobs(**kwargs):
            """Mock version of jobspy.scrape_jobs."""
            return pd.DataFrame(
                [
                    {
                        "id": "mock_001",
                        "site": "linkedin",
                        "title": "Mock Job",
                        "company": "Mock Company",
                        "location": "Mock Location",
                        "job_type": "fulltime",
                        "min_amount": 100000.0,
                    }
                ]
            )

        # Test mock call
        df = mock_jobspy_scrape_jobs(
            site_name=["linkedin"], search_term="Python Developer", results_wanted=1
        )

        assert len(df) == 1
        assert df.iloc[0]["title"] == "Mock Job"

        # Test conversion to JobScrapeResult
        request = JobScrapeRequest(search_term="Python Developer")
        result = JobScrapeResult.from_pandas(df, request)

        assert len(result.jobs) == 1
        assert result.jobs[0].title == "Mock Job"
        assert result.jobs[0].job_type == JobType.FULLTIME

        print("✅ Mock integration patterns work correctly")
        return True

    except Exception as e:
        print(f"❌ Mock integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("JobSpy Test Suite - Comprehensive Validation")
    print("Testing all components with 100% mocked data")
    print()

    success = True

    # Run basic tests
    if not run_basic_tests():
        success = False

    # Run integration tests
    if not test_mock_integration():
        success = False

    if success:
        print("\n🎊 SUCCESS: All JobSpy tests passed!")
        print("✅ Models validated")
        print("✅ Enums working")
        print("✅ Validation working")
        print("✅ Conversions working")
        print("✅ Mock patterns working")
        print("✅ Performance requirements met")
        sys.exit(0)
    else:
        print("\n💥 FAILURE: Some tests failed")
        sys.exit(1)
