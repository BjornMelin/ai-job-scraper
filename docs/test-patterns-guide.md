# Test Patterns & Best Practices Guide

> **Library-First Testing**: Comprehensive guide to testing patterns using modern libraries, minimizing custom code and maximizing maintainability.

## Testing Library Decision Matrix

### When to Use Each Testing Library

| Scenario | Library | Version | Decision Criteria | Alternative Considered |
|----------|---------|---------|-------------------|------------------------|
| **Test Data Generation** | factory-boy | 3.3.3 | Realistic data, SQLModel integration | Manual fixtures (higher maintenance) |
| **HTTP API Mocking** | responses | 0.25.8 | Simple decorator API | httpx-mock (more complex setup) |
| **Property-Based Testing** | hypothesis | 6.138.3 | Edge case automation | Manual edge cases (incomplete coverage) |
| **Parallel Execution** | pytest-xdist | 3.8.0 | Work-stealing load balancing | pytest-parallel (less mature) |
| **Data Generation** | faker | 25.9.2 | Domain-specific data | Random module (unrealistic data) |
| **HTTP Recording** | vcrpy | 7.0.0 | Reproducible external tests | Live API calls (unreliable) |

## Core Testing Patterns

### 1. Factory-Based Test Data Generation

**Library-First Approach**: Use factory-boy with SQLModel for realistic test data

```python
# tests/factories.py - Production pattern
from factory.alchemy import SQLAlchemyModelFactory
from factory import Faker, fuzzy, SubFactory, Trait

class JobFactory(SQLAlchemyModelFactory):
    """Generate realistic AI/ML job data."""
    
    class Meta:
        model = JobSQL
        sqlalchemy_session_persistence = "commit"
    
    # Realistic AI/ML job titles
    title = fuzzy.FuzzyChoice([
        "Senior AI Engineer", "Machine Learning Engineer",
        "Data Scientist", "MLOps Engineer"
    ])
    
    # AI/ML salary ranges
    salary = LazyFunction(lambda: _generate_realistic_salary())
    location = fuzzy.FuzzyChoice(["San Francisco, CA", "Remote", "New York, NY"])
    
    # Traits for different scenarios
    class Params:
        senior = Trait(
            title="Principal AI Engineer",
            salary=LazyFunction(lambda: (180_000, 350_000))
        )
        remote = Trait(location="Remote")
        favorited = Trait(favorite=True, notes="Interesting opportunity")

# Usage in tests
def test_job_filtering(test_session):
    # Create 10 senior remote jobs
    jobs = JobFactory.create_batch(10, senior=True, remote=True)
    
    result = filter_jobs(location="Remote", seniority="senior")
    assert len(result) == 10
```

### 2. HTTP Service Testing

**responses Library Pattern**: Mock external APIs with decorator pattern

```python
# tests/unit/scraping/test_scrapers.py
import responses
from src.scraper_job_boards import scrape_indeed_jobs

@responses.activate
def test_indeed_api_success():
    """Test successful Indeed API response handling."""
    # Mock the external API call
    responses.add(
        responses.GET,
        "https://api.indeed.com/v1/jobs/search",
        json={
            "jobs": [
                {"title": "AI Engineer", "company": "TechCorp", "location": "SF"}
            ]
        },
        status=200
    )
    
    # Test internal logic without external dependency
    result = scrape_indeed_jobs("AI Engineer", location="San Francisco")
    
    assert len(result) == 1
    assert result[0]["title"] == "AI Engineer"

@responses.activate  
def test_indeed_api_failure():
    """Test API failure handling."""
    responses.add(
        responses.GET,
        "https://api.indeed.com/v1/jobs/search", 
        json={"error": "Rate limited"},
        status=429
    )
    
    # Should handle gracefully
    result = scrape_indeed_jobs("AI Engineer")
    assert result == []  # Graceful degradation
```

### 3. VCR Pattern for External Services

**vcrpy Integration**: Record/replay HTTP interactions

```python
# tests/integration/test_scraping_workflow.py
import pytest
import vcr

# Configure VCR with filters
my_vcr = vcr.VCR(
    serializer='yaml',
    cassette_library_dir='tests/cassettes/scraping',
    record_mode='once',  # Record once, then replay
    filter_headers=['authorization', 'x-api-key'],  # Remove sensitive headers
    filter_query_parameters=['api_key']
)

@pytest.mark.vcr
def test_linkedin_job_scraping():
    """Test LinkedIn scraping with recorded HTTP interactions."""
    with my_vcr.use_cassette('linkedin_ai_jobs.yaml'):
        result = scrape_linkedin_jobs("AI Engineer", pages=2)
        
        assert len(result) > 0
        assert all("title" in job for job in result)
        assert all("company" in job for job in result)
```

### 4. Property-Based Testing with Hypothesis

**Edge Case Automation**: Use hypothesis for comprehensive edge case testing

```python
# tests/property/test_validation_properties.py
from hypothesis import given, strategies as st
from src.core_utils import parse_salary_range

@given(st.text())
def test_salary_parsing_never_crashes(salary_text):
    """Salary parsing should never crash on any string input."""
    # Should not raise exceptions on any input
    result = parse_salary_range(salary_text)
    assert result is None or isinstance(result, tuple)

@given(st.integers(min_value=0, max_value=1_000_000))  
def test_salary_validation_properties(salary):
    """Test salary validation properties."""
    from src.ui.utils.validators import validate_salary
    
    result = validate_salary(salary)
    
    if result is True:
        # Valid salaries should be reasonable
        assert 30_000 <= salary <= 500_000
    else:
        # Invalid salaries should be outside reasonable range
        assert salary < 30_000 or salary > 500_000

@given(st.lists(st.dictionaries(
    keys=st.sampled_from(['title', 'company', 'salary']),
    values=st.text(min_size=1, max_size=100)
), min_size=0, max_size=1000))
def test_job_filtering_properties(job_list):
    """Job filtering should maintain invariants."""
    from src.services.job_service import filter_jobs
    
    result = filter_jobs(job_list, title_contains="Engineer")
    
    # Properties that must always hold
    assert isinstance(result, list)
    assert len(result) <= len(job_list)  # Never return more than input
    # All results should contain "Engineer" if filtering worked
    if result:
        assert all("Engineer" in job.get("title", "") for job in result)
```

### 5. Streamlit Component Testing

**Session State Mocking**: Test Streamlit components in isolation

```python
# tests/unit/ui/test_job_page.py
from unittest.mock import patch, MagicMock
import pytest
from src.ui.pages.jobs import jobs_page_main

@pytest.fixture
def mock_streamlit_session():
    """Mock Streamlit session state."""
    return {
        'current_company_id': None,
        'selected_jobs': [],
        'search_filters': {},
        'page_size': 25,
        'current_page': 1
    }

def test_jobs_page_renders_without_data(mock_streamlit_session):
    """Test jobs page handles empty state gracefully."""
    with patch('streamlit.session_state', mock_streamlit_session), \
         patch('src.services.job_service.get_jobs', return_value=[]):
        
        # Should not raise exceptions
        jobs_page_main()
        
        # Verify empty state handling
        assert mock_streamlit_session['selected_jobs'] == []

def test_jobs_page_with_data(mock_streamlit_session, sample_jobs):
    """Test jobs page with sample job data."""
    with patch('streamlit.session_state', mock_streamlit_session), \
         patch('src.services.job_service.get_jobs', return_value=sample_jobs):
        
        jobs_page_main()
        
        # Verify data processing
        assert isinstance(mock_streamlit_session.get('search_filters'), dict)

# Integration test with realistic data
def test_job_filtering_integration(test_session):
    """Test job filtering with factory-generated data."""
    # Create realistic test data
    jobs = JobFactory.create_batch(50, 
        senior=True,      # 50% senior positions
        remote=True       # 50% remote positions
    )
    
    # Test filtering combinations
    remote_jobs = filter_jobs(location="Remote")
    senior_jobs = filter_jobs(seniority="senior")
    
    assert len(remote_jobs) > 0
    assert len(senior_jobs) > 0
```

### 6. Database Testing Patterns

**Session Management**: Isolated tests with factory data

```python
# tests/unit/database/test_models.py
def test_job_creation_with_factory(test_session):
    """Test job creation with realistic factory data."""
    # Factory handles all the realistic data generation
    job = JobFactory.create(
        title="Senior AI Engineer",
        salary=(150_000, 250_000),
        location="San Francisco, CA"
    )
    
    # Test database constraints and relationships
    assert job.id is not None
    assert job.company_id is not None
    assert job.title == "Senior AI Engineer"
    assert job.salary == (150_000, 250_000)
    
    # Test relationships are properly configured
    assert job.company is not None
    assert job.company.id == job.company_id

def test_job_search_performance(test_session):
    """Test job search performance with large dataset."""
    # Create realistic dataset
    dataset = create_realistic_dataset(
        session=test_session,
        companies=20,
        jobs_per_company=50,  # 1,000 jobs total
        senior_ratio=0.3,
        remote_ratio=0.4
    )
    
    # Test search performance
    import time
    start_time = time.time()
    results = search_jobs(query="AI Engineer", location="Remote")
    execution_time = time.time() - start_time
    
    assert len(results) > 0
    assert execution_time < 0.5  # Performance requirement
```

## Testing Anti-Patterns to Avoid

### ❌ What NOT to Test

**Over-Testing (YAGNI Violations)**:

```python
# DON'T TEST - Property getters/setters
def test_job_title_getter():
    job = Job(title="Engineer")
    assert job.title == "Engineer"  # Trivial, no business logic

# DON'T TEST - Framework internals  
def test_streamlit_rendering():
    st.write("Hello")
    # Testing Streamlit itself, not our logic

# DON'T TEST - Library functionality
def test_pandas_dataframe_creation():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert len(df) == 3  # Testing pandas, not our code
```

**Brittle Tests (DRY Violations)**:

```python
# DON'T - Hard-coded IDs and data
def test_job_lookup():
    job = get_job_by_id(12345)  # Brittle - ID might not exist
    assert job.title == "Exact Title Match"  # Brittle string matching

# DON'T - Testing implementation details
def test_internal_caching():
    service = JobService()
    service._cache = {}  # Testing private implementation
    # This will break when refactoring
```

### ✅ What TO Test

**Business Logic & Edge Cases**:

```python
# DO TEST - Business rules
def test_salary_range_validation():
    assert validate_salary_range(50_000, 100_000) == True
    assert validate_salary_range(100_000, 50_000) == False  # Min > Max
    
# DO TEST - Error handling
def test_api_failure_graceful_degradation():
    with patch('requests.get', side_effect=ConnectionError):
        result = scrape_jobs("AI Engineer")
        assert result == []  # Graceful failure
        
# DO TEST - Integration workflows
def test_complete_job_application_flow():
    job = JobFactory.create()
    application = apply_to_job(job.id, user_profile)
    assert application.status == "submitted"
```

## Performance Testing Patterns

### Benchmark Integration

```python
# tests/performance/test_search_performance.py
import pytest

@pytest.mark.benchmark
def test_job_search_performance(benchmark):
    """Benchmark job search performance."""
    # Setup realistic data
    jobs = JobFactory.create_batch(10_000)
    
    # Benchmark the search function
    result = benchmark(search_jobs, query="AI Engineer", limit=100)
    
    assert len(result) <= 100
    # Benchmark will automatically measure execution time

@pytest.mark.performance  
def test_database_query_performance(test_session):
    """Test database query performance with large dataset."""
    # Create 10,000 jobs across 100 companies
    create_realistic_dataset(test_session, companies=100, jobs_per_company=100)
    
    import time
    start = time.time()
    
    # Complex query that might be slow
    results = (test_session.query(JobSQL)
               .join(CompanySQL)
               .filter(JobSQL.salary[0] > 100_000)
               .filter(CompanySQL.active == True)
               .all())
    
    duration = time.time() - start
    
    assert duration < 0.1  # Performance requirement: 100ms
    assert len(results) > 0
```

## Mock Strategy Guidelines

### When to Mock

```python
# Mock external services (network calls)
@responses.activate
def test_external_api():
    responses.add(responses.GET, "https://api.external.com/jobs")
    # Test our logic, not the external service

# Mock slow operations (database in unit tests)  
def test_job_processing(mock_database):
    with patch('src.database.get_session', return_value=mock_database):
        # Fast unit test without real database
```

### When NOT to Mock

```python  
# Don't mock internal business logic
def test_salary_calculation():
    # Test the real calculation, don't mock it
    result = calculate_total_compensation(base=100_000, equity=50_000)
    assert result == 150_000

# Don't mock simple data structures
def test_job_filtering():
    jobs = [JobFactory.build() for _ in range(10)]  # Real objects
    # Test real filtering logic
```

### Mock Count Monitoring

```python
# Track mock usage to prevent over-mocking
def test_mock_count_monitoring():
    """Ensure mock count stays reasonable."""
    # Current suite has ~189 mocks across 1,261 tests
    # Target: <200 mocks (15% of tests maximum)
    
    mock_count = count_mocks_in_test_suite()
    test_count = count_total_tests()
    
    mock_ratio = mock_count / test_count
    assert mock_ratio < 0.20  # Less than 20% of tests should use mocks
```

## Library Configuration Best Practices

### pytest Configuration Optimization

```toml
# pyproject.toml - Production-optimized settings
[tool.pytest.ini_options]
# Performance optimizations
addopts = [
    "-n=auto",                    # Parallel execution (12 workers)
    "--dist=worksteal",          # Efficient load balancing
    "--benchmark-disable",       # Disable benchmarks by default
    "--strict-markers",          # Enforce marker definitions
    "--tb=short",               # Concise tracebacks
]

# Test discovery optimization
python_files = ["test_*.py", "*_test.py"]
python_functions = ["test_*"]
testpaths = ["tests"]
norecursedirs = ["src"]  # Don't scan source for tests
```

### Factory Configuration

```python
# tests/conftest.py - Session management
@pytest.fixture(autouse=True)
def configure_factories(test_session):
    """Configure all factories with test session."""
    # Set session for all factories automatically
    for factory_class in [CompanyFactory, JobFactory]:
        factory_class._meta.sqlalchemy_session = test_session
```

### Coverage Configuration  

```toml
[tool.coverage.run]
source = ["src"]
branch = true  # Enable branch coverage
relative_files = true  # Portable coverage paths

[tool.coverage.report] 
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

---

**Next**: Maintenance & Operations Guide - CI/CD integration, performance monitoring, and long-term maintainability patterns.
