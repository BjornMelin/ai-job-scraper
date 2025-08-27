# AI Job Scraper Test Suite Documentation

## Table of Contents

- [Overview](#overview)
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Pytest Markers](#pytest-markers)
- [Fixtures Reference](#fixtures-reference)
- [Test Data Generation](#test-data-generation)
- [Performance Benchmarking](#performance-benchmarking)
- [Best Practices](#best-practices)
- [Writing New Tests](#writing-new-tests)
- [CI/CD Integration](#cicd-integration)

## Overview

The AI Job Scraper test suite is a comprehensive, performance-focused testing framework that emphasizes:

- **Fast Feedback**: Efficient unit tests with parallel execution
- **Realistic Data**: Factory-based test data generation with Faker
- **Performance Monitoring**: Built-in benchmarking and regression detection
- **External API Mocking**: VCR-based HTTP recording/playback
- **Property-Based Testing**: Hypothesis-driven edge case discovery
- **CI/CD Ready**: Multiple execution modes for different environments

## Test Organization

```
tests/
├── analytics/          # Analytics and reporting tests
├── benchmarks/         # Performance benchmark tests
├── e2e/               # End-to-end workflow tests
├── integration/       # Integration tests (database, services)
├── performance/       # Performance-specific tests
├── property/          # Property-based tests (Hypothesis)
├── search/            # Search functionality tests
├── services/          # Service layer tests
├── ui/                # Streamlit UI component tests
├── unit/              # Fast unit tests
│   ├── core/          # Core functionality
│   ├── database/      # Database models and schemas
│   ├── models/        # Data model tests
│   ├── scraping/      # Web scraping tests
│   └── ui/            # UI utility tests
├── utils/             # Test utilities and helpers
├── fixtures/          # Reusable fixtures
├── cassettes/         # VCR HTTP recording files
├── conftest.py        # Global pytest configuration
└── factories.py       # Test data factories
```

## Running Tests

### Using the Test Runner Script

The project includes a comprehensive test runner (`scripts/run_tests.py`) with multiple execution modes:

```bash
# Fast unit tests only (development)
uv run python scripts/run_tests.py fast

# Unit tests with coverage
uv run python scripts/run_tests.py unit

# Integration tests
uv run python scripts/run_tests.py integration

# Performance/benchmark tests
uv run python scripts/run_tests.py performance

# Complete test suite with coverage
uv run python scripts/run_tests.py coverage

# Smoke tests (critical functionality)
uv run python scripts/run_tests.py smoke

# CI/CD mode (comprehensive)
uv run python scripts/run_tests.py ci

# All test categories
uv run python scripts/run_tests.py all

# Debug mode (detailed output)
uv run python scripts/run_tests.py debug
```

### Direct Pytest Commands

```bash
# Run specific test categories
uv run pytest -m unit                    # Unit tests only
uv run pytest -m "integration and not slow"  # Fast integration tests
uv run pytest -m "performance" --benchmark-enable  # Performance tests

# Run specific test files
uv run pytest tests/unit/core/test_config.py
uv run pytest tests/integration/test_workflow_integration.py

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Parallel execution (auto-detects CPU cores)
uv run pytest -n auto

# Specific test pattern
uv run pytest -k "test_search" -v
```

## Pytest Markers

The test suite uses comprehensive markers for fine-grained test selection:

### Core Categories
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Database and service integration tests
- `@pytest.mark.e2e` - End-to-end workflow tests

### Performance & Timing
- `@pytest.mark.slow` - Long-running tests (>5 seconds)
- `@pytest.mark.performance` - Performance monitoring tests
- `@pytest.mark.benchmark` - Benchmark tests (use `--benchmark-enable`)
- `@pytest.mark.fast` - Very fast tests (<100ms)

### Execution Modes
- `@pytest.mark.parallel_safe` - Safe for parallel execution
- `@pytest.mark.serial` - Must run serially (database locks, etc.)

### External Dependencies
- `@pytest.mark.http` - Tests making HTTP requests
- `@pytest.mark.vcr` - Tests using VCR cassettes
- `@pytest.mark.api` - API interaction tests
- `@pytest.mark.database` - Database access required

### UI & Components
- `@pytest.mark.ui` - UI component tests
- `@pytest.mark.streamlit` - Streamlit-specific tests

### Test Types
- `@pytest.mark.property` - Property-based tests (Hypothesis)
- `@pytest.mark.smoke` - Critical smoke tests
- `@pytest.mark.regression` - Regression tests
- `@pytest.mark.memory` - Memory usage monitoring
- `@pytest.mark.cpu` - CPU usage monitoring

### Example Usage

```python
import pytest

@pytest.mark.unit
@pytest.mark.fast
def test_config_defaults():
    """Fast unit test for configuration defaults."""
    pass

@pytest.mark.integration
@pytest.mark.database
@pytest.mark.slow
def test_database_migration():
    """Slow integration test requiring database."""
    pass

@pytest.mark.performance
@pytest.mark.benchmark
def test_search_performance(benchmark):
    """Performance benchmark test."""
    pass

@pytest.mark.property
@pytest.mark.unit
def test_salary_parsing_properties():
    """Property-based test for salary parsing."""
    pass
```

## Fixtures Reference

### Database Fixtures

#### Core Database
```python
def test_database_operation(session):
    """Test using transactional database session."""
    # session: Session - Auto-rollback database session
    pass

def test_with_isolated_db(isolated_test_db):
    """Test requiring completely fresh database."""
    # isolated_test_db: Engine - Fresh database engine
    pass
```

#### Parallel Execution
```python
def test_concurrent_access(engine, parallel_engine_pool):
    """Test for parallel execution scenarios."""
    # engine: Engine - Worker-specific database engine
    # parallel_engine_pool: Factory for creating isolated engines
    pass
```

### Test Data Fixtures

#### Sample Data
```python
def test_with_sample_data(sample_company, sample_job):
    """Test with single sample records."""
    # sample_company: CompanySQL
    # sample_job: JobSQL
    pass

def test_with_multiple_data(multiple_companies, multiple_jobs):
    """Test with multiple records."""
    # multiple_companies: list[CompanySQL] (5 companies)
    # multiple_jobs: list[JobSQL] (20 jobs)
    pass

def test_with_realistic_dataset(realistic_dataset):
    """Test with realistic data distribution."""
    # realistic_dataset: dict with companies and jobs
    pass
```

#### Factory Fixtures
```python
def test_custom_data(session):
    """Create custom test data using factories."""
    # Using factories directly
    company = CompanyFactory.create(session=session, active=True)
    jobs = JobFactory.create_batch(5, session=session, senior=True, remote=True)
```

### Configuration Fixtures

```python
def test_with_settings(test_settings):
    """Test with test configuration."""
    # test_settings: Settings - Test-specific configuration
    pass

def test_file_operations(test_data_dir, temp_dir):
    """Test file operations."""
    # test_data_dir: Path - Directory for test files
    # temp_dir: Path - Temporary directory (session-scoped)
    pass
```

### Performance Fixtures

```python
def test_performance_monitoring(performance_monitor):
    """Test with performance monitoring."""
    # performance_monitor: dict - CPU, memory, timing metrics
    pass

def test_memory_tracking(memory_tracker):
    """Test with memory usage tracking."""
    # memory_tracker: Memory diff analysis after test
    pass
```

### VCR (HTTP Recording) Fixtures

```python
def test_api_interaction(cassettes_cleanup):
    """Test API calls with VCR recording."""
    from tests.fixtures.vcr import api_vcr
    
    with api_vcr.use_cassette('api_test.yaml'):
        # HTTP calls will be recorded/replayed
        pass
```

## Test Data Generation

### Factory Boy Integration

The test suite uses Factory Boy with Faker for realistic test data:

```python
from tests.factories import CompanyFactory, JobFactory

# Create single instances
company = CompanyFactory.create(session=session)
job = JobFactory.create(session=session, company_id=company.id)

# Create with traits
senior_job = JobFactory.create(session=session, senior=True, remote=True)
inactive_company = CompanyFactory.create(session=session, inactive=True)

# Batch creation
companies = CompanyFactory.create_batch(10, session=session, established=True)
applied_jobs = JobFactory.create_batch(5, session=session, applied=True)
```

### Available Factory Traits

#### CompanyFactory Traits
- `inactive=True` - Inactive companies (active=False)
- `established=True` - High scrape counts and success rates

#### JobFactory Traits
- `senior=True` - Senior-level positions with higher salaries
- `junior=True` - Entry-level positions
- `remote=True` - Remote work location
- `favorited=True` - Favorited with notes and "Interested" status
- `applied=True` - Applied status with application date

### Dictionary Factories (No Database)

For tests not requiring database persistence:

```python
from tests.factories import CompanyDictFactory, JobDictFactory

company_data = CompanyDictFactory.build()
job_data = JobDictFactory.build()
```

### Helper Functions

```python
from tests.factories import create_sample_companies, create_sample_jobs

# Convenience functions
companies = create_sample_companies(session, count=5, established=True)
jobs = create_sample_jobs(session, count=10, company=companies[0], senior=True)
```

## Performance Benchmarking

### Benchmark Test Structure

```python
import pytest
from tests.benchmarks.conftest import benchmark_isolation

@pytest.mark.performance
@pytest.mark.benchmark
def test_database_insert_performance(benchmark, benchmark_dataset, warmup_session):
    """Benchmark database insert performance."""
    
    def insert_jobs():
        # Operation to benchmark
        return create_test_jobs(session, 100)
    
    with benchmark_isolation():
        result = benchmark(insert_jobs)
    
    # Assertions on performance
    assert benchmark.stats.mean < 0.5  # Under 500ms average
```

### Benchmark Scales

The benchmark suite supports multiple data scales:

- `micro` - 10 records (quick verification)
- `small` - 100 records (unit test scale)  
- `medium` - 1,000 records (integration scale)
- `large` - 10,000 records (performance testing)
- `xlarge` - 50,000 records (stress testing)

### Performance Thresholds

Configured thresholds for regression detection:

```python
PERFORMANCE_THRESHOLDS = {
    "database_creation_per_record": 0.01,  # 10ms per record max
    "search_response_time": 0.5,           # 500ms max
    "pagination_response_time": 0.1,       # 100ms max  
    "memory_growth_mb": 100,              # 100MB max growth
}
```

### Memory & Resource Monitoring

```python
@pytest.mark.memory
def test_memory_usage(memory_tracker, system_resource_monitor):
    """Test memory usage patterns."""
    
    # Perform memory-intensive operation
    large_data = create_large_dataset()
    
    # memory_tracker provides detailed memory diff analysis
    # system_resource_monitor provides CPU, memory, connections
```

### Benchmark Utilities

```python
def test_search_benchmark(search_benchmark_suite, session):
    """Benchmark search operations."""
    
    metrics = search_benchmark_suite["benchmark_fts_search"](
        session, "machine learning", iterations=100
    )
    
    assert metrics["avg_search_time"] < 0.1  # Under 100ms average
    assert metrics["searches_per_second"] > 10  # At least 10 QPS
```

## Best Practices

### Test Organization

1. **Use Descriptive Names**: Test functions should describe exactly what they test
```python
def test_job_creation_with_valid_salary_range():
    """Test job creation succeeds with valid salary range."""
```

2. **One Assertion Theme**: Each test should focus on one aspect
```python
# Good - focused test
def test_salary_parsing_handles_k_suffix():
    """Test salary parser correctly handles 'k' suffix."""
    
# Avoid - multiple unrelated assertions
def test_job_everything():
    """Test job creation, validation, and deletion."""  # Too broad
```

3. **Use Fixtures for Setup**: Don't repeat setup code
```python
def test_job_search(realistic_dataset, search_service):
    """Test uses fixtures for clean setup."""
```

### Performance Considerations

1. **Mark Slow Tests**: Help developers run fast feedback loops
```python
@pytest.mark.slow
@pytest.mark.integration
def test_full_scraping_workflow():
    """Mark tests that take >5 seconds."""
```

2. **Use Parallel-Safe Patterns**: Avoid shared state
```python
@pytest.mark.parallel_safe
def test_pure_function():
    """Pure functions are inherently parallel-safe."""
    
@pytest.mark.serial
def test_database_migration():
    """Database migrations must run serially."""
```

3. **Optimize Database Tests**: Use transactions, not recreation
```python
def test_with_rollback(session):
    """session fixture automatically rolls back changes."""
    # Changes are automatically rolled back
```

### Mocking Strategy

1. **Mock at Boundaries**: Mock external services, not internal logic
```python
from tests.fixtures.vcr import llm_vcr

def test_ai_extraction():
    with llm_vcr.use_cassette('extraction_test.yaml'):
        result = ai_client.extract_job_data(html_content)
        # Real AI call recorded/replayed
```

2. **Use VCR for HTTP**: Record real HTTP interactions
```python
def test_scraping_integration():
    with scraping_vcr.use_cassette('company_page.yaml'):
        jobs = scraper.scrape_company_page(company_url)
```

### Property-Based Testing

1. **Test Business Rules**: Use Hypothesis for edge case discovery
```python
@pytest.mark.property
@given(salary_range=salary_range_strategy())
def test_salary_range_properties(salary_range):
    """Property: min salary should never exceed max salary."""
    min_sal, max_sal = salary_range
    assert min_sal <= max_sal
```

2. **Use Custom Strategies**: Create realistic data generators
```python
@st.composite
def job_posting_strategy(draw):
    """Generate realistic job posting data."""
    return {
        "title": draw(st.sampled_from(AI_ML_TITLES)),
        "location": draw(st.sampled_from(TECH_LOCATIONS)),
        "posted_date": draw(date_range_strategy()),
    }
```

## Writing New Tests

### Unit Test Template

```python
"""Tests for [module/functionality]."""

import pytest
from unittest.mock import Mock, patch

from src.module import function_to_test

class TestFunctionName:
    """Test cases for function_to_test."""
    
    @pytest.mark.unit
    @pytest.mark.fast
    def test_happy_path(self):
        """Test function works with valid input."""
        result = function_to_test("valid_input")
        assert result == "expected_output"
    
    @pytest.mark.unit
    def test_edge_case(self):
        """Test function handles edge case properly."""
        result = function_to_test("")
        assert result is None
    
    @pytest.mark.unit
    def test_error_handling(self):
        """Test function raises appropriate error."""
        with pytest.raises(ValueError, match="Invalid input"):
            function_to_test(None)
```

### Integration Test Template

```python
"""Integration tests for [module/service]."""

import pytest

@pytest.mark.integration
@pytest.mark.database
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    def test_create_and_retrieve(self, session, sample_company):
        """Test creating and retrieving database records."""
        # Test with real database session
        job = JobFactory.create(session=session, company_id=sample_company.id)
        
        retrieved = session.get(JobSQL, job.id)
        assert retrieved.title == job.title
    
    @pytest.mark.slow
    def test_bulk_operations(self, session):
        """Test bulk database operations."""
        jobs = JobFactory.create_batch(100, session=session)
        assert len(jobs) == 100
```

### Performance Test Template

```python
"""Performance tests for [functionality]."""

import pytest

@pytest.mark.performance
@pytest.mark.benchmark
class TestPerformance:
    """Performance tests for functionality."""
    
    def test_operation_performance(self, benchmark, benchmark_dataset):
        """Test operation meets performance requirements."""
        
        def operation_to_test():
            return perform_operation(benchmark_dataset)
        
        with benchmark_isolation():
            result = benchmark(operation_to_test)
        
        # Assert performance requirements
        assert benchmark.stats.mean < 1.0  # Under 1 second
        assert len(result) == expected_count
    
    @pytest.mark.memory
    def test_memory_usage(self, memory_tracker, large_benchmark_dataset):
        """Test operation doesn't leak memory."""
        process_large_dataset(large_benchmark_dataset)
        
        # memory_tracker automatically analyzes memory usage
        # Test passes if no significant leaks detected
```

### Property-Based Test Template

```python
"""Property-based tests for [functionality]."""

import pytest
from hypothesis import given, strategies as st, assume

@pytest.mark.property
class TestProperties:
    """Property-based tests for functionality."""
    
    @given(
        input_data=st.text(min_size=1, max_size=100),
        config=st.integers(min_value=1, max_value=100)
    )
    def test_function_properties(self, input_data, config):
        """Test function maintains invariants for all inputs."""
        assume(len(input_data.strip()) > 0)  # Skip empty inputs
        
        result = function_under_test(input_data, config)
        
        # Property: result should always be valid
        assert result is not None
        assert len(result) >= len(input_data)  # Property example
```

## CI/CD Integration

### GitHub Actions Configuration

The test suite integrates with CI/CD pipelines through the test runner:

```yaml
# .github/workflows/test.yml
- name: Run Fast Tests
  run: uv run python scripts/run_tests.py smoke

- name: Run Full Test Suite
  run: uv run python scripts/run_tests.py ci

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

### Test Execution Modes

- **PR Checks**: `smoke` mode for quick validation
- **Main Branch**: `ci` mode for comprehensive testing
- **Nightly**: `all` mode including performance regressions
- **Release**: `all` + `benchmark` with performance analysis

### Environment Variables

```bash
# VCR Recording Control
VCR_RECORD=true              # Enable recording new cassettes
VCR_RECORD_MODE=once         # Record mode (once, new_episodes, all)

# Performance Monitoring  
STORE_BENCHMARK_METRICS=true # Save benchmark results
CLEAN_CASSETTES=true         # Clean up test cassettes

# Test Configuration
TESTING=true                 # Enable test mode
LOG_LEVEL=WARNING           # Reduce log noise in tests
```

### Performance Regression Detection

```bash
# Run benchmarks and compare against baseline
uv run python scripts/run_tests.py performance

# Store results for regression analysis
STORE_BENCHMARK_METRICS=true uv run pytest -m benchmark --benchmark-autosave
```

This comprehensive test suite ensures the AI Job Scraper maintains high quality, performance, and reliability while supporting rapid development cycles and confident deployments.