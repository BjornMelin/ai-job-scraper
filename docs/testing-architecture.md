# Test Architecture Documentation

> **Technical Overview**: Production-ready test suite architecture with parallel execution, factory-based test data, and library-first patterns for maintainability.

## Architecture Overview

### Current Metrics (Evidence-Based)

- **Test Suite Size**: 1,261 tests across 78 files
- **Coverage**: 26.03% line coverage (1,156/4,441 lines), 14.78% branch coverage
- **Execution Performance**: ~8.7s with 12 parallel workers (pytest-xdist)
- **Target**: 80% line coverage, <60s full suite execution

### Library Stack (Production Versions)

| Library | Version | Purpose | Configuration |
|---------|---------|---------|---------------|
| **pytest** | 8.4.1 | Test framework | Core execution engine |
| **pytest-xdist** | 3.8.0 | Parallel execution | 12 workers, worksteal distribution |
| **factory-boy** | 3.3.3 | Test data generation | SQLModel integration, realistic data |
| **faker** | 25.9.2 | Data generation | Seeded (42) for reproducibility |
| **hypothesis** | 6.138.3 | Property-based testing | Edge case validation |
| **responses** | 0.25.8 | HTTP mocking | External service isolation |
| **pytest-cov** | 6.2.1 | Coverage measurement | HTML/XML/terminal reports |

## Performance Benchmarks (Measured)

### Current Performance

```bash
# Parallel execution (12 workers)
Test Collection: 1,261 tests in 3.77s
Parallel Execution: ~8.7s total runtime
CPU Utilization: 893% (8.9x cores effectively used)
Memory Usage: Optimized for SQLite in-memory per worker
```

### Performance Targets

```bash
# Production targets
Full Suite: <60s (currently at 8.7s ✅)  
Unit Tests: <15s (subset execution)
Integration Tests: <30s (database workflows)
Coverage Collection: +2s overhead (acceptable)
```

## Test Architecture Patterns

### Factory Pattern Implementation

The test suite leverages **factory-boy 3.3.3** with SQLModel for realistic test data generation:

```python
# tests/factories.py - Production patterns
from factory.alchemy import SQLAlchemyModelFactory
from factory import Faker, fuzzy, LazyFunction

class CompanyFactory(SQLAlchemyModelFactory):
    """Factory for realistic tech company data."""
    
    class Meta:
        model = CompanySQL
        sqlalchemy_session_persistence = "commit"
        # Injected per test session
        sqlalchemy_session = None
    
    # Realistic AI/ML company data
    name = Faker("company")
    url = Faker("url", schemes=["https"])
    active = fuzzy.FuzzyChoice([True, True, True, False])  # 75% active
    success_rate = fuzzy.FuzzyFloat(0.5, 1.0)
    
    # Traits for different scenarios
    class Params:
        inactive = factory.Trait(active=False, scrape_count=0)
        established = factory.Trait(
            scrape_count=fuzzy.FuzzyInteger(20, 100),
            success_rate=fuzzy.FuzzyFloat(0.8, 1.0)
        )
```

### Realistic Test Data Generation

**AI/ML Job Focus**:

```python
# Domain-specific test data
AI_ML_TITLES = [
    "Senior AI Engineer", "Machine Learning Engineer", 
    "Data Scientist", "MLOps Engineer", "NLP Engineer"
]

TECH_LOCATIONS = [
    "San Francisco, CA", "Remote", "New York, NY", 
    "Seattle, WA", "Austin, TX"
]

def _generate_realistic_salary() -> tuple[int | None, int | None]:
    """Generate realistic AI/ML salary ranges."""
    base_ranges = [
        (90_000, 130_000),   # Junior
        (120_000, 180_000),  # Mid-level  
        (160_000, 250_000),  # Senior
        (200_000, 350_000),  # Staff/Principal
    ]
    # Returns realistic ranges with variation
```

### Session Management Pattern

**SQLite In-Memory Strategy**:

```python
# tests/conftest.py - Database per worker
@pytest.fixture(scope="session")
def test_engine():
    """Create SQLite in-memory engine for each worker."""
    engine = create_engine(
        "sqlite:///:memory:", 
        echo=False,
        connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)
    return engine

@pytest.fixture
def test_session(test_engine):
    """Provide isolated session per test."""
    with Session(test_engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
```

### HTTP Mocking Strategy

**responses Library Integration**:

```python
# External service isolation
@responses.activate  
def test_job_scraping_integration():
    """Test external API integration with mocking."""
    responses.add(
        responses.GET,
        "https://jobs.lever.co/company-name",
        json={"jobs": [...]},
        status=200
    )
    # Test internal logic without external dependencies
```

## Test Organization Structure

### Directory Architecture

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── core/               # Configuration, utils, constants
│   ├── database/           # Models, schemas, database logic  
│   ├── services/           # Business logic services
│   ├── scraping/          # Web scraping components
│   └── ui/                # UI component tests
├── integration/            # Multi-component workflows  
│   ├── test_database_transactions.py
│   ├── test_scraping_workflow.py
│   └── test_analytics_integration.py
├── e2e/                   # End-to-end user workflows
├── performance/           # Performance regression tests
├── property/              # Hypothesis property-based tests
└── benchmarks/            # Performance benchmarking
```

### Test Markers (pytest Configuration)

```python
# pyproject.toml markers for test categorization
markers = [
    "fast: marks tests as fast unit tests",
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests", 
    "database: marks tests requiring database access",
    "ui: marks tests for UI components",
    "parallel_safe: marks tests safe for parallel execution",
    "serial: marks tests that must run serially"
]
```

## Pytest Configuration (Production)

### Core Configuration

```toml
# pyproject.toml - Evidence-based settings
[tool.pytest.ini_options]
minversion = "8.0"
addopts = [
    "--strict-markers",
    "--strict-config", 
    "--cov=src",
    "--cov-report=html:htmlcov",
    "--cov-report=xml:coverage.xml", 
    "--cov-branch",
    "--cov-fail-under=80",
    # Parallel execution
    "-n=auto",              # Auto-detect CPU cores (12 workers measured)
    "--dist=worksteal",     # Work-stealing load balancing
    "--benchmark-disable",  # Enable with --benchmark-enable
]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"] 
python_functions = ["test_*"]
```

### Warning Suppression (Streamlit Context)

```toml
filterwarnings = [
    "ignore:.*ScriptRunContext.*:UserWarning",  # Streamlit testing context
    "ignore:.*pytest_asyncio.*:UserWarning",    # Async test warnings
    "ignore:.*VCR.*:UserWarning",               # HTTP recording warnings
]
```

## Test Execution Patterns

### Parallel Execution Model

- **Worker Distribution**: 12 workers (auto-detected)
- **Load Balancing**: Work-stealing algorithm
- **Isolation**: SQLite in-memory per worker
- **Session Management**: Isolated sessions prevent conflicts

### Test Collection Strategy

```bash
# Collection performance
pytest --collect-only: 3.77s for 1,261 tests
Collection errors: 2 (addressed in maintenance)
File discovery: 78 test files scanned
```

### Performance Optimization

- **Database**: In-memory SQLite per worker (no I/O bottleneck)
- **Mocking**: responses library for HTTP isolation
- **Data Generation**: Seeded faker for reproducibility
- **Coverage**: Parallel-safe coverage collection

## Library Integration Decisions

### Factory-Boy vs Manual Setup

**Chosen**: Factory-Boy with SQLAlchemy integration
**Rationale**:

- Realistic test data with minimal maintenance
- Trait system for scenario variations
- SQLModel compatibility out-of-the-box
- Reduced test setup complexity

### pytest-xdist vs pytest-parallel

**Chosen**: pytest-xdist 3.8.0
**Rationale**:

- Work-stealing load balancing (more efficient)
- Mature codebase with SQLAlchemy support
- Better isolation guarantees
- Performance: 8.9x speedup measured

### Responses vs httpx-mock  

**Chosen**: responses 0.25.8
**Rationale**:

- Simple decorator-based API (`@responses.activate`)
- Established patterns in codebase
- VCR.py compatibility for recording
- Minimal configuration overhead

## Architecture Decisions Summary

| Decision | Library Choice | Alternative Considered | Rationale |
|----------|----------------|------------------------|-----------|
| Test Data | factory-boy 3.3.3 | Manual fixtures | Realistic data, less maintenance |
| Parallel Execution | pytest-xdist 3.8.0 | pytest-parallel | Better load balancing, SQLAlchemy support |
| HTTP Mocking | responses 0.25.8 | httpx-mock | Simpler API, existing patterns |
| Property Testing | hypothesis 6.138.3 | None | Edge case coverage automation |
| Coverage | pytest-cov 6.2.1 | coverage.py direct | Pytest integration, parallel support |

---

**Next**: Coverage Strategy Documentation - detailed analysis of current 26.03% coverage and path to 80% target.
