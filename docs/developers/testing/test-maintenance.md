# Test Maintenance & Operations Guide

> **Production Operations**: CI/CD integration, performance monitoring, and sustainable maintenance patterns for production-ready test suite deployment.

## CI/CD Integration (GitHub Actions)

### Production Test Workflow Configuration

```yaml
# .github/workflows/tests.yml - Production-ready CI/CD
name: Test Suite
on: 
  push: { branches: [main, develop] }
  pull_request: { branches: [main] }

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
        
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          
      # Use uv for fast dependency management (2-3x faster than pip)
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
        
      - name: Install dependencies  
        run: uv sync --dev
        
      # Run tests with evidence-based configuration
      - name: Run test suite
        run: |
          uv run pytest \
            --cov=src \
            --cov-report=xml \
            --cov-report=html \
            --cov-fail-under=80 \
            --benchmark-skip \
            -n=auto \
            --dist=worksteal \
            --tb=short
            
      # Performance monitoring
      - name: Performance regression check
        run: |
          uv run pytest tests/performance/ \
            --benchmark-enable \
            --benchmark-compare-fail=mean:10% \
            --benchmark-storage=file://.benchmarks
            
      # Upload coverage for analysis
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
          fail_ci_if_error: true
          
  # Separate job for long-running integration tests
  integration:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - name: Run integration tests
        run: |
          uv run pytest tests/integration/ \
            --maxfail=3 \
            -v \
            --tb=long
```

### Quality Gates (Evidence-Based)

```yaml
# Quality thresholds based on current metrics
coverage_requirements:
  line_coverage: 80%      # Target from current 26%
  branch_coverage: 60%    # Target from current 15%
  
performance_requirements:
  test_execution: 60s     # Current: 8.7s (with room for growth)
  integration_tests: 30s   # Separate timeout for integration
  
test_reliability:
  max_flaky_tests: 5      # Maximum acceptable flaky tests
  mock_ratio: 20%         # Maximum 200 mocks for 1,000 tests
```

## Performance Monitoring & Regression Prevention

### Benchmark Integration (pytest-benchmark)

```python
# tests/benchmarks/conftest.py - Benchmark configuration
import pytest

@pytest.fixture(scope="session")
def benchmark_config():
    """Configure benchmarks for CI/CD integration."""
    return {
        'min_rounds': 3,
        'max_time': 2.0,
        'disable_gc': True,
        'timer': 'time.perf_counter'
    }

# tests/performance/test_search_performance.py
@pytest.mark.benchmark
def test_job_search_performance(benchmark, realistic_dataset):
    """Monitor job search performance regression."""
    # Benchmark with 10,000 jobs (realistic production scale)
    jobs = realistic_dataset['jobs']  # 10,000 jobs
    
    def search_operation():
        return search_jobs(
            query="AI Engineer", 
            location="Remote",
            limit=100
        )
    
    result = benchmark(search_operation)
    
    # Performance assertions (evidence-based)
    assert len(result) <= 100
    assert benchmark.stats.mean < 0.5  # 500ms performance requirement

@pytest.mark.benchmark
def test_database_query_performance(benchmark, test_session):
    """Monitor database query performance."""
    # Create realistic dataset
    create_realistic_dataset(test_session, companies=100, jobs_per_company=100)
    
    def complex_query():
        return (test_session.query(JobSQL)
                .join(CompanySQL) 
                .filter(JobSQL.salary_min > 100_000)
                .filter(CompanySQL.active == True)
                .limit(1000)
                .all())
    
    result = benchmark(complex_query)
    assert len(result) > 0
    assert benchmark.stats.mean < 0.1  # 100ms requirement
```

### Performance Alerting

```python
# scripts/performance_monitor.py - Production monitoring
import json
import subprocess
from pathlib import Path

def monitor_test_performance():
    """Monitor test suite performance and alert on regression."""
    
    # Run performance tests
    result = subprocess.run([
        "uv", "run", "pytest", 
        "tests/performance/",
        "--benchmark-enable",
        "--benchmark-json=performance.json"
    ], capture_output=True, text=True)
    
    # Parse performance data
    with open("performance.json") as f:
        perf_data = json.load(f)
    
    # Alert on regressions
    for benchmark in perf_data['benchmarks']:
        mean_time = benchmark['stats']['mean']
        
        # Alert thresholds (evidence-based)
        thresholds = {
            'search_jobs': 0.5,        # 500ms max
            'database_query': 0.1,     # 100ms max  
            'full_test_suite': 60.0    # 60s max
        }
        
        test_name = benchmark['name']
        threshold = thresholds.get(test_name.split('_')[-1], 1.0)
        
        if mean_time > threshold:
            print(f"⚠️  Performance regression detected: {test_name}")
            print(f"   Current: {mean_time:.3f}s, Threshold: {threshold:.3f}s")
            return False
    
    return True

if __name__ == "__main__":
    success = monitor_test_performance()
    exit(0 if success else 1)
```

## Coverage Monitoring & Maintenance

### Coverage Regression Prevention

```bash
# scripts/coverage_monitor.sh - Production coverage monitoring
#!/bin/bash

# Run coverage with strict thresholds
uv run pytest \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --cov-fail-under=80 \
  --cov-branch

# Check specific module coverage (critical modules)
python -c "
import xml.etree.ElementTree as ET

tree = ET.parse('coverage.xml')
root = tree.getroot()

# Critical modules that must maintain high coverage
critical_modules = [
    'src/ai_models.py',
    'src/scraper.py', 
    'src/services/job_service.py',
    'src/database.py'
]

for package in root.findall('.//package'):
    for class_elem in package.findall('.//class'):
        filename = class_elem.get('filename')
        line_rate = float(class_elem.get('line-rate'))
        
        if any(module in filename for module in critical_modules):
            if line_rate < 0.85:  # 85% minimum for critical modules
                print(f'⚠️  Critical module {filename} below 85% coverage: {line_rate:.1%}')
                exit(1)
                
print('✅ All critical modules meet coverage requirements')
"
```

### New Code Coverage Requirements

```python
# scripts/diff_coverage.py - Enforce new code coverage
import subprocess
import sys
from pathlib import Path

def check_new_code_coverage():
    """Ensure new code has 90%+ coverage."""
    
    # Get changed files from git
    result = subprocess.run([
        "git", "diff", "--name-only", "origin/main"
    ], capture_output=True, text=True)
    
    changed_files = [
        f for f in result.stdout.strip().split('\n') 
        if f.startswith('src/') and f.endswith('.py')
    ]
    
    if not changed_files:
        print("✅ No source code changes detected")
        return True
    
    # Run coverage on changed files only
    for file_path in changed_files:
        result = subprocess.run([
            "uv", "run", "pytest", 
            f"--cov={file_path}",
            "--cov-report=term-missing",
            "--cov-fail-under=90"  # 90% for new code
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ New code in {file_path} below 90% coverage")
            print(result.stdout)
            return False
    
    print("✅ All new code meets 90% coverage requirement")
    return True

if __name__ == "__main__":
    success = check_new_code_coverage()
    sys.exit(0 if success else 1)
```

## Test Data Management

### Factory Data Maintenance

```python
# tests/data_management/test_factory_realism.py
def test_factory_data_realism():
    """Ensure factory data remains realistic over time."""
    
    # Generate sample data
    companies = CompanyFactory.create_batch(100)
    jobs = JobFactory.create_batch(1000)
    
    # Validate data realism
    company_names = [c.name for c in companies]
    job_titles = [j.title for j in jobs]
    salaries = [j.salary for j in jobs if j.salary]
    
    # Company realism checks
    assert len(set(company_names)) > 80  # 80% unique company names
    assert all(len(name) < 50 for name in company_names)  # Reasonable length
    
    # Job title realism
    ai_ml_keywords = ['AI', 'ML', 'Machine Learning', 'Data Science', 'Engineer']
    title_relevance = sum(
        any(keyword in title for keyword in ai_ml_keywords)
        for title in job_titles
    ) / len(job_titles)
    assert title_relevance > 0.8  # 80% AI/ML relevant titles
    
    # Salary realism (based on 2024 market data)
    salary_mins = [s[0] for s in salaries if s[0]]
    salary_maxs = [s[1] for s in salaries if s[1]]
    
    assert 70_000 <= min(salary_mins) <= 90_000    # Realistic minimum
    assert 300_000 <= max(salary_maxs) <= 500_000  # Realistic maximum
    assert all(s_min < s_max for s_min, s_max in salaries)  # Min < Max

def test_factory_performance():
    """Monitor factory performance to prevent test slowdown."""
    import time
    
    # Benchmark factory performance
    start_time = time.time()
    companies = CompanyFactory.create_batch(100)
    jobs = JobFactory.create_batch(1000) 
    creation_time = time.time() - start_time
    
    # Performance requirements (evidence-based)
    assert creation_time < 5.0  # 5s for 1,100 records
    assert len(companies) == 100
    assert len(jobs) == 1000
```

### Test Database Maintenance

```python
# tests/maintenance/test_database_health.py
def test_test_database_isolation():
    """Ensure test database isolation works correctly."""
    
    # Each test should start with clean state
    initial_companies = test_session.query(CompanySQL).count()
    initial_jobs = test_session.query(JobSQL).count()
    
    assert initial_companies == 0
    assert initial_jobs == 0
    
    # Create test data
    company = CompanyFactory.create()
    job = JobFactory.create(company_id=company.id)
    
    # Verify data exists in current test
    assert test_session.query(CompanySQL).count() == 1
    assert test_session.query(JobSQL).count() == 1
    
    # Data should be isolated to this test session

def test_database_schema_compatibility():
    """Ensure test database schema matches production."""
    from src.models import CompanySQL, JobSQL
    from sqlalchemy import inspect
    
    inspector = inspect(test_engine)
    
    # Check all required tables exist
    table_names = inspector.get_table_names()
    required_tables = ['companies', 'jobs']
    
    for table in required_tables:
        assert table in table_names, f"Missing table: {table}"
    
    # Verify key columns exist
    company_columns = [col['name'] for col in inspector.get_columns('companies')]
    job_columns = [col['name'] for col in inspector.get_columns('jobs')]
    
    assert 'id' in company_columns
    assert 'name' in company_columns
    assert 'url' in company_columns
    
    assert 'id' in job_columns  
    assert 'company_id' in job_columns
    assert 'title' in job_columns
```

## Mock Management & Maintenance

### Mock Count Monitoring

```python
# tests/maintenance/test_mock_health.py
import ast
import os
from pathlib import Path

def test_mock_count_sustainability():
    """Monitor mock count to prevent maintenance overhead."""
    
    mock_count = 0
    test_count = 0
    
    # Scan all test files
    test_dir = Path("tests")
    for test_file in test_dir.rglob("test_*.py"):
        with open(test_file) as f:
            content = f.read()
            tree = ast.parse(content)
            
            # Count test functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    test_count += 1
                    
                    # Count mocks in this test
                    mock_usage = content.count('@patch') + content.count('Mock(') + \
                                content.count('@responses.activate') + \
                                content.count('with patch(')
                    mock_count += mock_usage
    
    # Evidence-based thresholds
    mock_ratio = mock_count / test_count if test_count > 0 else 0
    
    print(f"📊 Mock Analysis:")
    print(f"   Total tests: {test_count}")
    print(f"   Total mocks: {mock_count}")
    print(f"   Mock ratio: {mock_ratio:.1%}")
    
    # Sustainability thresholds (based on current 189 mocks / 1,261 tests = 15%)
    assert mock_count < 250  # Hard limit for maintainability
    assert mock_ratio < 0.20  # 20% maximum mock ratio
    
def test_mock_dependency_health():
    """Check for brittle mock dependencies."""
    
    # Scan for anti-patterns in mock usage
    problematic_patterns = []
    
    for test_file in Path("tests").rglob("test_*.py"):
        with open(test_file) as f:
            content = f.read()
            
            # Check for brittle mock patterns
            if 'patch("builtins.' in content:
                problematic_patterns.append(f"{test_file}: Mocking builtins")
                
            if '_private_method' in content and 'patch' in content:
                problematic_patterns.append(f"{test_file}: Mocking private methods")
                
            if content.count('patch(') > 5:  # More than 5 patches in one file
                problematic_patterns.append(f"{test_file}: Excessive mocking")
    
    # Report but don't fail (for gradual improvement)
    if problematic_patterns:
        print("⚠️  Mock anti-patterns detected:")
        for pattern in problematic_patterns[:10]:  # Limit output
            print(f"   {pattern}")
        print("   Consider refactoring to reduce coupling")
```

## Long-term Maintenance Strategies

### Library Update Management

```python
# scripts/library_health_check.py
import subprocess
import json
from packaging import version

def check_library_versions():
    """Monitor test library versions for security and compatibility."""
    
    # Get current versions
    result = subprocess.run([
        "uv", "pip", "list", "--format=json"
    ], capture_output=True, text=True)
    
    installed = {pkg['name']: pkg['version'] for pkg in json.loads(result.stdout)}
    
    # Critical test libraries to monitor
    critical_libraries = {
        'pytest': '8.0.0',          # Minimum version  
        'pytest-xdist': '3.0.0',
        'factory-boy': '3.0.0',
        'hypothesis': '6.0.0',
        'responses': '0.20.0'
    }
    
    outdated = []
    for lib, min_version in critical_libraries.items():
        current = installed.get(lib)
        if current and version.parse(current) < version.parse(min_version):
            outdated.append(f"{lib}: {current} < {min_version}")
    
    if outdated:
        print("📦 Library updates needed:")
        for lib in outdated:
            print(f"   {lib}")
        return False
    
    print("✅ All test libraries are up to date")
    return True

def check_security_vulnerabilities():
    """Check for security vulnerabilities in test dependencies."""
    
    result = subprocess.run([
        "uv", "pip", "audit"  # When available
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("🔒 Security vulnerabilities detected in test dependencies")
        print(result.stdout)
        return False
        
    print("✅ No security vulnerabilities in test dependencies")
    return True
```

### Test Suite Health Monitoring

```bash
# scripts/test_health_report.py
#!/usr/bin/env python3

import subprocess
import json
import time
from pathlib import Path

def generate_health_report():
    """Generate comprehensive test suite health report."""
    
    print("🏥 Test Suite Health Report")
    print("=" * 50)
    
    # Test execution metrics
    start_time = time.time()
    result = subprocess.run([
        "uv", "run", "pytest", 
        "--collect-only", "-q"
    ], capture_output=True, text=True)
    
    collection_time = time.time() - start_time
    test_count = result.stdout.count(" test collected")
    
    print(f"📊 Execution Metrics:")
    print(f"   Test count: {test_count}")
    print(f"   Collection time: {collection_time:.2f}s")
    
    # Coverage analysis
    coverage_result = subprocess.run([
        "uv", "run", "pytest", 
        "--cov=src", "--cov-report=json:coverage.json",
        "--tb=no", "-q"
    ], capture_output=True)
    
    if Path("coverage.json").exists():
        with open("coverage.json") as f:
            coverage_data = json.load(f)
            
        total_lines = coverage_data['totals']['num_statements']
        covered_lines = coverage_data['totals']['covered_lines']
        coverage_percent = (covered_lines / total_lines) * 100
        
        print(f"📈 Coverage Analysis:")
        print(f"   Line coverage: {coverage_percent:.1f}%")
        print(f"   Lines covered: {covered_lines:,}/{total_lines:,}")
    
    # File count analysis
    test_files = list(Path("tests").rglob("test_*.py"))
    source_files = list(Path("src").rglob("*.py"))
    
    print(f"📁 File Statistics:")
    print(f"   Test files: {len(test_files)}")
    print(f"   Source files: {len(source_files)}")
    print(f"   Test/Source ratio: {len(test_files)/len(source_files):.2f}")
    
    # Quality indicators
    print(f"🎯 Quality Indicators:")
    
    # Check for TODO/FIXME markers
    todo_count = 0
    for test_file in test_files:
        with open(test_file) as f:
            content = f.read()
            todo_count += content.count("TODO") + content.count("FIXME")
    
    print(f"   Technical debt markers: {todo_count}")
    
    # Performance trend (if historical data available)
    if Path(".benchmarks").exists():
        print(f"   Benchmark history: Available")
    else:
        print(f"   Benchmark history: No data")
    
    print("\n🎉 Health check completed!")

if __name__ == "__main__":
    generate_health_report()
```

## Continuous Improvement Process

### Monthly Maintenance Tasks

```python
# scripts/monthly_maintenance.py
def monthly_test_maintenance():
    """Automated monthly maintenance tasks."""
    
    tasks = [
        ("Update library versions", update_test_libraries),
        ("Clean unused factories", clean_unused_factories),
        ("Optimize slow tests", identify_slow_tests),
        ("Review mock usage", audit_mock_usage),
        ("Update test data", refresh_test_datasets),
        ("Performance baseline", update_performance_baselines)
    ]
    
    results = []
    for task_name, task_func in tasks:
        try:
            print(f"🔧 {task_name}...")
            task_func()
            results.append((task_name, "✅ SUCCESS"))
        except Exception as e:
            results.append((task_name, f"❌ FAILED: {str(e)}"))
    
    # Generate maintenance report
    print("\n📋 Monthly Maintenance Report:")
    for task_name, status in results:
        print(f"   {task_name}: {status}")

def identify_slow_tests():
    """Identify tests taking >5s for optimization."""
    result = subprocess.run([
        "uv", "run", "pytest", 
        "--durations=10",
        "--tb=no", "-v"
    ], capture_output=True, text=True)
    
    slow_tests = []
    for line in result.stdout.split('\n'):
        if 's call' in line and float(line.split()[0][:-1]) > 5.0:
            slow_tests.append(line.strip())
    
    if slow_tests:
        print("🐌 Slow tests identified for optimization:")
        for test in slow_tests[:5]:  # Top 5
            print(f"   {test}")
```

---

**Production Readiness Checklist**:

- ✅ **Test Architecture**: Comprehensive library-first patterns documented
- ✅ **Coverage Strategy**: Evidence-based path from 26% to 80% coverage  
- ✅ **Test Patterns**: Library integration examples with anti-pattern avoidance
- ✅ **Operations Guide**: CI/CD integration and maintenance automation

**Deployment Timeline**: 12-week implementation plan with measurable milestones and sustainable maintenance processes.
