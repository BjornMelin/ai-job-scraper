# 🛠️ Developer Guide: AI Job Scraper

This guide provides a comprehensive technical overview for developers looking to understand, contribute to, or extend the AI Job Scraper codebase.

## 🏗️ Architecture Overview

The application follows a modern, modular architecture that separates concerns into distinct layers, ensuring maintainability and scalability.

```mermaid
graph TD
    subgraph "UI Layer (src/ui)"
        A[main.py & st.navigation] --> B[Pages (jobs.py, companies.py)]
        B --> C[Components (job_card.py, sidebar.py)]
        C --> D[State (session_state.py)]
    end
    
    subgraph "Service Layer (src/services)"
        E[JobService]
        F[CompanyService]
        G[SmartSyncEngine]
    end
    
    subgraph "Scraping & Agentic Workflows (src)"
        H[scraper.py Orchestrator] --> I[scraper_company_pages.py]
        H --> J[scraper_job_boards.py]
    end
    
    subgraph "Data Layer (src)"
        K[database.py] --> L[models.py: SQLModel]
    end

    B -- Uses --> E & F
    H -- Uses --> G
    E & F & G -- Interact with --> K
```

### Core Modules

| Module/Directory        | Purpose                                                                                             |
| ----------------------- | --------------------------------------------------------------------------------------------------- |
| `src/main.py`           | Main application entry point. Handles page configuration and multi-page navigation via `st.navigation`. |
| `src/ui/pages/`         | Contains the Streamlit code for each distinct page of the application (Jobs, Companies, Analytics).      |
| `src/ui/components/`    | Reusable UI components (e.g., job cards, sidebar) used by the pages.                                |
| `src/ui/state/`         | Manages the application's UI state using Streamlit's native `session_state`.                        |
| `src/services/`         | Encapsulates all business logic (database, analytics, cost monitoring).                      |
| `src/scraper.py`        | The main orchestrator for all scraping tasks.                                                       |
| `src/scraper_*.py`      | Specialized modules for scraping job boards (`JobSpy`) and company pages (`ScrapeGraphAI`).         |
| `src/database.py`       | Handles database engine creation, session management, and SQLite configuration.                    |
| `src/models.py`         | Defines the application's data structures using `SQLModel`.                                         |
| `src/config.py`         | Manages application settings and secrets using `pydantic-settings`.                                 |

## 📚 Technical Stack

* **UI Framework:** Streamlit 1.47+ with native caching

* **Database ORM:** SQLModel for SQLite operations

* **Analytics Engine:** DuckDB 0.9.0+ with sqlite_scanner extension

* **Scraping Libraries:** ScrapeGraphAI (for company pages), JobSpy (for job boards)

* **LLM Integration:** LiteLLM unified client + Instructor validation

* **LLM Providers:** OpenAI, Groq (configurable via LiteLLM)

* **Package Management:** `uv`

* **Code Quality:** `ruff`

## 🔍 Code Deep Dive

### UI Architecture (`src/ui/`)

The UI is a multi-page Streamlit application.

* **Navigation:** `src/main.py` uses `st.navigation()` to define the pages. This is the modern, recommended approach and handles routing, state, and icons.

* **State Management:** We have deliberately moved away from a custom `StateManager` singleton. All UI state is managed directly via `st.session_state`, which is simpler and more idiomatic for Streamlit. The `src/ui/state/session_state.py` module provides helper functions to initialize the default state.

* **Componentization:** Pages in `src/ui/pages/` are responsible for the overall layout and data fetching, while smaller, reusable parts of the UI (like a single job card) are defined in `src/ui/components/`.

### Service Layer (`src/services/`)

This layer abstracts all business logic away from the UI.

* **`JobService` & `CompanyService`:** Provide simple, static methods for CRUD (Create, Read, Update, Delete) operations on jobs and companies. They contain all `SQLModel` query logic.

* **`AnalyticsService`:** DuckDB-powered analytics using sqlite_scanner for zero-ETL data analysis. Provides job trends, company metrics, and salary analytics.

* **`CostMonitor`:** SQLModel-based cost tracking with $50 monthly budget monitoring and service-level cost breakdown.

* **`SmartSyncEngine`:** Handles updating the database with scraped data without destroying user edits. See `ADR-008` for details.

### Background Task Management (`src/ui/utils/background_helpers.py`)

Background task management follows ADR-017's simplified threading approach, reducing complexity by 94% while maintaining essential functionality for I/O-bound scraping operations.

#### Simplified Threading Architecture

**Core Design Principles:**

* **Standard Threading**: Uses Python's built-in `threading.Thread` (no custom ThreadPoolExecutor)
* **Native Streamlit Integration**: Leverages `st.session_state` and `st.status` components
* **Session State Coordination**: Thread-safe coordination via atomic session operations
* **Streamlit Context**: Proper threading context with `add_script_run_ctx()`

**5-Function API:**

```python
from src.ui.utils.background_helpers import (
    start_background_scraping,
    is_scraping_active, 
    stop_all_scraping,
    get_company_progress,
    throttled_rerun
)

# Start background scraping
task_id = start_background_scraping()

# Check if scraping is active
if is_scraping_active():
    # Get real-time progress
    progress = get_company_progress()
    
    # Display progress with throttled refresh
    for company, status in progress.items():
        st.info(f"{company}: {status.jobs_found} jobs found")
    
    # Auto-refresh every 2 seconds
    throttled_rerun(should_rerun=True)

# Stop all tasks
stop_all_scraping()
```

#### Threading Implementation Pattern

**Standard Thread Creation:**

```python
def scraping_worker():
    """Background thread worker with Streamlit context."""
    try:
        st.session_state.scraping_active = True
        
        with st.status("🔍 Scraping jobs...", expanded=True) as status:
            companies = st.session_state.get('selected_companies', [])
            
            for i, company in enumerate(companies):
                status.write(f"Processing {company} ({i+1}/{len(companies)})")
                # Scraping operations...
                
            status.update(label="✅ Scraping completed!", state="complete")
                
    except Exception as e:
        st.error(f"Scraping failed: {str(e)}")
    finally:
        st.session_state.scraping_active = False

# Create thread with Streamlit context
thread = threading.Thread(target=scraping_worker, daemon=True)
add_script_run_ctx(thread)  # Essential for Streamlit compatibility
thread.start()
```

#### Session State Coordination

**Atomic Operations:**

```python
# Thread-safe coordination using session state locks
_session_state_lock = threading.Lock()

def _atomic_check_and_set(key: str, check_value, set_value) -> bool:
    """Atomically check and set session state value."""
    with _session_state_lock:
        current_value = st.session_state.get(key)
        if current_value == check_value:
            st.session_state[key] = set_value
            return True
        return False

# Prevent concurrent execution
if not _atomic_check_and_set("scraping_active", False, True):
    st.warning("Scraping already in progress")
    return
```

#### Progress Display Integration

**Auto-Refresh Fragments:**

```python
@st.fragment(run_every="2s")
def background_task_status_fragment():
    """Auto-refreshing status display during background tasks."""
    if not is_scraping_active():
        return
        
    # Display progress with real-time updates
    progress = get_company_progress()
    for company_name, company_status in progress.items():
        st.progress(
            company_status.jobs_found / 100,  # Normalize progress
            text=f"{company_name}: {company_status.status}"
        )
```

#### Key Advantages

**Simplified Maintenance:**

* 94% code reduction compared to custom ThreadPoolExecutor implementations
* No complex task queue management or worker pool coordination
* Standard library patterns reduce debugging complexity

**Native Streamlit Integration:**

* Direct `st.status` component usage for progress display
* Built-in session state coordination patterns
* Fragment-based auto-refresh without manual polling

**Optimal for I/O-Bound Operations:**

* Threading approach ideal for network requests and database operations
* No CPU-bound processing overhead from complex task management
* Memory-efficient for long-running scraping sessions (100MB max)

### Analytics & Cost Monitoring (`src/services/`)

The application includes comprehensive analytics and cost monitoring built with modern Python libraries.

#### DuckDB Analytics Service (`analytics_service.py`)

**Zero-ETL Analytics Architecture:**

* **DuckDB sqlite_scanner**: Direct SQLite database scanning without ETL processes
* **No Separate Database**: DuckDB reads SQLite files directly in-memory
* **Streamlit Caching**: 5-minute TTL for dashboard performance
* **Automatic Fallback**: Graceful degradation when DuckDB unavailable

**Core Analytics Functions:**

```python
analytics = AnalyticsService()

# Job market trends over time
trends = analytics.get_job_trends(days=30)

# Company hiring metrics with salary analysis
companies = analytics.get_company_analytics()

# Salary statistics and ranges
salaries = analytics.get_salary_analytics(days=90)
```

**Technical Implementation:**

* **Connection**: In-memory DuckDB with sqlite_scanner extension
* **Query Pattern**: Direct SQL on SQLite tables via `sqlite_scan()`
* **Performance**: Sub-second analytics queries on 500K+ records
* **Caching**: Streamlit `@st.cache_data` with configurable TTL

#### Cost Monitor Service (`cost_monitor.py`)

**$50 Monthly Budget Tracking:**

* **SQLModel Integration**: Type-safe cost entry models with timezone handling
* **Service Breakdown**: AI, proxy, and scraping cost categories
* **Real-time Alerts**: 80% and 100% budget threshold notifications
* **Dashboard Integration**: Live cost tracking with Plotly visualizations

**Cost Tracking Usage:**

```python
monitor = CostMonitor()

# Track AI operation costs
monitor.track_ai_cost("gpt-4", 1000, 0.02, "job_extraction")

# Track proxy requests
monitor.track_proxy_cost(150, 5.00, "iproyal_residential")

# Track scraping operations
monitor.track_scraping_cost("OpenAI", 25, 0.10)

# Get monthly budget summary
summary = monitor.get_monthly_summary()
```

**Data Model:**

```python
class CostEntry(SQLModel, table=True):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    service: str       # "ai", "proxy", "scraping"
    operation: str     # Operation description
    cost_usd: float    # Cost in USD
    extra_data: str    # JSON metadata
```

#### Analytics Dashboard (`src/ui/pages/analytics.py`)

**Dashboard Features:**

* **Cost Monitoring**: Monthly budget tracking with service breakdowns
* **Job Trends**: Time-series analysis with configurable date ranges
* **Company Analytics**: Hiring metrics and salary statistics
* **Interactive Charts**: Plotly-powered visualizations
* **Real-time Updates**: Streamlit fragments with auto-refresh

**Performance Characteristics:**

* **Cost Queries**: <100ms response times with SQLite
* **Analytics Queries**: <2s DuckDB aggregations on large datasets
* **Dashboard Load**: <500ms initial page render with caching
* **Memory Usage**: Minimal - DuckDB runs in-memory, no persistent files

## 🤝 Contributing Guidelines

### Development Environment Setup

1. **Clone the repository:**

    ```bash
    git clone https://github.com/BjornMelin/ai-job-scraper.git
    cd ai-job-scraper
    ```

2. **Install dependencies:**
    `uv` is the required package manager.

    ```bash
    uv sync
    ```

3. **Set up pre-commit hooks (recommended):**
    This will automatically format and lint your code on every commit.

    ```bash
    uv run pre-commit install
    ```

4. **Run the app:**

    ```bash
    streamlit run src/main.py
    ```

### Code Style & Quality

* **Formatting & Linting:** We use `ruff` for all code quality checks. Please run `uv run ruff format .` and `uv run ruff check . --fix` before committing.

* **Type Hinting:** All functions and methods must have full type hints.

* **Docstrings:** Use Google-style docstrings for all public modules, classes, and functions.

### Pull Request Process

1. Create a feature branch from `main`.
2. Implement your changes, adhering to the architecture and code style.
3. Add or update tests for your changes in the `tests/` directory.
4. Ensure all quality checks and tests pass: `uv run pytest`.
5. Submit a pull request with a clear description of the changes and reference any relevant issues or ADRs.

## 🧪 Testing Architecture

The application maintains >90% test coverage with a comprehensive test suite organized by testing strategy.

### Test Structure (`tests/`)

```text
tests/
├── unit/                    # Unit tests for isolated components
│   ├── core/               # Config, constants, utilities
│   ├── database/           # Database models, schemas, operations  
│   ├── models/             # SQLModel validation and parsing
│   ├── scraping/           # Scraper components and integrations
│   ├── services/           # Service layer business logic
│   └── ui/                 # UI component unit tests
├── integration/            # Integration tests across services
│   ├── test_analytics_integration.py
│   ├── test_scraping_workflow.py
│   └── test_session_isolation.py
├── performance/            # Performance regression tests
│   ├── test_pagination.py
│   ├── test_search_performance.py
│   └── test_performance_optimization.py
├── services/               # Service-specific test suites
│   ├── test_analytics_service.py
│   ├── test_cost_monitor.py
│   └── test_job_service_modern.py
├── ui/                     # Streamlit UI testing
│   ├── components/         # Component-level tests
│   ├── pages/              # Page-level integration tests
│   └── utils/              # UI utility and helper tests
└── compatibility/          # Cross-version compatibility tests
```

### Testing Approach

**Unit Tests (>70% of test suite)**:

* **Fast execution**: <5s total runtime for unit tests
* **Isolated components**: No external dependencies
* **SQLModel focus**: Database model validation, relationship testing
* **Service layer**: Business logic verification with mocked dependencies

**Integration Tests**:

* **Cross-service workflows**: End-to-end scraping and analytics pipelines  
* **Database integration**: Real SQLite database operations with test isolation
* **Analytics validation**: DuckDB sqlite_scanner functionality testing
* **Cost monitoring**: Budget tracking and alert system validation

**Performance Tests**:

* **Search benchmarks**: FTS5 query performance with various dataset sizes
* **Analytics benchmarks**: DuckDB aggregation performance regression detection
* **UI rendering**: Streamlit component load time validation
* **Memory usage**: Resource utilization monitoring during operations

**UI Component Tests**:

* **Streamlit mocking**: Session state and component behavior validation
* **Interactive elements**: Job card interactions, filter operations, modal displays
* **Analytics dashboard**: Chart rendering and data visualization testing
* **Background task UI**: Progress tracking and real-time update validation

### Test Execution

**Standard Test Run:**

```bash
# Run all tests with coverage reporting
uv run pytest

# Run specific test categories
uv run pytest tests/unit/            # Unit tests only
uv run pytest tests/integration/     # Integration tests only  
uv run pytest tests/performance/     # Performance benchmarks
uv run pytest -m "not slow"         # Skip long-running tests
```

**Development Workflow:**

```bash
# Quick feedback loop during development
uv run pytest tests/unit/services/test_analytics_service.py -v

# Test specific functionality with coverage
uv run pytest tests/services/test_cost_monitor.py --cov=src.services.cost_monitor

# Performance regression testing
uv run pytest tests/performance/ --benchmark-only
```

### Key Testing Patterns

**SQLModel Testing**:

```python
def test_cost_entry_model():
    """Test SQLModel cost entry validation and timezone handling."""
    entry = CostEntry(service="ai", operation="gpt-4", cost_usd=0.02)
    assert entry.timestamp.tzinfo == UTC
    assert entry.service == "ai"
```

**DuckDB Analytics Testing**:

```python
def test_analytics_service_duckdb_integration():
    """Test DuckDB sqlite_scanner analytics functionality."""
    analytics = AnalyticsService()
    trends = analytics.get_job_trends(days=7)
    assert trends["method"] == "duckdb_sqlite_scanner"
    assert trends["status"] == "success"
```

**Streamlit Component Testing**:

```python
def test_analytics_dashboard_rendering(mock_streamlit):
    """Test analytics page rendering with mocked Streamlit context."""
    with patch('streamlit.session_state', {}):
        render_analytics_page()
        mock_streamlit.title.assert_called_with("📊 Analytics Dashboard")
```

### Test Infrastructure

**Fixtures and Mocking**:

* **Database fixtures**: Temporary SQLite databases for test isolation
* **Streamlit mocking**: Session state and component mocking for UI tests  
* **Analytics fixtures**: Sample data for analytics and cost monitoring tests
* **Background task mocking**: Thread and progress tracking validation

**Coverage Goals**:

* **Overall**: >90% code coverage
* **New features**: 95% coverage requirement
* **Critical paths**: 100% coverage (cost tracking, data synchronization)
* **UI components**: >85% coverage with interaction testing

**Test Data Management**:

* **Isolation**: Each test creates independent database instances
* **Cleanup**: Automatic test database and file cleanup
* **Sample data**: Realistic job and company data for comprehensive testing
* **Performance data**: Benchmarking datasets for regression detection

### Continuous Integration

**GitHub Actions Integration**:

* **Matrix testing**: Python 3.12+ across operating systems
* **Dependency testing**: Latest and pinned dependency versions
* **Performance monitoring**: Benchmark comparison and regression alerts
* **Coverage reporting**: Automatic coverage analysis and PR comments

**Quality Gates**:

* All tests must pass before merging
* Coverage cannot decrease from baseline
* Performance benchmarks must meet baseline requirements
* Ruff linting and formatting must pass
