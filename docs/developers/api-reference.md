# 📖 API Reference: AI Job Scraper

This document provides a technical reference for the core data models and services in the AI Job Scraper application.

## 🗄️ Data Models (`src/models.py`)

The application uses `SQLModel` for its data layer, combining the features of SQLAlchemy and Pydantic.

### `CompanySQL`

Represents a company whose career page is a source for job scraping.

| Field          | Type                | Description                                                  |
| -------------- | ------------------- | ------------------------------------------------------------ |
| `id`           | `int` (PK)          | Auto-incrementing primary key.                               |
| `name`         | `str` (Unique)      | The unique name of the company.                              |
| `url`          | `str`               | The URL of the company's main careers page.                  |
| `active`       | `bool`              | If `True`, the company will be included in scraping runs.    |
| `last_scraped` | `datetime \| None`  | Timestamp of the last time this company was scraped.         |
| `scrape_count` | `int`               | A counter for the total number of scrape attempts.           |
| `success_rate` | `float`             | A weighted success rate for scraping this company.           |
| `jobs`         | `list["JobSQL"]`    | SQLAlchemy relationship to the jobs from this company.       |

### `JobSQL`

Represents a single job posting scraped from a source.

| Field                | Type                               | Description                                                              |
| -------------------- | ---------------------------------- | ------------------------------------------------------------------------ |
| `id`                 | `int` (PK)                         | Auto-incrementing primary key.                                           |
| `company_id`         | `int` (FK)                         | Foreign key linking to the `CompanySQL` table.                           |
| `title`              | `str`                              | The title of the job posting.                                            |
| `description`        | `str`                              | The full description of the job.                                         |
| `link`               | `str` (Unique)                     | The unique URL to the job application or details page.                   |
| `location`           | `str`                              | The physical or remote location of the job.                              |
| `posted_date`        | `datetime \| None`                 | The date the job was originally posted.                                  |
| `salary`             | `tuple[int, int] \| None`          | A tuple representing the parsed (min, max) salary range.                 |
| `favorite`           | `bool`                             | **User-editable:** `True` if the user has favorited this job.            |
| `notes`              | `str`                              | **User-editable:** Personal notes added by the user.                     |
| `content_hash`       | `str`                              | An MD5 hash of the job's content, used for change detection.             |
| `application_status` | `str`                              | **User-editable:** The user's application status (e.g., "New", "Applied"). |
| `application_date`   | `datetime \| None`                 | **User-editable:** The date the user marked the job as "Applied".        |
| `archived`           | `bool`                             | `True` if the job is no longer found on the source but has user data.    |
| `last_seen`          | `datetime \| None`                 | Timestamp of the last time this job was seen in a scrape.                |
| `company_relation`   | `CompanySQL`                       | SQLAlchemy relationship to the parent company.                           |

## 🔧 Core Services API (`src/services/`)

### `CompanyService`

Provides methods for managing company records.

* `get_all_companies() -> list[CompanySQL]`

* `add_company(name: str, url: str) -> CompanySQL`

* `toggle_company_active(company_id: int) -> bool`

* `get_active_companies() -> list[CompanySQL]`

* `update_company_scrape_stats(company_id: int, success: bool, ...)`

### `JobService`

Provides methods for querying and updating job records.

* `get_filtered_jobs(filters: dict) -> list[JobSQL]`: The primary method for fetching jobs for the UI. Takes a dictionary of filter criteria including optional salary range filters.
  * `filters["salary_min"]`: Minimum salary filter (inclusive). Jobs with max salary >= this value are included. Only applied if > 0.
  * `filters["salary_max"]`: Maximum salary filter (inclusive). Jobs with min salary <= this value are included. When set to 750000, acts as unbounded (includes all jobs >= 750k).
  * The salary filtering uses smart overlap logic to match jobs whose salary range overlaps with the user's filter range.

* `update_job_status(job_id: int, status: str) -> bool`

* `toggle_favorite(job_id: int) -> bool`

* `update_notes(job_id: int, notes: str) -> bool`

* `get_job_counts_by_status() -> dict[str, int]`

### `SmartSyncEngine`

Handles the intelligent synchronization of scraped data with the database.

* `sync_jobs(jobs: list[JobSQL]) -> dict[str, int]`: The main entry point for the engine. It takes a list of scraped `JobSQL` objects and performs the full sync logic, returning a dictionary of statistics (inserted, updated, archived, deleted, skipped).

## 🔄 Background Task Management (`src/ui/utils/background_helpers.py`)

Simplified threading-based background task management following ADR-017's minimal approach.

### Core API Functions

* **`start_background_scraping(stay_active_in_tests: bool = False) -> str`**
  * Initiates background scraping using standard Python threading
  * Returns unique task ID for progress tracking
  * Uses Streamlit session state coordination to prevent concurrent execution
  * Integrates with `add_script_run_ctx()` for proper Streamlit context
  * **Parameters:**
    * `stay_active_in_tests`: If `True`, keeps task active in test environment
  
* **`is_scraping_active() -> bool`**
  * Checks current scraping status via session state flags
  * Thread-safe through Streamlit's session state management
  * Returns `True` if scraping operation is running, `False` otherwise

* **`stop_all_scraping() -> int`**
  * Stops all active scraping operations with proper thread cleanup
  * Uses session state coordination to signal graceful termination
  * Returns count of stopped tasks (0 or 1 in simplified implementation)
  * Includes timeout-based thread cleanup to prevent hanging

* **`get_company_progress() -> dict[str, CompanyProgress]`**
  * Returns real-time company-level scraping progress
  * Progress data includes status, job counts, timing, and error information
  * Used by UI components for live progress display

* **`throttled_rerun(session_key: str = "last_refresh", interval_seconds: float = 2.0, *, should_rerun: bool = True) -> None`**
  * Auto-refresh utility with rate limiting to prevent excessive reruns
  * Only triggers `st.rerun()` when specified interval has elapsed
  * Essential for background task status updates without performance impact

### Data Models

**`CompanyProgress` (dataclass)**

```python
@dataclass
class CompanyProgress:
    name: str
    status: str = "Pending"
    jobs_found: int = 0
    start_time: datetime | None = None
    end_time: datetime | None = None
    error: str | None = None
```

**`ProgressInfo` (dataclass)**

```python
@dataclass  
class ProgressInfo:
    progress: float      # 0.0 to 1.0
    message: str
    timestamp: datetime
```

### Integration Patterns

**Streamlit Components:**

* Uses `st.session_state` for task coordination
* Integrates with `st.status` component for progress display
* Supports `@st.fragment` decorators for auto-refresh display

**Threading Requirements:**

* Standard Python `threading.Thread` with `daemon=True`
* Requires `add_script_run_ctx(thread)` for Streamlit compatibility
* Session state flags coordinate thread lifecycle

**Usage Example:**

```python
from src.ui.utils.background_helpers import (
    start_background_scraping,
    is_scraping_active, 
    get_company_progress
)

# Start background operation
task_id = start_background_scraping()

# Check status
if is_scraping_active():
    progress = get_company_progress()
    st.info(f"Scraping {len(progress)} companies...")
```

## 📊 Analytics & Monitoring Services

### `AnalyticsService` (`src/services/analytics_service.py`)

DuckDB-powered analytics service providing zero-ETL data analysis.

#### Constructor

```python
AnalyticsService(db_path: str = "jobs.db")
```

#### Core Methods

* **`get_job_trends(days: int = 30) -> AnalyticsResponse`**
  * Returns job posting trends over specified time period
  * Uses DuckDB's `DATE_TRUNC` for daily aggregation
  * Includes trend data, total jobs, and method metadata
  * Cached for 5 minutes via `@st.cache_data(ttl=300)`

* **`get_company_analytics() -> AnalyticsResponse`**
  * Returns company hiring metrics with salary statistics
  * Aggregates total jobs, average salary ranges per company
  * Limited to top 20 companies by job count
  * Includes JSON salary parsing via DuckDB's `json_extract`

* **`get_salary_analytics(days: int = 90) -> AnalyticsResponse`**
  * Returns salary statistics for specified period
  * Calculates average, min, max salary ranges and standard deviation
  * Filters jobs with non-null salary data
  * Uses DuckDB's native statistical functions

* **`get_status_report() -> dict[str, Any]`**
  * Returns service configuration and connection status
  * Includes DuckDB availability, database path, connection state

#### Response Format

```python
type AnalyticsResponse = dict[str, Any]
# Standard fields:
# - "status": "success" | "error"
# - "method": "duckdb_sqlite_scanner"
# - "error": str (if status == "error")
# - Data fields vary by method
```

#### Technical Implementation

* **DuckDB Connection**: In-memory with sqlite_scanner extension
  * **Query Pattern**: Direct SQLite scanning via `sqlite_scan(db_path, table)`
  * **Fallback Handling**: Graceful degradation when DuckDB unavailable
  * **Streamlit Integration**: Native caching with configurable TTL

### `CostMonitor` (`src/services/cost_monitor.py`)

SQLModel-based cost tracking service with $50 monthly budget monitoring.

#### Data Model

**`CostEntry` (SQLModel table)**

```python
class CostEntry(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    service: str = Field(index=True)  # "ai", "proxy", "scraping"  
    operation: str                    # Operation description
    cost_usd: float                   # Cost in USD
    extra_data: str = ""              # JSON metadata
```

#### Constructor - Costs

```python
CostMonitor(db_path: str = "costs.db")
```

* Creates SQLite database with `costs.db` default
  * Sets monthly budget to $50.00
  * Auto-creates tables via `SQLModel.metadata.create_all()`

#### Cost Tracking Methods

* **`track_ai_cost(model: str, tokens: int, cost: float, operation: str) -> None`**
  * Records AI/LLM operation costs
  * Stores model name and token count in `extra_data`
  * Triggers budget alerts after cost addition

* **`track_proxy_cost(requests: int, cost: float, endpoint: str) -> None`**
  * Records proxy service costs
  * Stores request count and endpoint in `extra_data`
  * Used for IPRoyal residential proxy tracking

* **`track_scraping_cost(company: str, jobs_found: int, cost: float) -> None`**
  * Records scraping operation costs  
  * Stores company name and job count in `extra_data`
  * Tracks per-company scraping expenses

#### Budget Monitoring Methods

* **`get_monthly_summary() -> dict[str, Any]`**
  * Returns current month cost breakdown by service
  * Includes total cost, remaining budget, utilization percentage
  * Cached for 1 minute via `@st.cache_data(ttl=60)`
  * Budget status: "within_budget", "moderate_usage", "approaching_limit", "over_budget"

* **`get_cost_alerts() -> list[dict[str, str]]`**
  * Returns active cost alerts for dashboard display
  * Alert types: "error" (>100%), "warning" (>80%)
  * Structured for Streamlit alert components

#### Budget Alert System

* **80% Threshold**: Warning alerts via Streamlit and logging
* **100% Threshold**: Error alerts via Streamlit and logging
* **Real-time Monitoring**: Triggered after each cost entry
* **Streamlit Integration**: Automatic UI alerts when available

### Analytics Dashboard Integration (`src/ui/pages/analytics.py`)

The analytics page integrates both services with Plotly visualizations:

* **Cost Monitoring**: Pie charts for service breakdown, metric cards for budget status
* **Job Trends**: Line charts with configurable time ranges (7/30/90 days)  
* **Company Analytics**: Bar charts for top companies, interactive data tables
* **Salary Analytics**: Metric displays for salary statistics and ranges
* **Service Status**: Expandable technical status information
