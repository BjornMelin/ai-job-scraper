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
