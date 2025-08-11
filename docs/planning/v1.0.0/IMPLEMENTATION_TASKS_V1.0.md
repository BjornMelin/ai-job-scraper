# AI Job Scraper - Detailed Implementation Plan (V1.0 - Job Hunter's MVP)

## Introduction

This document outlines the tasks for the **V1.0 "Job Hunter's MVP" Release**. It assumes that all tasks in the **V0.0 Foundational Refactoring** phase have been completed. The application now has a stable architecture, a correct database schema, and a data-safe synchronization engine. The goal of this phase is to build the essential user-facing features on top of this solid foundation, incorporating key UI requirements for immediate value.

---

## 🚀 V1.0: The "Job Hunter's MVP" Release

### **T1.1: Implement Core Job Browser & Application Tracking**

- **Release**: V1.0

- **Priority**: **High**

- **Status**: **✅ COMPLETED** (PR #26)

- **Prerequisites**: `T0.1`, `T0.2`, `T0.3`

- **Requirements File**: `docs/planning/v1.0.0/REQUIREMENTS_V1.0.md`

- **Related Requirements**: `UI-JOBS-01`, `UI-JOBS-02`, `UI-JOBS-03`, `UI-JOBS-04`, `UI-TRACK-01`

- **Libraries**: `streamlit==1.47.1`

- **Description**: Build the primary user interface for browsing, filtering, and managing jobs within the newly established component architecture.

- **Developer Context**: All work for this task will happen within the `src/ui/` directory. You will be creating new services and components that interact with the database models and state manager created in Phase 0.

- **Sub-tasks & Instructions**:
  - **T1.1.1: Implement Job Service**: ✅ **COMPLETED**
    - **Instructions**: In the `src/services/` directory, create a new file `job_service.py`.
    - **Instructions**: Create a `JobService` class with static methods. Implement `get_filtered_jobs(filters: dict)`. This method will query the database using the synchronous session from `database.py` and apply filters for text search, company, and application status.
    - **Instructions**: Implement `update_job_status(job_id: int, status: str)`, `toggle_favorite(job_id: int)`, and `update_notes(job_id: int, notes: str)`. These methods will update a single job record in the database.
    - **Success Criteria**: The service provides a clean API for the UI to fetch and update job data without containing any UI code itself.
  - **T1.1.2: Build Interactive Job Card Component**: ✅ **COMPLETED**
    - **Instructions**: In `src/ui/components/cards/`, create `job_card.py`.
    - **Instructions**: The `render_job_card(job: JobSQL)` function should accept a `JobSQL` object. Inside, use `st.container` with a border.
    - **Instructions**: Add a `st.selectbox` for the `application_status` field. Its `on_change` parameter must point to a callback function that calls `JobService.update_job_status`.
    - **Instructions**: Add a `st.button` for the favorite toggle (using a heart icon: ❤️/🤍). Its `on_click` must call `JobService.toggle_favorite`.
    - **Instructions**: Add a `st.button("View Details")`. Its `on_click` should update a session state variable, e.g., `st.session_state.expanded_job_id = job.id`.
    - **Success Criteria**: The job card correctly displays job data and its interactive widgets successfully update the database via the `JobService`.
  - **T1.1.3: Build Job Details Expander**: ✅ **COMPLETED** (Enhanced with Modal)
    - **Instructions**: In the rendering loop within `src/ui/pages/jobs.py`, check if a job's ID matches `st.session_state.get('expanded_job_id')`.
    - **Instructions**: If it matches, render an `st.expander("Details")` directly below its card. Inside the expander, display `st.markdown(job.description)` and a `st.text_area("Notes", value=job.notes)`. The text area's `on_change` callback should save the notes to the database via the `JobService.update_notes` method.
    - **Success Criteria**: Clicking "View Details" reveals the job description and notes area below the card. Editing notes persists the changes.

### **T1.2: Implement Background Scraping & Progress Dashboard**

- **Release**: V1.0

- **Priority**: **High**

- **Status**: **✅ COMPLETED** (PR #27)

- **Prerequisites**: `T0.3`

- **Requirements File**: `docs/planning/v1.0.0/REQUIREMENTS_V1.0.md`

- **Related Requirements**: `SYS-ARCH-04`, `SCR-PROG-01`, `UI-PROG-02`, `SCR-CTRL-01`

- **Libraries**: `asyncio`, `streamlit==1.47.1`

- **Description**: Connect the refactored scraper to the UI, allowing users to trigger scrapes and see a rich, real-time progress dashboard with calculated metrics.

- **Developer Context**: This task combines the original background task integration with the enhanced dashboard from the V1.1 plan, as they are functionally codependent.

- **Sub-tasks & Instructions**:
  - **T1.2.1: Implement `BackgroundTaskManager`**: ✅ **COMPLETED**
    - **Instructions**: Create `src/ui/utils/background_tasks.py`. Implement the `BackgroundTaskManager` and `StreamlitTaskManager` classes. The `update_progress` callback is the key communication mechanism. Ensure the progress data structure includes `start_time` for each company.
    - **Success Criteria**: The utility is created and can launch the `scrape_all` function from `src/scraper.py` in a separate thread or async task.
  - **T1.2.2: Create Progress Formatting Utilities**: ✅ **COMPLETED**
    - **Instructions**: Create a new file `src/ui/utils/formatters.py`. Inside, create utility functions: `calculate_scraping_speed(jobs_found, start_time, end_time)` which returns jobs/minute, and `calculate_eta(total_companies, completed_companies, time_elapsed)` which returns a formatted time string.
    - **Success Criteria**: The utility functions correctly calculate and format the required metrics.
  - **T1.2.3: Create the Company Progress Card Component**: ✅ **COMPLETED**
    - **Instructions**: Create `src/ui/components/progress/company_progress_card.py`. This component will take a company's progress info as input.
    - **Instructions**: Inside the card (using `st.container` with a border), display the company name, a `st.progress` bar, and use `st.metric` to display "Jobs Found" and "Scraping Speed".
    - **Success Criteria**: A reusable card component is created that can visually represent the progress of a single company.
  - **T1.2.4: Build the Enhanced Scraping Page UI**: ✅ **COMPLETED**
    - **Instructions**: In `src/ui/pages/scraping.py`, create the UI with a "Start Scraping" button that calls the `StreamlitTaskManager`.
    - **Instructions**: In the progress section, use `st.columns` to create a responsive grid. In a loop, instantiate and render your new `CompanyProgressCard` for each company in `st.session_state.progress_data`.
    - **Instructions**: Above the grid, add `st.metric` displays for the overall "ETA" and "Total Jobs Found", calculated using the new formatter utilities.
    - **Success Criteria**: The user can start a scrape and see a grid of professional-looking progress cards. The metrics for speed and ETA update in real-time during a scraping session.

### **T1.3: Implement Essential Company & Settings Management**

- **Release**: V1.0

- **Priority**: **Medium**

- **Status**: **✅ COMPLETED** (PR #26 & PR #28)

- **Prerequisites**: `T0.1`, `T0.2`

- **Requirements File**: `docs/planning/v1.0.0/REQUIREMENTS_V1.0.md`

- **Related Requirements**: `UI-COMP-01`, `UI-COMP-02`, `UI-SETT-01`, `SYS-ARCH-05`

- **Libraries**: `streamlit==1.47.1`

- **Description**: Build the final administrative UIs required for the application to be configurable and usable, incorporating specific, high-value controls from the UI requirements.

- **Developer Context**: This involves building out the `companies.py` and `settings.py` pages within the new architecture.

- **Sub-tasks & Instructions**:
  - **T1.3.1: Build Company Management Page**: ✅ **COMPLETED**
    - **Instructions**: In `src/ui/pages/companies.py`, use an `st.expander` to house a form for adding a new company.
    - **Instructions**: Below the form, fetch all companies using a new `CompanyService` and display them in a list or grid. Each company listed must have a visible `st.toggle` to control its `active` status, fulfilling `UI-COMP-02`.
    - **Success Criteria**: Users can add new companies to be scraped and can enable or disable scraping for existing companies via a toggle.
  - **T1.3.2: Build Essential Settings Page**: ✅ **COMPLETED**
    - **Instructions**: In `src/ui/pages/settings.py`, build the settings UI. It **must** include:
            1. `st.text_input(type="password")` for API key management with a "Test Connection" button.
            2. A `st.toggle` or `st.radio` for the **"LLM Provider Toggle (OpenAI ↔ Groq)"**.
            3. A `st.slider` for the **"Max jobs per company"** limit to prevent runaway scraping.
    - **Success Criteria**: The user can configure API keys, switch between LLM providers, and set a job limit per company.

---

## ❗ V1.0 QA Findings & Final Tasks

The following tasks have been identified during the final QA review and must be completed for the v1.0 release.

### **T1.4: Implement Database Migration Strategy**

- **Release**: V1.0

- **Priority**: **CRITICAL**

- **Status**: **✅ COMPLETED** (PR #30)

- **Prerequisites**: `T0.1` (Database Schema)

- **Related Requirements**: `NFR-MAINT-02`

- **Libraries**: `alembic`

- **Description**: This is a **release-blocking** task. The application currently lacks a database migration system, which means any future schema change would result in complete data loss for existing users. This task implements Alembic to manage schema changes non-destructively, as specified in ADR-012.

- **Sub-tasks & Instructions**:
  - **T1.4.1: Install and Configure Alembic**: ✅ **COMPLETED**
    - **Instructions**: Add `alembic` as a dependency in `pyproject.toml`.
    - **Instructions**: Run `alembic init alembic` to create the migration environment.
    - **Instructions**: Configure `alembic/env.py` to connect to the application's database and target the `SQLModel.metadata` from `src/models.py`.
    - **Success Criteria**: Alembic is correctly configured and can connect to the application's database.
  - **T1.4.2: Generate Initial Migration Script**: ✅ **COMPLETED**
    - **Instructions**: Run `alembic revision --autogenerate -m "Initial schema"` to create the first migration script that reflects the current database schema.
    - **Instructions**: Review the generated script to ensure it accurately represents all tables, columns, and constraints from `src/models.py`.
    - **Success Criteria**: A migration script exists in `alembic/versions/` that can create the entire database from scratch.
  - **T1.4.3: Integrate Migration into Application Startup**: ✅ **COMPLETED**
    - **Instructions**: In `src/main.py` or a startup script, add logic to programmatically run `alembic upgrade head`. This will ensure that any pending migrations are automatically applied when the application starts.
    - **Success Criteria**: When the application is launched with an existing database, it automatically applies any new migrations.

### **T1.5: Complete Advanced Job Filtering**

- **Release**: V1.0

- **Priority**: **High**

- **Status**: **✅ COMPLETED** (PR #31)

- **Prerequisites**: `T1.1`

- **Related Requirements**: `UI-JOBS-04`

- **Libraries**: `streamlit`

- **Description**: The v1.0 requirements mandate filtering by salary range, but this feature is currently missing. This task implements the final piece of the advanced filtering functionality.

- **Sub-tasks & Instructions**:
  - **T1.5.1: Update `JobService` for Salary Queries**: ✅ **COMPLETED**
    - **Instructions**: Open `src/services/job_service.py`. Modify the `get_filtered_jobs` method to accept `salary_min: int | None` and `salary_max: int | None`.
    - **Instructions**: Add logic to the SQLAlchemy query to filter based on the salary range. Since the `salary` column is JSON `(min, max)`, you will need to use `func.json_extract` to query the values.
    - **Success Criteria**: The service layer can now filter jobs based on a minimum and maximum salary.
    - **Implementation Notes**: Enhanced to support high-value jobs up to and beyond $1M/year. When salary_max is set to 750000, it acts as unbounded to include all jobs >= $750k.
  - **T1.5.2: Add Salary Slider to Filter UI**: ✅ **COMPLETED**
    - **Instructions**: Open `src/ui/components/sidebar.py`. Add a `st.slider` widget with two thumbs to allow the user to select a min/max salary range.
    - **Instructions**: Pass the values from this new slider to the `JobService.get_filtered_jobs` call.
    - **Success Criteria**: The UI now includes a salary range slider that correctly filters the job results.
    - **Implementation Notes**: Implemented with $0-$750k range, $25k increments, and "750k+" unbounded mode. Professional formatting with k/M notation and clear visual indicators.

### **T1.6: Refactor Data Logic and Optimize UI**

- **Release**: V1.0

- **Priority**: **Medium**

- **Status**: **✅ COMPLETED** (PR #32)

- **Prerequisites**: `T1.1`, `T1.3`

- **Related Requirements**: `NFR-MAINT-01`, `NFR-PERF-01`

- **Libraries**: N/A (Refactoring)

- **Description**: This task groups the remaining architectural and performance improvements identified during the QA review to enhance maintainability and UI responsiveness.

- **Sub-tasks & Instructions**:
  - **T1.6.1: Relocate `bulk_get_or_create_companies`**: ✅ **COMPLETED**
    - **Instructions**: Move the `bulk_get_or_create_companies` function from `src/scraper.py` to `src/services/company_service.py`.
    - **Instructions**: Update the import and function call within `src/scraper.py` to reflect the new location.
    - **Success Criteria**: The scraper module is now purely for orchestration, and all company-related database logic resides in the `CompanyService`.
  - **T1.6.2: Optimize UI Tab Filtering**: ✅ **COMPLETED**
    - **Instructions**: Modify `JobService.get_filtered_jobs` to accept `favorites_only: bool` and `application_status: list` as parameters to push filtering to the database.
    - **Instructions**: In `src/ui/pages/jobs.py`, update the "Favorites" and "Applied" tabs to call the service with these new parameters instead of filtering the full job list in Python.
    - **Success Criteria**: The UI remains functionally identical, but performance is significantly improved for large datasets as filtering now happens in the database.
  - **T1.6.3: Centralize Application Status Constants**: ✅ **COMPLETED**
    - **Instructions**: In `src/constants.py`, define a new constant: `APPLICATION_STATUSES = ["New", "Interested", "Applied", "Rejected"]`.
    - **Instructions**: In `src/ui/components/cards/job_card.py`, import and use this constant for the `st.selectbox` options, removing the hardcoded list.
    - **Success Criteria**: The application statuses are now managed from a single source of truth, reducing the risk of inconsistency.
