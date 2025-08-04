# AI Job Scraper - Detailed Implementation Plan (V1.0 - Job Hunter's MVP)

## Introduction

This document outlines the tasks for the **V1.0 "Job Hunter's MVP" Release**. It assumes that all tasks in the **V0.0 Foundational Refactoring** phase have been completed. The application now has a stable architecture, a correct database schema, and a data-safe synchronization engine. The goal of this phase is to build the essential user-facing features on top of this solid foundation, incorporating key UI requirements for immediate value.

---

## 🚀 V1.0: The "Job Hunter's MVP" Release

### **T1.1: Implement Core Job Browser & Application Tracking**

- **Release**: V1.0
- **Priority**: **High**
- **Status**: **PENDING**
- **Prerequisites**: `T0.1`, `T0.2`, `T0.3`
- **Requirements File**: `docs/planning/v1.0.0/REQUIREMENTS_V1.0.md`
- **Related Requirements**: `UI-JOBS-01`, `UI-JOBS-02`, `UI-JOBS-03`, `UI-JOBS-04`, `UI-TRACK-01`
- **Libraries**: `streamlit==1.47.1`
- **Description**: Build the primary user interface for browsing, filtering, and managing jobs within the newly established component architecture.
- **Developer Context**: All work for this task will happen within the `src/ui/` directory. You will be creating new services and components that interact with the database models and state manager created in Phase 0.

- **Sub-tasks & Instructions**:
  - **T1.1.1: Implement Job Service**:
    - **Instructions**: In the `src/services/` directory, create a new file `job_service.py`.
    - **Instructions**: Create a `JobService` class with static methods. Implement `get_filtered_jobs(filters: dict)`. This method will query the database using the synchronous session from `database.py` and apply filters for text search, company, and application status.
    - **Instructions**: Implement `update_job_status(job_id: int, status: str)`, `toggle_favorite(job_id: int)`, and `update_notes(job_id: int, notes: str)`. These methods will update a single job record in the database.
    - **Success Criteria**: The service provides a clean API for the UI to fetch and update job data without containing any UI code itself.
  - **T1.1.2: Build Interactive Job Card Component**:
    - **Instructions**: In `src/ui/components/cards/`, create `job_card.py`.
    - **Instructions**: The `render_job_card(job: JobSQL)` function should accept a `JobSQL` object. Inside, use `st.container` with a border.
    - **Instructions**: Add a `st.selectbox` for the `application_status` field. Its `on_change` parameter must point to a callback function that calls `JobService.update_job_status`.
    - **Instructions**: Add a `st.button` for the favorite toggle (using a heart icon: ❤️/🤍). Its `on_click` must call `JobService.toggle_favorite`.
    - **Instructions**: Add a `st.button("View Details")`. Its `on_click` should update a session state variable, e.g., `st.session_state.expanded_job_id = job.id`.
    - **Success Criteria**: The job card correctly displays job data and its interactive widgets successfully update the database via the `JobService`.
  - **T1.1.3: Build Job Details Expander**:
    - **Instructions**: In the rendering loop within `src/ui/pages/jobs.py`, check if a job's ID matches `st.session_state.get('expanded_job_id')`.
    - **Instructions**: If it matches, render an `st.expander("Details")` directly below its card. Inside the expander, display `st.markdown(job.description)` and a `st.text_area("Notes", value=job.notes)`. The text area's `on_change` callback should save the notes to the database via the `JobService.update_notes` method.
    - **Success Criteria**: Clicking "View Details" reveals the job description and notes area below the card. Editing notes persists the changes.

### **T1.2: Implement Background Scraping Integration**

- **Release**: V1.0
- **Priority**: **High**
- **Status**: **PENDING**
- **Prerequisites**: `T0.3`
- **Requirements File**: `docs/planning/v1.0.0/REQUIREMENTS_V1.0.md`
- **Related Requirements**: `SYS-ARCH-04`, `SCR-PROG-01`, `UI-PROG-01`, `SCR-CTRL-01`
- **Libraries**: `asyncio`
- **Description**: Connect the refactored, data-safe scraper from Phase 0 to the user interface, allowing users to trigger scrapes and see real-time progress.
- **Developer Context**: This task involves creating the background task utility and building the UI on the `scraping.py` page.

- **Sub-tasks & Instructions**:
  - **T1.2.1: Implement `BackgroundTaskManager`**:
    - **Instructions**: Create `src/ui/utils/background_tasks.py`. Implement the `BackgroundTaskManager` and `StreamlitTaskManager` classes as defined in `02-technical-architecture.md`. The `update_progress` callback in `StreamlitTaskManager` is the key mechanism for communicating with the UI.
    - **Success Criteria**: The utility is created and can launch the `scrape_all` function from `src/scraper.py` in a separate thread or async task.
  - **T1.2.2: Build the Scraping Page UI**:
    - **Instructions**: In `src/ui/pages/scraping.py`, create the UI with a "Start Scraping" button. When this button is clicked, call `StreamlitTaskManager.start_background_scraping`.
    - **Instructions**: Add a UI section that is only visible when `st.session_state.get("scraping_active", False)`. Inside this section, display an overall `st.progress` bar. Iterate through `st.session_state.progress_data` and display the status of each company being scraped using simple `st.text` elements (e.g., `st.text(f"{company_name}: {progress_info.status}")`).
    - **Success Criteria**: The user can start a scrape from the UI. The UI remains responsive, and the progress section updates in real-time to show which part of the scraping process is currently running.

### **T1.3: Implement Essential Company & Settings Management**

- **Release**: V1.0
- **Priority**: **Medium**
- **Status**: **PENDING**
- **Prerequisites**: `T0.1`, `T0.2`
- **Requirements File**: `docs/planning/v1.0.0/REQUIREMENTS_V1.0.md`
- **Related Requirements**: `UI-COMP-01`, `UI-COMP-02`, `UI-SETT-01`, `SYS-ARCH-05`
- **Libraries**: `streamlit==1.47.1`
- **Description**: Build the final administrative UIs required for the application to be configurable and usable, incorporating specific, high-value controls from the UI requirements.
- **Developer Context**: This involves building out the `companies.py` and `settings.py` pages within the new architecture.

- **Sub-tasks & Instructions**:
  - **T1.3.1: Build Company Management Page**:
    - **Instructions**: In `src/ui/pages/companies.py`, use an `st.expander` to house a form for adding a new company.
    - **Instructions**: Below the form, fetch all companies using a new `CompanyService` and display them in a list or grid. Each company listed must have a visible `st.toggle` to control its `active` status, fulfilling `UI-COMP-02`.
    - **Success Criteria**: Users can add new companies to be scraped and can enable or disable scraping for existing companies via a toggle.
  - **T1.3.2: Build Essential Settings Page**:
    - **Instructions**: In `src/ui/pages/settings.py`, build the settings UI. It **must** include:
            1. `st.text_input(type="password")` for API key management with a "Test Connection" button.
            2. A `st.toggle` or `st.radio` for the **"LLM Provider Toggle (OpenAI ↔ Groq)"**.
            3. A `st.slider` for the **"Max jobs per company"** limit to prevent runaway scraping.
    - **Success Criteria**: The user can configure API keys, switch between LLM providers, and set a job limit per company.
