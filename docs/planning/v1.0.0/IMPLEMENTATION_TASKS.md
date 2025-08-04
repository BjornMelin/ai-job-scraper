# AI Job Scraper - Detailed Implementation Plan

## Introduction

This document contains the full, prioritized list of tasks required to build the AI Job Scraper application. Each task is designed to be a self-contained unit of work with clear instructions, dependencies, and success criteria. Follow the tasks in the specified order to ensure a smooth and efficient development process.

---

## 🚀 V1.0: The "Job Hunter's MVP" Release

**Goal**: To build the essential, high-impact features that will allow you to start using the application for your job hunt as quickly as possible.

### **T1.1: Foundational Architecture Setup**

- **Release**: V1.0
- **Priority**: **Critical**
- **Related Requirements**: `SYS-ARCH-01`, `SYS-ARCH-02`, `SYS-ARCH-03`, `NFR-MAINT-01`
- **Libraries**: `streamlit==1.47.1`
- **Description**: This task establishes the core project structure, state management, and navigation system. It is the bedrock upon which all other features will be built.

- **Architecture Diagram**:

  ```mermaid
  graph TD
      subgraph "src"
          A[main.py] --> B[ui/pages]
          A --> C[ui/state]
          A --> D[ui/styles]
          B --> E[dashboard.py]
          B --> F[jobs.py]
      end
  ```

- **Sub-tasks & Instructions**:
  - **T1.1.1: Create Directory Structure**:
    - **Instructions**: In your project root, execute the command: `mkdir -p src/ui/{pages,components,state,styles,utils}` and `mkdir -p src/ui/components/{cards,forms,layouts}`.
    - **Success Criteria**: The specified directories and subdirectories exist in the project root.
  - **T1.1.2: Implement Centralized State Manager**:
    - **Instructions**: Create the file `src/ui/state/app_state.py`. Implement the `AppState` dataclass and `StateManager` singleton class exactly as defined in the planning document `07-next-steps.md`.
    - **Success Criteria**: The `StateManager.get_state()` method can be called from other parts of the app to retrieve a consistent state object.
  - **T1.1.3: Implement Custom CSS Theme System**:
    - **Instructions**: Create `src/ui/styles/__init__.py`. Implement the `load_custom_styles` function and copy the full CSS content from `07-next-steps.md`. This includes the Inter font import and CSS variables for theming.
    - **Success Criteria**: The CSS is loaded by the main app, and the font/colors are visibly different from the default Streamlit theme.
  - **T1.1.4: Create Main Application Entrypoint & Navigation**:
    - **Instructions**: Create `src/ui/main.py`. Implement the main application logic that initializes the page config, loads styles, gets state, and handles page routing via `st.sidebar.radio`. Create placeholder files in `src/ui/pages/` for each view (`dashboard.py`, `jobs.py`, etc.), each containing a simple `render()` function with an `st.title()`.
    - **Success Criteria**: The application runs via `streamlit run src/ui/main.py`. The sidebar navigation is present and clicking on different pages correctly renders the corresponding title.

### **T1.2: Core Database Models & Relationships**

- **Release**: V1.0
- **Priority**: **Critical**
- **Prerequisites**: `T1.1`
- **Related Requirements**: `DB-SCHEMA-01`, `DB-SCHEMA-02`, `DB-SCHEMA-03`
- **Libraries**: `sqlmodel==0.0.19`, `sqlalchemy==2.0.42`
- **Description**: Define the core database schema using SQLModel. This task is critical as it structures the data that the entire application will rely on.

- **Schema Diagram**:

  ```mermaid
  erDiagram
      CompanySQL {
          int id PK
          string name
          string url
          datetime last_scraped
          int scrape_count
          float success_rate
      }
      JobSQL {
          int id PK
          int company_id FK
          string title
          string description
          string link
          string content_hash
          bool favorite
          string notes
          string application_status
          datetime application_date
      }
      CompanySQL ||--o{ JobSQL : "has"
  ```

- **Sub-tasks & Instructions**:
  - **T1.2.1: Define SQLModel Classes**:
    - **Instructions**: In `src/models.py`, define the `CompanySQL` and `JobSQL` classes using `SQLModel`. Ensure all fields specified in the diagram and `DB-SCHEMA-02`/`DB-SCHEMA-03` are included with correct types.
    - **Instructions**: Crucially, define the relationship by adding `company_id: int = Field(foreign_key="companysql.id")` to `JobSQL`.
    - **Success Criteria**: The Python models are defined without syntax errors.
  - **T1.2.2: Implement Database Engine and Session Logic**:
    - **Instructions**: In `src/database.py`, create the SQLite engine (`create_engine`) and the function to create all tables (`SQLModel.metadata.create_all`). Implement a `get_session` context manager to provide database sessions to services.
    - **Success Criteria**: Running a setup script successfully creates a `database.db` file with the `companysql` and `jobsql` tables.

### **T1.3: Smart Sync Engine Implementation**

- **Release**: V1.0
- **Priority**: **High**
- **Prerequisites**: `T1.2`
- **Related Requirements**: `DB-SYNC-01`, `DB-SYNC-02`, `DB-SYNC-03`, `DB-SYNC-04`
- **Libraries**: `hashlib`, `json`
- **Description**: Build the intelligent data synchronization engine. This is a core backend feature that prevents data duplication and ensures data integrity over time.

- **Sub-tasks & Instructions**:
  - **T1.3.1: Implement Content Hashing Utility**:
    - **Instructions**: In a new `src/services/utils.py` file, create a `generate_content_hash(job_data: dict) -> str` function. It should concatenate key fields (title, location, a snippet of the description) into a JSON string and return its MD5 hexdigest.
    - **Success Criteria**: The function consistently produces the same hash for the same input data.
  - **T1.3.2: Build the SmartSyncEngine Class**:
    - **Instructions**: Create `src/services/database_sync.py` and implement the `SmartSyncEngine` class as detailed in `03-database-optimization.md`.
    - **Instructions**: The `_analyze_sync_operations` method should be the core logic hub, comparing incoming job hashes to existing ones and categorizing them into `inserts`, `updates`, `touches`, and `deletes`.
    - **Instructions**: The `_execute_sync_operations` method must handle the database transactions. For updates, it must fetch the existing job, preserve user fields (`favorite`, `notes`, `application_status`), and then apply the new data. For deletes, it must check for user data and perform a soft delete (set `archived = True`) if present.
    - **Success Criteria**: Given a list of scraped jobs and a database session, the engine correctly identifies and prepares the required database operations.
  - **T1.3.3: Integrate Engine into Scraper Service**:
    - **Instructions**: In your main scraper service, after scraping jobs for a company, instantiate `SmartSyncEngine` and call `sync_company_jobs` instead of performing manual inserts.
    - **Success Criteria**: A full scraping run completes using the new engine, and the database state is correct (no duplicates, updates applied, old jobs archived/deleted).

### **T1.4: Background Scraping & Real-Time Progress UI**

- **Release**: V1.0
- **Priority**: **High**
- **Prerequisites**: `T1.1`
- **Related Requirements**: `SYS-ARCH-04`, `SCR-EXEC-01`, `SCR-PROG-01`, `SCR-CTRL-01`, `UI-PROG-01`
- **Libraries**: `asyncio`
- **Description**: Implement the system for running the scraping process in the background and displaying its progress to the user in real-time.

- **Sub-tasks & Instructions**:
  - **T1.4.1: Implement Background Task Manager**:
    - **Instructions**: Create `src/ui/utils/background_tasks.py` and implement the `BackgroundTaskManager` and `StreamlitTaskManager` classes as defined in `02-technical-architecture.md`. The key is the `update_progress` callback within `StreamlitTaskManager` that modifies `st.session_state` and calls `st.rerun()`.
    - **Success Criteria**: The `start_background_scraping` method can be called and it successfully launches an `asyncio` task.
  - **T1.4.2: Build the Scraping Page UI**:
    - **Instructions**: In `src/ui/pages/scraping.py`, build the UI. Add a `st.multiselect` for company selection and a "Start Scraping" button. When this button is clicked, call `StreamlitTaskManager.start_background_scraping`.
    - **Instructions**: Below the controls, add a section that only renders if `st.session_state.get("scraping_active", False)`. In this section, display an overall `st.progress` bar. Then, iterate through `st.session_state.progress_data.items()` and for each company, display its name and status (e.g., `st.text(f"{company_name}: {progress_info.status}")`).
    - **Success Criteria**: The UI is present. Clicking "Start" makes the progress section appear and disables the "Start" button.
  - **T1.4.3: Connect Scraper Logic to Progress Callback**:
    - **Instructions**: In the `_execute_scraping_workflow` method of your `BackgroundTaskManager`, ensure you call the `progress_callback` at key points: before starting a company, after finishing a company, and if an error occurs.
    - **Success Criteria**: While the background task runs, the statuses on the Scraping page update in real-time from "Queued" to "Scraping" to "Done" or "Error".

### **T1.5: Core Job Browser & Application Tracking**

- **Release**: V1.0
- **Priority**: **High**
- **Prerequisites**: `T1.2`, `T1.3`
- **Related Requirements**: `UI-JOBS-01`, `UI-JOBS-02`, `UI-JOBS-03`, `UI-JOBS-04`, `UI-TRACK-01`
- **Libraries**: `streamlit==1.47.1`
- **Description**: Build the primary user interface for browsing, filtering, and managing jobs. This is the most important page in the application.

- **Sub-tasks & Instructions**:
  - **T1.5.1: Implement Job Service**:
    - **Instructions**: Create `src/services/job_service.py`. Implement the `get_filtered_jobs` method as defined in `07-next-steps.md`, ensuring it can filter by text, company IDs, and application status. Also include `toggle_favorite` and `update_application_status` methods.
    - **Success Criteria**: The service methods correctly query the database and return the expected job lists.
  - **T1.5.2: Build the Filter Panel**:
    - **Instructions**: In `src/ui/pages/jobs.py`, use `st.sidebar` to house the filter controls. Add a `st.text_input` for search, a `st.multiselect` for companies, and a `st.multiselect` for application statuses. Store the selected filter values in `st.session_state`.
    - **Success Criteria**: The filter controls are visible in the sidebar and their values are correctly managed in the session state.
  - **T1.5.3: Build the Interactive Job Card & Grid**:
    - **Instructions**: Create `src/ui/components/cards/job_card.py`. The card should be wrapped in an `st.container` with a border. It must contain: `st.markdown` for title/company, a `st.button` for the favorite toggle (using a heart icon), and a `st.selectbox` for the application status. Finally, add a `st.button("View Details")`.
    - **Instructions**: In `src/ui/pages/jobs.py`, fetch the filtered jobs from `JobService` and render them in a responsive grid using `st.columns`.
    - **Success Criteria**: A grid of jobs is displayed. Interacting with the favorite button or status selectbox on a card updates the job in the database and the UI reflects the change on the next rerun.
  - **T1.5.4: Implement "Expander" Job Details**:
    - **Instructions**: Modify the job grid rendering loop. If the "View Details" button for a specific job is clicked, store its ID in `st.session_state.expanded_job_id`. In the next rerun, if a job's ID matches the one in the session state, render an `st.expander("Details")` directly below its card. Inside the expander, display the full `job.description` and a `st.text_area` for `job.notes`.
    - **Success Criteria**: Clicking "View Details" reveals the job description and notes area below the card without a full page reload. Editing notes and clicking away saves the notes.

### **T1.6: Essential Company & Settings Management**

- **Release**: V1.0
- **Priority**: **Medium**
- **Prerequisites**: `T1.1`
- **Related Requirements**: `UI-COMP-01`, `UI-COMP-02`, `UI-SETT-01`
- **Libraries**: `streamlit==1.47.1`, `validators==0.33.0`
- **Description**: Build the final pieces of the core UI needed for configuration and management.

- **Sub-tasks & Instructions**:
  - **T1.6.1: Implement Company Management Page**:
    - **Instructions**: Build out `src/ui/pages/companies.py`. Use an `st.expander` to house the "Add Company" form. Render the list of existing companies using a `CompanyCard` component that displays the company name and an `st.toggle` for the `active` status.
    - **Success Criteria**: Users can add new companies, and activate/deactivate existing ones.
  - **T1.6.2: Implement Settings Page**:
    - **Instructions**: Build out `src/ui/pages/settings.py`. Use `st.tabs` to create sections. In the "API" tab, add `st.text_input` fields for the OpenAI and Groq API keys (with `type="password"`).
    - **Success Criteria**: Users can enter and save their API keys, which are then available for the scraper service to use.
