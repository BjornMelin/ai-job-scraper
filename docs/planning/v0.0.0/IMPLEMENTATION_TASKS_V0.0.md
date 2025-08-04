# AI Job Scraper - Detailed Implementation Plan (V0.0 - Foundational Refactoring)

## Introduction

This document outlines the critical, prerequisite tasks for **Phase 0: Foundational Refactoring & Stabilization**. The primary goal of this phase is to address the architectural and data integrity issues identified in the existing codebase. **All tasks in this document must be completed before beginning any work on V1.0 features.** This phase transforms the proof-of-concept into a stable, maintainable platform ready for future development.

---

## ⚙️ Phase 0: Foundational Refactoring & Stabilization

### **T0.1: Standardize Database and Refactor Models**

- **Release**: V0.0 (Prerequisite for V1.0)
- **Priority**: **CRITICAL**
- **Status**: **DONE**
- **Requirements File**: `docs/planning/v0.0.0/REQUIREMENTS_V0.0.md`
- **Related Requirements**: `DB-SCHEMA-01`, `DB-SCHEMA-02`, `DB-SCHEMA-03`, `NFR-MAINT-01`
- **Libraries**: `sqlmodel==0.0.24`, `sqlalchemy==2.0.42`
- **Description**: This task unifies database access to a standard synchronous model, which is more stable and less complex within the Streamlit environment. It also upgrades the database models to support all planned features, fixing a critical data structure gap.
- **Developer Context**: This task replaces the `async` engine in `src/database.py` and significantly modifies `src/models.py`. It will require deleting the old `jobs.db` file to allow for the creation of a new, correctly structured database.

- **Sub-tasks & Instructions**:
  - **T0.1.1: Convert to Synchronous Database Engine**:
    - **Instructions**: Open `src/database.py`. Remove all `asyncio`, `aiosqlite`, and `Async...` related imports and code.
    - **Instructions**: Replace the entire file content with a standard synchronous SQLAlchemy setup:

          ```python
          from sqlalchemy import create_engine
          from sqlalchemy.orm import sessionmaker
          from sqlmodel import SQLModel
          from .config import Settings

          settings = Settings()
          engine = create_engine(settings.db_url) # e.g., "sqlite:///jobs.db"
          SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

          def create_db_and_tables():
              SQLModel.metadata.create_all(engine)
          ```

    - **Success Criteria**: The application can connect to the database using a standard synchronous SQLAlchemy engine. The `nest_asyncio` dependency can be removed from `app.py` later.
  - **T0.1.2: Upgrade `JobSQL` and `CompanySQL` Models**:
    - **Instructions**: Open `src/models.py`. This is a critical data structure change.
    - **Instructions**: In the `JobSQL` class, **remove the `company: str` field**. Add the following new fields:

          ```python
          company_id: int | None = Field(default=None, foreign_key="companysql.id")
          content_hash: str = Field(index=True)
          application_status: str = Field(default="New", index=True)
          application_date: datetime | None = None
          archived: bool = Field(default=False, index=True)
          # The existing 'favorite' and 'notes' fields are correct.
          ```

    - **Instructions**: In the `CompanySQL` class, add the following new fields for tracking scraping metrics:

          ```python
          last_scraped: datetime | None = None
          scrape_count: int = Field(default=0)
          success_rate: float = Field(default=1.0)
          ```

    - **Instructions**: After making these changes, **delete the old `jobs.db` file**. A new one will be created with the correct schema when the application or a setup script runs `create_db_and_tables()`.
    - **Success Criteria**: The new `jobs.db` file is created with the `jobsql` and `companysql` tables containing all the new fields and the correct foreign key relationship.

### **T0.2: Architect the UI - Deconstruct `app.py`**

- **Release**: V0.0 (Prerequisite for V1.0)
- **Priority**: **CRITICAL**
- **Status**: **IN PROGRESS**
- **Requirements File**: `docs/planning/v0.0.0/REQUIREMENTS_V0.0.md`
- **Related Requirements**: `SYS-ARCH-01`, `SYS-ARCH-02`, `SYS-ARCH-03`
- **Libraries**: `streamlit==1.47.1`
- **Description**: This task implements the planned component-based architecture by deconstructing the monolithic `app.py` file into a modular and maintainable structure.
- **Developer Context**: This is a pure refactoring task. The goal is to achieve the same UI functionality as the old `app.py` but with code organized into the new `src/ui/` directory structure.

- **Architecture Diagram**:

  ```mermaid
  graph TD
      A[src/main.py] -- Manages --> B(Page Routing);
      B -- Renders --> C[src/ui/pages/jobs.py];
      C -- Uses --> D[src/ui/components/sidebar.py];
      C -- Uses --> E[src/ui/components/cards/job_card.py];
      A -- Initializes --> F[src/ui/state/app_state.py];
  ```

- **Sub-tasks & Instructions**:
  - **T0.2.1: Execute Foundational Architecture Setup**:
    - **Instructions**: Create the directory structure: `mkdir -p src/ui/{pages,components,state,styles,utils}` and subdirectories for `components/cards` and `components/layouts`.
    - **Instructions**: Create `src/ui/state/app_state.py` with the `StateManager` singleton.
    - **Instructions**: Create `src/ui/styles/theme.py` (or similar) to hold the CSS content from `static/css/main.css` and any future styles.
    - **Instructions**: Create a new entrypoint file, `src/main.py`, which will handle page config, style loading, and sidebar navigation.
    - **Success Criteria**: A runnable, empty, multi-page application structure exists.
  - **T0.2.2: Migrate UI Logic into Components**:
    - **Instructions**: This is a methodical copy-paste-and-refactor process.
    - **Step 1 (Sidebar)**: Create `src/ui/components/sidebar.py`. Move all the code from the `with st.sidebar:` block in the old `app.py` into a `render_sidebar()` function in this new file.
    - **Step 2 (Job Display)**: Create `src/ui/pages/jobs.py`. This will be the new main UI file. Move the main content area logic from `app.py` here, including the tab creation (`st.tabs`) and the job display loops.
    - **Step 3 (Job Card)**: Create `src/ui/components/cards/job_card.py`. Move the code that renders a single job card (the `st.markdown` block with HTML) into a `render_job_card(job)` function. The `jobs.py` page will call this function inside its loop.
    - **Step 4 (Orchestration)**: In `src/main.py`, call `render_sidebar()`. The sidebar will control which page from `src/ui/pages/` gets rendered in the main content area.
    - **Success Criteria**: The old `app.py` file is now deleted. The application, when run from `src/main.py`, looks and functions identically to the original version, but the code is now cleanly organized into the new `src/ui/` structure.

### **T0.3: Refactor Scraping Workflow & Implement Smart Sync**

- **Release**: V0.0 (Prerequisite for V1.0)
- **Priority**: **CRITICAL**
- **Status**: **PENDING**
- **Requirements File**: `docs/planning/v0.0.0/REQUIREMENTS_V0.0.md`
- **Related Requirements**: `DB-SYNC-01`, `DB-SYNC-02`, `DB-SYNC-03`, `DB-SYNC-04`
- **Libraries**: `hashlib`, `json`
- **Description**: Decouple the data extraction logic from the database writing logic. Replace the old, destructive `update_db` function with the robust `SmartSyncEngine`.
- **Developer Context**: This task fundamentally changes how data is persisted. It involves modifying all scraper files and creating a new, central service for data synchronization.

- **Sub-tasks & Instructions**:
  - **T0.3.1: Modify Scrapers to Return Data (Decoupling)**:
    - **Instructions**: Open `src/scraper_company_pages.py`. Find the `save_jobs` node in the `LangGraph` workflow. Remove the database session logic from it. Modify it to simply retrieve the `normalized_jobs` from the state and return them. The graph's final output will now be a list of `JobSQL` objects.
    - **Instructions**: Open `src/scraper_job_boards.py`. Ensure the `scrape_job_boards` function returns a list of normalized `JobSQL` objects, not a DataFrame.
    - **Success Criteria**: The scraper modules no longer have any direct database write access. They are now pure data extractors.
  - **T0.3.2: Implement the `SmartSyncEngine`**:
    - **Instructions**: Create a new file `src/services/database_sync.py`.
    - **Instructions**: Implement the `SmartSyncEngine` class as detailed in `03-database-optimization.md`. It must have a primary method like `sync_jobs(jobs: list[JobSQL])`.
    - **Instructions**: The engine's logic must correctly identify jobs to insert, update, or archive based on `content_hash` and the presence of user data, exactly as specified in `DB-SYNC-03` and `DB-SYNC-04`.
    - **Success Criteria**: The `SmartSyncEngine` is a self-contained service that can take a list of jobs and intelligently persist them to the database without data loss.
  - **T0.3.3: Create the Master Scraper Orchestrator**:
    - **Instructions**: Open `src/scraper.py`. Rewrite the `scrape_all` function.
    - **Instructions**: It should first call the company page workflow and the job board scraper to collect two lists of `JobSQL` objects.
    - **Instructions**: Combine these lists and perform a final deduplication based on the `link` field.
    - **Instructions**: Instantiate the `SmartSyncEngine` and pass the final, clean list of jobs to its `sync_jobs` method. **Remove the old `update_db` function entirely.**
    - **Success Criteria**: Running the main `scrape_all` function now correctly uses the new, data-safe `SmartSyncEngine` to update the database. User-entered data is preserved across scrapes.
