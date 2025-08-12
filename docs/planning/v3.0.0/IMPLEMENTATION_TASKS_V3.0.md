# AI Job Scraper - Detailed Implementation Plan (V2.0)

## Introduction

This plan details the tasks for the **V2.0 "Polished & Professional" Release**. It is the final phase of development, focusing on robustness, user experience refinement, and long-term maintainability. It assumes a fully functional V1.1 application is the starting point.

---

## ✨ V2.0: The "Polished & Professional" Release

**Goal**: To finalize the application by implementing a comprehensive test suite, adding a final layer of UI polish, and creating thorough documentation.

### **T3.1: Implement Comprehensive Testing Suite**

- **Release**: V2.0
- **Priority**: **Critical**
- **Status**: **PENDING**
- **Prerequisites**: A functional V1.1 application.
- **Related Requirements**: `NFR-TEST-01`, `NFR-TEST-02`, `NFR-TEST-03`
- **Libraries**: `pytest==8.2.2`, `pytest-mock==3.14.0`, `pytest-asyncio==0.23.7`
- **Description**: This is the most critical task for V2.0. It involves building a robust, automated test suite to ensure the application is reliable and to prevent regressions in the future.

- **Architecture Diagram**:

  ```mermaid
  graph TD
      A[Test Suite] --> B[Unit Tests];
      A --> C[Integration Tests];
      
      subgraph "Codebase"
          D[Services];
          E[Utilities];
          F[Database Models];
      end

      B --> D;
      B --> E;
      C --> F;
      C --> D;
  ```

- **Sub-tasks & Instructions**:
  - **T3.1.1: Set Up Test Environment**:
    - **Instructions**: Create a `tests/` directory in the project root. Inside, create subdirectories for `unit/` and `integration/`.
    - **Instructions**: Configure `pytest.ini` to define test paths and environment variables. Set up a separate test database configuration that uses an in-memory SQLite database or a temporary file.
    - **Success Criteria**: Running `pytest` from the root directory successfully discovers and runs tests from the `tests/` directory.
  - **T3.1.2: Write Unit Tests for Services & Utilities**:
    - **Instructions**: For each service (`JobService`, `CompanyService`, `AnalyticsService`), create a corresponding test file (e.g., `tests/unit/test_job_service.py`).
    - **Instructions**: Use `pytest-mock`'s `mocker` fixture to patch database sessions and other external dependencies. Write tests for each public method to validate its business logic in isolation. For example, test that `AnalyticsService.get_job_trends` correctly processes a mocked DataFrame.
    - **Success Criteria**: Unit tests for all service methods are implemented and pass, validating the core logic of the application.
  - **T3.1.3: Write Integration Tests for Core Workflows**:
    - **Instructions**: In the `tests/integration/` directory, write tests that use a real (but temporary) database instance.
    - **Instructions**: Test the full `SmartSyncEngine` workflow: create a company, run the sync engine with a list of mock jobs, and assert that the database state is correct (inserts, updates, and archives).
    - **Instructions**: Test the interaction between services, e.g., add a company with `CompanyService`, add jobs for it with `SmartSyncEngine`, and then filter them with `JobService`.
    - **Success Criteria**: Integration tests validate that the key application workflows function correctly from end-to-end.
  - **T3.1.4: Measure and Achieve Test Coverage**:
    - **Instructions**: Install `pytest-cov`. Run tests with the command `pytest --cov=src`.
    - **Instructions**: Analyze the coverage report. Identify any critical, untested code paths in your services and add tests to cover them until the overall coverage exceeds 80%.
    - **Success Criteria**: The test coverage report shows >80% coverage for the `src/services` and `src/utils` directories.

### **T3.2: Implement UI Polish & Advanced UX**

- **Release**: V2.0
- **Priority**: **High**
- **Status**: **PENDING**
- **Prerequisites**: A functional V1.1 application.
- **Related Requirements**: `UI-UX-01`, `UI-UX-02`, `UI-UX-03`
- **Libraries**: `streamlit==1.47.1`
- **Description**: This task focuses on the final "10%" of the user experience that makes an application feel truly professional and delightful to use.

- **Sub-tasks & Instructions**:
  - **T3.2.1: Implement Skeleton Loading States**:
    - **Instructions**: Identify data-heavy pages, primarily the "Jobs" page. Before the data-fetching call to `JobService` is complete, render a placeholder layout.
    - **Instructions**: Use `st.empty()` to create a container. Inside, use a loop with `st.columns` to draw a grid of grey, non-interactive boxes that mimic the shape and layout of the job cards. Once the data is loaded, replace the content of the `st.empty()` container with the actual job grid.
    - **Success Criteria**: When the "Jobs" page is first loaded, a skeleton layout appears instantly, which is then replaced by the job cards once the database query finishes.
  - **T3.2.2: Refine Micro-interactions with CSS**:
    - **Instructions**: Open `src/ui/styles/__init__.py`. Add CSS rules for enhanced hover effects and transitions.
    - **Instructions**: For all buttons and interactive cards, add a `transition: all 0.2s ease-in-out;` property.
    - **Instructions**: Add a `:hover` pseudo-class to these elements to slightly change their `transform` (e.g., `transform: translateY(-2px);`) or `box-shadow` to provide clear visual feedback on mouseover.
    - **Success Criteria**: Hovering over any button or job card results in a smooth, subtle animation.
  - **T3.2.3: Implement a Sliding Filter Panel (Optional Upgrade)**:
    - **Instructions**: This is an ambitious but high-impact polish. If time permits, replace the `st.sidebar` filter with a custom component using `streamlit-elements` that slides in from the side when a "Filters" button is clicked. This keeps the main view less cluttered.
    - **Success Criteria**: A filter panel slides in over the main content, providing a more modern feel than the default sidebar.

### **T3.3: Implement Advanced Data Auditing**

- **Release**: V2.0
- **Priority**: **Medium**
- **Status**: **PENDING**
- **Prerequisites**: `T1.3` (Smart Sync Engine)
- **Related Requirements**: `DB-AUDIT-01`, `DB-AUDIT-02`
- **Libraries**: `sqlmodel==0.0.19`
- **Description**: Enhance the database with detailed auditing tables. This is a "power feature" for long-term maintenance and debugging, providing full visibility into how data changes over time.

- **Sub-tasks & Instructions**:
  - **T3.3.1: Create Auditing Models**:
    - **Instructions**: In `src/models.py`, define the `SyncLogSQL` and `JobChangeSQL` models as specified in `03-database-optimization.md`. `SyncLogSQL` will log entire sync operations, while `JobChangeSQL` will log specific field changes (e.g., `field_name`, `old_value`, `new_value`).
    - **Success Criteria**: The new models are defined and can be created in the database.
  - **T3.3.2: Integrate Logging into SmartSyncEngine**:
    - **Instructions**: Open `src/services/database_sync.py`.
    - **Instructions**: In the `_execute_sync_operations` method, after each database action (insert, update, delete), create and add a corresponding `SyncLogSQL` record to the session before committing.
    - **Instructions**: For update operations, before applying the changes, compare the old and new job data field by field. For each detected change, create and add a `JobChangeSQL` record.
    - **Success Criteria**: After a scraping run, the `synclogsql` and `jobchangesql` tables are populated with accurate records detailing all the changes that occurred during the sync.

### **T3.4: Create Final Documentation**

- **Release**: V2.0
- **Priority**: **Medium**
- **Status**: **PENDING**
- **Prerequisites**: A complete and stable V2.0 application.
- **Related Requirements**: `NFR-DOCS-01`, `NFR-DOCS-02`
- **Libraries**: N/A
- **Description**: Create the final documentation for both end-users and future developers. This is essential for the project's long-term usability and maintainability.

- **Sub-tasks & Instructions**:
  - **T3.4.1: Write the User Guide**:
    - **Instructions**: Create a new file in the project root named `USER_GUIDE.md`.
    - **Instructions**: Write clear, step-by-step instructions covering:
            1. **Installation**: How to set up the Python environment and install dependencies with `uv`.
            2. **Configuration**: How to find and enter API keys on the Settings page.
            3. **First Use**: A walkthrough of adding a company, running a scrape, and browsing the results.
            4. **Core Features**: A brief explanation of the job browser, filtering, and application tracking.
    - **Success Criteria**: A person with basic computer skills can follow the guide to get the application running and use its main features.
  - **T3.4.2: Review and Finalize Developer Documentation**:
    - **Instructions**: Read through the entire `src/` directory. Ensure that all public classes and complex functions have clear, Google-style docstrings explaining their purpose, arguments, and return values.
    - **Instructions**: Add inline comments (`#`) to explain any particularly complex or non-obvious lines of code.
    - **Instructions**: Create or update the main `README.md` to include a "Developer Setup" section that explains how to set up the development environment and run the test suite.
    - **Success Criteria**: A new developer can clone the repository, read the `README.md`, and understand the project's architecture and how to contribute without needing to ask for help.
