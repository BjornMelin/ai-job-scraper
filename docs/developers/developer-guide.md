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
| `src/ui/pages/`         | Contains the Streamlit code for each distinct page of the application (Jobs, Companies, etc.).      |
| `src/ui/components/`    | Reusable UI components (e.g., job cards, sidebar) used by the pages.                                |
| `src/ui/state/`         | Manages the application's UI state using Streamlit's native `session_state`.                        |
| `src/services/`         | Encapsulates all business logic (database interactions, data synchronization).                      |
| `src/scraper.py`        | The main orchestrator for all scraping tasks.                                                       |
| `src/scraper_*.py`      | Specialized modules for scraping job boards (`JobSpy`) and company pages (`ScrapeGraphAI`).         |
| `src/database.py`       | Handles database engine creation, session management, and performance listeners.                    |
| `src/models.py`         | Defines the application's data structures using `SQLModel`.                                         |
| `src/config.py`         | Manages application settings and secrets using `pydantic-settings`.                                 |

## 📚 Technical Stack

* **UI Framework:** Streamlit 1.47+

* **Database ORM:** SQLModel

* **Scraping Libraries:** ScrapeGraphAI (for company pages), JobSpy (for job boards)

* **Agentic Workflows:** LangGraph

* **LLM Providers:** OpenAI, Groq (configurable)

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

* **`SmartSyncEngine`:** A critical component that handles the complex logic of updating the database with scraped data without destroying user edits. See `ADR-008` for a detailed breakdown.

### Background Tasks (`src/ui/utils/background_tasks.py`)

Scraping is a long-running task and is handled in a background thread to keep the UI responsive.

* **Simplicity:** The implementation uses Python's standard `threading.Thread`.

* **UI Updates:** The background thread communicates progress back to the main thread by updating `st.session_state`. The UI on the `scraping.py` page periodically reruns to read this state and display the latest progress.

* **Safety:** The system is designed to run one scraping task at a time. Database sessions for background threads are managed carefully to prevent conflicts.

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
