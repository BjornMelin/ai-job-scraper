# 🕵️‍♂️ AI Job Scraper: Your Modern, Privacy-First Job Hunting Co-Pilot

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)![LangGraph](https://img.shields.io/badge/LangGraph-2C2C2C?style=for-the-badge)![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GitHub](https://img.shields.io/badge/GitHub-BjornMelin-181717?logo=github)](https://github.com/BjornMelin)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-BjornMelin-0077B5?logo=linkedin)](https://www.linkedin.com/in/bjorn-melin/)

AI Job Scraper is a modern, open-source Python application designed to automate and streamline your job search for roles in the AI and Machine Learning industry. It automatically scrapes job postings from top AI companies, filters for relevant roles, and provides a powerful, interactive Streamlit dashboard to track and manage your applications—all while ensuring your data remains private and stored locally.

## ✨ Key Features

* **🤖 Agentic & Hybrid Scraping:** Utilizes `ScrapeGraphAI` and `LangGraph` for intelligent, prompt-based scraping of company career pages and `JobSpy` for high-speed scraping of major job boards.

* **⚡ High-Performance Backend:** Employs a library-first approach with `SQLModel` for the database, a `SmartSyncEngine` to prevent data loss, and optimized background tasks for a non-blocking UI.

* **🎨 Modern, Interactive UI:** A fully-featured Streamlit dashboard with real-time progress updates, advanced filtering, application status tracking, and a responsive, card-based job browser.

* **🏢 Dynamic Company Management:** Easily add, edit, and toggle companies for scraping directly from the UI.

* **🛡️ Robust & Resilient:** Built-in proxy rotation, user-agent randomization, and automatic retries to handle bot detection and network errors gracefully.

* **🐳 Docker Ready:** Comes with a multi-stage `Dockerfile` and `docker-compose.yml` for easy, secure, and repeatable deployments.

* **🔒 Privacy-First:** All your data, notes, and application statuses are stored in a local SQLite database. No personal data ever leaves your machine.

## 🏗️ Architecture

The application is built on a modern, component-based architecture that separates concerns for maintainability and scalability.

```mermaid
graph TD
    subgraph "UI Layer (Streamlit)"
        UI_MAIN[main.py]
        UI_PAGES[Multi-Page System]
        UI_COMP[Component Library]
        UI_STATE[Session State]
    end
    
    subgraph "Business Logic (Services)"
        BL_SCRAPER[Scraper Service]
        BL_SYNC[Smart Sync Engine]
        BL_JOB[Job Service]
        BL_COMPANY[Company Service]
    end
    
    subgraph "Data Layer"
        DB_SQL[SQLite Database]
        DB_MODELS[SQLModel Entities]
    end
    
    subgraph "External Services"
        EXT_LLM[LLM Providers: OpenAI/Groq]
        EXT_PROXIES[Proxy Pool]
        EXT_SITES[Job Boards & Company Pages]
    end
    
    UI_MAIN --> UI_PAGES
    UI_PAGES --> UI_COMP
    UI_COMP --> UI_STATE
    UI_STATE --> BL_JOB
    UI_STATE --> BL_COMPANY
    UI_PAGES -- Triggers --> BL_SCRAPER
    
    BL_SCRAPER --> BL_SYNC
    BL_SYNC --> DB_MODELS
    DB_MODELS --> DB_SQL
    BL_JOB --> DB_MODELS
    BL_COMPANY --> DB_MODELS
    
    BL_SCRAPER -- Uses --> EXT_LLM
    BL_SCRAPER -- Uses --> EXT_PROXIES
    BL_SCRAPER -- Accesses --> EXT_SITES
```

## 🚀 Getting Started

### Prerequisites

* Python 3.12+

* `uv` (or `pip`) Python package manager

* (Optional) Docker for containerized deployment

* (Optional) OpenAI or Groq API key for LLM-powered scraping

### Installation & Running

1. **Clone the repository:**

    ```bash
    git clone https://github.com/BjornMelin/ai-job-scraper.git
    cd ai-job-scraper
    ```

2. **Install dependencies with `uv`:**

    ```bash
    uv sync
    ```

3. **Set up your environment:**
    Copy the `.env.example` file to `.env` and add your API keys.

    ```bash
    cp .env.example .env
    # nano .env
    ```

4. **Initialize and seed the database:**
    This creates the `jobs.db` file and populates it with a curated list of top AI companies.

    ```bash
    uv run python -m src.seed seed
    ```

5. **Run the Streamlit application:**

    ```bash
    streamlit run src/main.py
    ```

6. **Open your browser** and navigate to `http://localhost:8501`.

For more detailed instructions, including Docker deployment, see the full **[Getting Started Guide](./docs/user/getting-started.md)**.

## 📚 Documentation Hub

* **[User Guide](./docs/user/user-guide.md):** Learn how to use all the features of the application.

* **[Developer Guide](./docs/developers/developer-guide.md):** Understand the architecture and how to contribute.

* **[Deployment Guide](./docs/developers/deployment.md):** Instructions for deploying the app to production.

## 🙌 Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and open a pull request. See the [Developer Guide](./docs/developers/developer-guide.md) for more details on setting up your environment and our coding standards.

## 📃 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
