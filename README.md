# 🕵️‍♂️ AI Job Scraper: Track Your Dream AI Roles Locally

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Crawl4AI](https://img.shields.io/badge/Crawl4AI-2C2C2C?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![GitHub](https://img.shields.io/badge/GitHub-BjornMelin-181717?logo=github)](https://github.com/BjornMelin)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-BjornMelin-0077B5?logo=linkedin)](https://www.linkedin.com/in/bjorn-melin/)

**AI Job Scraper** is an open-source Python application that automatically scrapes job postings from top AI companies (e.g., Anthropic, OpenAI, NVIDIA), filters for relevant roles (AI Engineer, MLOps, etc.), stores/updates them in a local database, and provides an interactive Streamlit dashboard for tracking/managing jobs. Built with Crawl4AI for efficient scraping, it ensures privacy with local processing and supports Docker for easy setup.

## ✨ Features of AI Job Scraper

- **Automated Scraping:** Fetch jobs from company sites using Crawl4AI (LLM/CSS strategies, async).

- **Relevance Filtering:** Regex for AI/ML roles; Modular for future LLM.

- **Persistent Storage:** SQLite DB with add/update/delete, preserving user edits (favorite/status/notes).

- **Interactive Dashboard:** Tabs (All/Favorites/Applied), views (list/card with sort/paginate/search), edits, CSV export, theme toggle.

- **Company Management:** Add/remove/activate sites via UI.

- **Robustness:** Retries/fallbacks/validation/logging; Docker support.

- **Privacy-Focused:** Local-only (optional OpenAI for extraction).

## 📖 Table of Contents

- [🕵️‍♂️ AI Job Scraper: Track Your Dream AI Roles Locally](#️️-ai-job-scraper-track-your-dream-ai-roles-locally)
  - [✨ Features of AI Job Scraper](#-features-of-ai-job-scraper)
  - [📖 Table of Contents](#-table-of-contents)
  - [🚀 Getting Started](#-getting-started)
    - [📋 Prerequisites](#-prerequisites)
    - [⚙️ Installation](#️-installation)
    - [▶️ Running the App](#️-running-the-app)
  - [💻 Usage](#-usage)
    - [🏢 Managing Companies](#-managing-companies)
    - [🔄 Rescraping Jobs](#-rescraping-jobs)
    - [📊 Viewing and Filtering Jobs](#-viewing-and-filtering-jobs)
    - [✏️ Editing and Tracking](#️-editing-and-tracking)
    - [📥 Exporting Data](#-exporting-data)
  - [🏗️ Architecture](#️-architecture)
  - [🛠️ Implementation Details](#️-implementation-details)
  - [🙌 Contributing](#-contributing)
  - [📃 License](#-license)

## 🚀 Getting Started

### 📋 Prerequisites

- Python 3.12+.

- (Optional) Docker for containerized run.

- (Optional) OpenAI key for enhanced extraction.

### ⚙️ Installation

1. Clone:

   ```bash
   git clone https://github.com/BjornMelin/ai-job-scraper.git
   cd ai-job-scraper
   ```

2. Install deps:

   ```bash
   uv sync
   ```

3. Seed DB (initial companies):

   ```bash
   uv run python seed.py
   ```

### ▶️ Running the App

**Locally:**

```bash
uv run streamlit run app.py
```

**With Docker:**

```bash
docker-compose up --build
```

Access at <http://localhost:8501>.

## 💻 Usage

### 🏢 Managing Companies

In sidebar: Edit active status, add new name/URL. Save updates DB; Scrapes only active.

### 🔄 Rescraping Jobs

Click "Rescrape Jobs": Fetches latest, filters, updates DB (adds new, updates changed, deletes missing, preserves edits).

### 📊 Viewing and Filtering Jobs

Tabs for All/Favorites/Applied. Global filters (company/keyword/date) + per-tab search. Toggle list (editable table) or card (visual grid with sort/paginate).

### ✏️ Editing and Tracking

In list: Edit favorite/status/notes inline, save to DB. In card: Quick toggles/selects/areas.

### 📥 Exporting Data

Download CSV per tab (filtered/edited data).

## 🏗️ Architecture

```mermaid
graph TD
    A[User] -->|Interact| B[Streamlit UI: Tabs/Views/Search/Edit/Export]
    B -->|Rescrape/Manage| C[Scraper: Crawl4AI Async/Retries/Fallbacks/Validate]
    C -->|Filter/Update| D[SQLite DB: Jobs/Companies (SQLAlchemy ORM)]
    B -->|Query/Filter| D
    E[Logging/Errors] -->|All| A & C & B
    F[Docker] -->|Container| B & D
```

## 🛠️ Implementation Details

- **Scraping:** Crawl4AI v0.7.2 with LLM schema for structured jobs, async gather for parallel.

- **Filtering:** Regex for relevance; Pydantic for validation.

- **DB:** SQLAlchemy v2.0.42 with models for jobs/companies; Hash for updates.

- **UI:** Streamlit v1.47.1 with data_editor, custom CSS for cards/theme, session_state for persistence.

- **Robustness:** Tenacity v9.1.2 retries, httpx v0.28.1 validation, logging throughout.

- **Code Quality:** Ruff linted, Google docstrings, tests in tests/.

- **Performance:** Async scraping, Pandas v2.3.1 for data, Docker for env.

## 🙌 Contributing

Fork, branch, PR. Follow KISS/DRY. See ADRs/PRD for guidance.

## 📃 License

MIT License—see [LICENSE](LICENSE).

---

<div align="center">

Built by [Bjorn Melin](https://bjornmelin.io)

</div>
