# 🚀 Getting Started with AI Job Scraper

Welcome! This guide will walk you through setting up and running the AI Job Scraper on your local machine in just a few minutes.

## 📋 Prerequisites

Before you begin, make sure you have the following:

* **Python 3.12 or newer.**

* **`uv`**, a fast Python package installer. If you don't have it, you can install it with `pip install uv`.

* **Git** for cloning the project repository.

* **(Optional but Recommended)** An **OpenAI** or **Groq** API key for the highest quality job scraping.

## ⚡ Quick Start (5-Minute Setup)

### 1. Clone the Repository

Open your terminal or command prompt and run the following command to download the project:

```bash
git clone https://github.com/BjornMelin/ai-job-scraper.git
cd ai-job-scraper
```

### 2. Install Dependencies

We use `uv` for fast and reliable dependency management. Run this command in the project directory:

```bash
uv sync
```

This will create a virtual environment and install all the necessary packages.

### 3. Configure Your API Keys

For the best results, you need to provide an LLM API key.

1. Find the file named `.env.example` in the project folder.
2. Make a copy of it and rename the copy to `.env`.
3. Open the new `.env` file and paste your API key from OpenAI or Groq.

```env

# .env file
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here

# Set USE_GROQ to true if you want to use the Groq API
USE_GROQ=false
```

### 4. Initialize the Database

Before the first run, you need to create the database and add a curated list of top AI companies.

```bash
uv run python -m src.seed
```

You should see a message confirming that companies have been seeded.

### 5. Run the Application

You're all set! Start the Streamlit web application with this command:

```bash
uv run streamlit run src/main.py
```

Your web browser should automatically open to `http://localhost:8501`.

**🎉 Congratulations! You are now running the AI Job Scraper!**

## ▶️ Your First Scrape

1. Navigate to the **"Scraping"** page using the left-hand menu.
2. Click the **"🚀 Start Scraping"** button.
3. Watch the real-time progress dashboard as the application finds jobs from the pre-configured companies.
4. Once complete, navigate to the **"Jobs"** page to see your results!

## 🔧 Troubleshooting

* **`uv: command not found`**: Make sure you have installed `uv` correctly. Try `pip install uv`.

* **Port 8501 already in use**: Another application is using the default port. Run `streamlit run src/main.py --server.port 8502` to use a different port.

* **API Key Errors**: Double-check that you have copied your API key correctly into the `.env` file and saved it.

For more detailed solutions, please see the full **[Troubleshooting Guide](./troubleshooting.md)**.
