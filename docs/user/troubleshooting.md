# 🔧 Troubleshooting Guide: AI Job Scraper

This guide helps you diagnose and resolve common issues.

## 🚨 Quick Diagnostics

If you're having trouble, first check the **"Database Health"** expander in the bottom-left of the sidebar. A green "Healthy" status means your database connection is good. If it shows a warning or error, this is likely the source of the problem.

## ❌ Installation & Startup Issues

### "Module not found" or `ImportError`

**Symptom:** The application fails to start with an error about a missing module (e.g., `streamlit`, `sqlmodel`).

**Solution:** Your dependencies are likely not installed correctly. Run `uv sync` again.

```bash
uv sync
```

### `uv: command not found`

**Symptom:** Your terminal doesn't recognize the `uv` command.

**Solution:** You need to install `uv`. The easiest way is with `pip`:

```bash
pip install uv
```

### "Port 8501 is already in use"

**Symptom:** The app fails to start because another process is using the default port.

**Solution:** Run the app on a different port.

```bash
streamlit run src/main.py --server.port 8502
```

## 🌐 Scraping & API Issues

### No Jobs Found After Scraping

**Symptom:** You run a scrape, it completes successfully, but the "Jobs" page is empty.

**Solutions:**

1. **Check Active Companies:** Go to the "Companies" page and ensure the companies you want to scrape have their "Active" toggle turned on.
2. **Check Your Filters:** Go to the "Jobs" page and click the "Clear All Filters" button in the sidebar to make sure you're not filtering out all results.
3. **Website Changes:** The company may have changed its website structure. The scraper will try to adapt, but sometimes it may fail. This usually resolves itself as the underlying scraping libraries are updated.

### API Key Errors

**Symptom:** You see an error message about "Authentication" or "Invalid API Key" on the "Settings" page.

**Solutions:**

1. **Verify Your Key:** Go to the "Settings" page and carefully re-paste your API key.
2. **Check `.env` File:** Ensure you have copied the key correctly into your `.env` file and that the file is in the root directory of the project.
3. **Check Account Status:** Log in to your OpenAI or Groq account to ensure your account is active and has available credits or usage limits.

## 💾 Database Issues

### "Database is locked"

**Symptom:** You see a red error message in the app about a `database is locked` error.

**Solution:** This can happen if multiple processes are trying to access the database file at once. The best solution is to simply stop and restart the application.

1. Press `Ctrl+C` in your terminal to stop the Streamlit server.
2. Run `streamlit run src/main.py` to start it again.

### Data Seems Out of Date or Incorrect

**Symptom:** The jobs list doesn't seem to update after a scrape, or data looks wrong.

**Solution:** This could be a caching issue or a failed scrape.

1. Perform a hard refresh in your browser (`Ctrl+Shift+R` or `Cmd+Shift+R`).
2. Run a new scrape from the "Scraping" page and watch the progress dashboard for any errors.
3. If problems persist, you can perform a hard reset by deleting the `jobs.db` file and re-initializing the database with `uv run python -m src.seed seed`. **Warning: This will delete all your existing job data and user notes.**

## 🆘 Getting Further Help

If your issue isn't listed here, please open an issue on our [GitHub repository](https://github.com/BjornMelin/ai-job-scraper/issues). Please include:

* Your operating system (e.g., Windows 11, macOS Sonoma).

* The exact error message you are seeing.

* The steps you took that led to the error.
