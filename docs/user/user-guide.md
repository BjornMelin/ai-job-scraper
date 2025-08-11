# 📖 User Guide: AI Job Scraper

This guide provides a complete walkthrough of all the features available in the AI Job Scraper dashboard.

## 🏠 The Main Dashboard (`Jobs` Page)

This is your central hub for viewing and managing job postings.

### The Job Browser

Jobs are displayed in a responsive grid of cards. Each card gives you a quick overview of the position:

* Job Title, Company, and Location

* A short preview of the description

* Application Status and Favorite indicator

### Filtering and Searching

On the left sidebar, you have powerful tools to narrow down your search:

* **Filter by Company:** Select one or more companies to focus on.

* **Search Keywords:** Type any term (like "Python", "Remote", or "Senior") to search across job titles and descriptions.

* **Salary Range:** Use the dual-handle slider to filter jobs by salary range. The slider ranges from $0 to $750k in $25k increments. When you set the maximum to $750k, it automatically includes all jobs above that amount (shown as "$750k+"), capturing executive and high-paying technical positions that may exceed $1M/year.

* **Date Range:** Use the date pickers to find jobs posted within a specific timeframe.

### Application Tracking

You can manage your application process directly from the job cards:

* **Status:** Use the dropdown menu to change a job's status from "New" to "Interested," "Applied," or "Rejected."

* **Favorites:** Click the heart icon (🤍/❤️) to add or remove a job from your favorites list.

* **View Details:** Clicking this button will expand the card to show the full job description and a text area for your personal notes.

## 🚀 The Scraping Dashboard (`Scraping` Page)

This page is your control center for all scraping operations.

### Controls

* **Start Scraping:** Begins the process of fetching jobs from all companies you've marked as "active."

* **Stop Scraping:** Halts the current scraping process.

* **Reset Progress:** Clears the dashboard's progress indicators after a run is complete.

### Real-Time Progress

When a scrape is active, this dashboard comes alive:

* **Overall Progress:** A main progress bar shows the total completion percentage and current stage (e.g., "Scraping OpenAI...").

* **Overall Metrics:** Key stats like Total Jobs Found, ETA, and the number of active companies are displayed.

* **Company Progress Grid:** Each company being scraped gets its own progress card, showing its individual status, jobs found, and scraping speed.

## 🏢 Company Management (`Companies` Page)

This page allows you to customize which companies the application scrapes.

### Adding a New Company

1. Click the **"➕ Add New Company"** expander.
2. Enter the company's name and the full URL to their careers or jobs page.
3. Click **"Add Company."**

### Managing Existing Companies

Each company is listed with its key details and a toggle switch.

* **Active Toggle:** Use the toggle to include or exclude a company from the next scraping run. This is useful for focusing your search or temporarily disabling a source that isn't working.

* **Scraping Stats:** You can see when a company was last scraped and its historical success rate.

## ⚙️ Settings (`Settings` Page)

This is where you configure the application's core settings.

* **API Configuration:**
  * Select your preferred LLM Provider (OpenAI for quality, Groq for speed).
  * Enter and test your API keys. The application will give you instant feedback on whether the connection is successful.

* **Scraping Configuration:**
  * Use the slider to set the maximum number of jobs to fetch from any single company. This is a safety feature to prevent excessively long scraping times.

* **Save Settings:** Click the save button to apply your changes.
