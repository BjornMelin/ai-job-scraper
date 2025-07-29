# 🚀 Getting Started with AI Job Scraper

Welcome to AI Job Scraper! This guide will walk you through setting up and running the application to start tracking AI job opportunities from top companies.

## 📋 Prerequisites

Before you begin, ensure you have the following installed on your system:

### Required

- **Python 3.12+** - Download from [python.org](https://python.org/downloads/)

- **uv** - Modern Python package manager for faster dependency management

- **Git** - For cloning the repository

### Optional

- **Docker & Docker Compose** - For containerized deployment

- **OpenAI API Key** - For enhanced job extraction (free tier available)

### System Requirements

- **Operating System**: Windows 10+, macOS 10.15+, or Linux

- **RAM**: 2GB minimum, 4GB recommended

- **Disk Space**: 500MB for application + dependencies

- **Network**: Internet connection for scraping and API calls

## ⚡ Quick Start (5 Minutes)

### Option 1: Local Installation (Recommended)

1. **Clone the repository**

   ```bash
   git clone https://github.com/BjornMelin/ai-job-scraper.git
   cd ai-job-scraper
   ```

2. **Install uv package manager** (if not already installed)

   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # On Windows (PowerShell)
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # Alternative: with pip
   pip install uv
   ```

3. **Install dependencies**

   ```bash
   uv sync
   ```

4. **Set up the database**

   ```bash
   uv run python seed.py
   ```

5. **Run the application**

   ```bash
   uv run streamlit run app.py
   ```

6. **Open your browser**
   Navigate to [http://localhost:8501](http://localhost:8501)

**🎉 Congratulations! Your AI Job Scraper is running!**

### Option 2: Docker Installation

1. **Clone and navigate**

   ```bash
   git clone https://github.com/BjornMelin/ai-job-scraper.git
   cd ai-job-scraper
   ```

2. **Build and run with Docker Compose**

   ```bash
   docker-compose up --build
   ```

3. **Access the application**
   Open [http://localhost:8501](http://localhost:8501) in your browser

## 🔑 Environment Configuration

### OpenAI API Key Setup (Optional but Recommended)

For enhanced job extraction accuracy, set up your OpenAI API key:

1. **Get an API key** from [OpenAI Platform](https://platform.openai.com/api-keys)

2. **Create a .env file** in the project root:

   ```bash
   touch .env
   ```

3. **Add your API key** to the .env file:

   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

**Note**: The application works without an API key using CSS-based extraction, but LLM extraction provides better accuracy and handles dynamic content.

### Database Configuration (Advanced)

By default, the application uses SQLite. To use a different database:

```env

# PostgreSQL example
DB_URL=postgresql://username:password@localhost:5432/ai_jobs

# MySQL example  
DB_URL=mysql://username:password@localhost:3306/ai_jobs
```

## 🏢 Initial Setup

### Adding Companies

The application comes pre-configured with major AI companies:

- Anthropic

- OpenAI  

- DeepMind

- xAI

- Meta AI

- Microsoft AI

- NVIDIA

To add more companies:

1. **Via the UI**: Use the sidebar "Manage Companies" section
2. **Via code**: Edit `seed.py` and add to the `SITES` dictionary:

   ```python
   SITES = {
       "your_company": "https://company.com/careers",
       # ... existing companies
   }
   ```

### First Scrape

1. **Click "Rescrape Jobs"** in the main interface
2. **Wait for completion** (typically 30-60 seconds for all companies)
3. **View results** in the "All Jobs" tab

**Expected first run**: 20-100 jobs depending on current openings

## 🎯 Basic Usage

### Dashboard Overview

The application features three main tabs:

- **📋 All Jobs**: View all scraped positions

- **⭐ Favorites**: Jobs you've marked as interesting  

- **✅ Applied**: Track your applications

### Key Features

1. **Global Filters** (Left Sidebar)
   - Filter by company
   - Search by keywords
   - Date range filtering

2. **View Modes**
   - **List View**: Editable table format
   - **Card View**: Visual grid with pagination

3. **Job Management**
   - Mark favorites with ⭐
   - Update application status
   - Add personal notes
   - Export to CSV

### Navigation Tips

- **Search within tabs**: Use the search box in each tab

- **Sort results**: Available in card view (by date, title, company)

- **Bulk editing**: Use list view for efficient editing

- **Quick actions**: Use card view for visual browsing

## 🔧 Verification & Troubleshooting

### Verify Installation

Run these commands to ensure everything is working:

```bash

# Check Python version
python --version  # Should be 3.12+

# Check uv installation
uv --version

# Test the scraper module
uv run python -c "from scraper import main; print('✅ Scraper imported successfully')"

# Test the Streamlit app
uv run python -c "import app; print('✅ App imported successfully')"
```

### Common Issues

#### "Module not found" errors

```bash

# Ensure dependencies are installed
uv sync
```

#### Port 8501 already in use

```bash

# Use a different port
uv run streamlit run app.py --server.port=8502
```

#### OpenAI API errors

- Verify your API key in `.env`

- Check your OpenAI account has credits

- The app will fallback to CSS extraction automatically

#### Empty job results

- Check company URLs are accessible

- Verify internet connection

- Some companies may have temporarily changed their career page structure

### Performance Verification

After your first successful scrape, check the logs for performance metrics:

```text
📊 Session Summary:
  Duration: 45.2s
  Companies: 7
  Jobs found: 67
  Cache hit rate: 85%
  LLM calls: 2
  Errors: 0
```

**Good performance indicators**:

- Cache hit rate > 70% (after first run)

- Duration < 60s for all companies

- Errors: 0

## 📚 Next Steps

Now that you're up and running:

1. **📖 Read the [User Guide](USER_GUIDE.md)** - Learn all features in detail
2. **🎨 Customize your setup** - Add companies, configure filters
3. **📊 Set up a routine** - Scrape daily or weekly for best results
4. **🔧 Explore advanced features** - Check the [Developer Guide](DEVELOPER_GUIDE.md) for customization

## 🆘 Getting Help

If you encounter issues:

1. **Check [Troubleshooting Guide](TROUBLESHOOTING.md)** for common solutions
2. **Review logs** in the terminal for error details  
3. **Open an issue** on [GitHub](https://github.com/BjornMelin/ai-job-scraper/issues)
4. **Check the [FAQ section](TROUBLESHOOTING.md#faq)** for quick answers

## 🌟 Pro Tips

- **Set up monitoring**: Run scrapes regularly to catch new postings

- **Use favorites strategically**: Mark jobs that closely match your criteria  

- **Export regularly**: Download CSV backups of your job data

- **Customize companies**: Focus on companies you're genuinely interested in

- **Leverage filters**: Use global filters to narrow down to your specific requirements

> **Happy job hunting! 🎯**
