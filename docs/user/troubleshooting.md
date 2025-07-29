# 🔧 Troubleshooting Guide: AI Job Scraper

This guide helps you diagnose and resolve common issues with the AI Job Scraper application.

## 🚨 Quick Diagnostics

### Health Check Commands

Run these commands to quickly identify issues:

```bash

# Check Python version (must be 3.12+)
python --version

# Verify uv installation
uv --version

# Test module imports
uv run python -c "import scraper, models, app; print('✅ All modules imported successfully')"

# Test database connection
uv run python -c "from models import engine; engine.execute('SELECT 1'); print('✅ Database connection works')"

# Test Streamlit
uv run python -c "import streamlit; print('✅ Streamlit available')"

# Check file permissions
ls -la jobs.db cache/
```

### System Requirements Check

```bash

# Check available memory (should be >2GB)
free -h

# Check disk space (should be >500MB free)
df -h .

# Test internet connectivity
curl -I https://openai.com/careers

# Check if port 8501 is available
netstat -tuln | grep 8501
```

## ❌ Installation Issues

### "Module not found" Errors

**Symptom**: `ImportError: No module named 'crawl4ai'` or similar

**Causes & Solutions**:

1. **Dependencies not installed**

   ```bash
   # Reinstall dependencies
   uv sync
   
   # Force clean install
   rm -rf .venv
   uv sync
   ```

2. **Wrong Python environment**

   ```bash
   # Check which Python uv is using
   uv run python --version
   
   # Use specific Python version
   uv python pin 3.12
   uv sync
   ```

3. **Corrupted virtual environment**

   ```bash
   # Remove and recreate environment
   rm -rf .venv
   uv venv
   uv sync
   ```

### uv Installation Problems

**Symptom**: `command not found: uv`

**Solutions**:

1. **Install uv (Linux/macOS)**

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Add to PATH if needed
   echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Install uv (Windows)**

   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Alternative: Use pip**

   ```bash
   pip install uv
   ```

### Playwright/Browser Issues

**Symptom**: Browser automation fails, Crawl4AI errors

**Solutions**:

1. **Install Playwright browsers**

   ```bash
   uv run python -m playwright install
   
   # Install system dependencies (Linux)
   uv run python -m playwright install-deps
   ```

2. **System package issues (Ubuntu/Debian)**

   ```bash
   sudo apt-get update
   sudo apt-get install -y \
     libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
     libcups2 libdbus-1-3 libdrm2 libxkbcommon0 \
     libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
     libgbm1 libasound2
   ```

3. **Docker environment**

   ```dockerfile
   # Ensure Dockerfile includes browser dependencies
   RUN apt-get update && apt-get install -y \
       libnss3 libnspr4 libatk1.0-0 \
       && rm -rf /var/lib/apt/lists/*
   ```

## 🌐 Network & API Issues

### OpenAI API Problems

**Symptom**: LLM extraction fails, API key errors

**Diagnostic Commands**:

```bash

# Check API key is set
echo $OPENAI_API_KEY

# Test API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

**Solutions**:

1. **Missing or invalid API key**

   ```bash
   # Create .env file with valid key
   echo "OPENAI_API_KEY=sk-proj-your-key-here" > .env
   
   # Verify key format (should start with sk-proj- or sk-)
   cat .env
   ```

2. **API quota exceeded**
   - Check your OpenAI account billing and usage
   - The app will fallback to CSS extraction automatically
   - Add credits to your OpenAI account

3. **Network connectivity issues**

   ```bash
   # Test OpenAI API connectivity
   curl -I https://api.openai.com/v1/models
   
   # Check for proxy/firewall issues
   curl --proxy http://your-proxy:port https://api.openai.com
   ```

### Website Access Problems

**Symptom**: Scraping fails for specific companies, timeout errors

**Diagnostic Steps**:

```bash

# Test company website access
curl -I https://careers.anthropic.com/jobs

# Check if site blocks automated requests
curl -H "User-Agent: Mozilla/5.0" https://openai.com/careers

# Test with different delay

# (Handled automatically by the application)
```

**Solutions**:

1. **Rate limiting (most common)**
   - The app includes company-specific delays
   - Wait and try again later
   - Some companies may have temporarily increased restrictions

2. **Website structure changes**

   ```bash
   # Clear cache to force re-extraction
   rm -rf cache/
   
   # The app will regenerate extraction schemas
   ```

3. **Temporary site outages**
   - Check if the company's careers page is accessible in browser
   - Wait and retry later
   - Individual company failures don't stop other companies

## 💾 Database Issues

### SQLite Database Problems

**Symptom**: Database locked, corruption errors, empty results

**Diagnostic Commands**:

```bash

# Check database exists and has correct permissions
ls -la jobs.db

# Check database integrity
sqlite3 jobs.db "PRAGMA integrity_check;"

# Check table contents
sqlite3 jobs.db "SELECT COUNT(*) FROM jobs;"
sqlite3 jobs.db "SELECT COUNT(*) FROM companies;"
```

**Solutions**:

1. **Database locked errors**

   ```bash
   # Close all running instances of the app
   pkill -f streamlit
   pkill -f "python.*app.py"
   
   # Restart the application
   uv run streamlit run app.py
   ```

2. **Corrupted database**

   ```bash
   # Backup existing data
   cp jobs.db jobs.db.backup
   
   # Re-initialize database
   rm jobs.db
   uv run python seed.py
   
   # Re-run scraping
   uv run python scraper.py
   ```

3. **Empty database after seeding**

   ```bash
   # Verify companies were added
   sqlite3 jobs.db "SELECT * FROM companies;"
   
   # Re-run seeding
   uv run python seed.py
   
   # Check for errors in logs
   ```

4. **Permission issues**

   ```bash
   # Fix database permissions
   chmod 644 jobs.db
   
   # Ensure directory is writable
   chmod 755 .
   ```

### PostgreSQL Connection Issues (Advanced Users)

**Symptom**: Connection refused, authentication failed

**Solutions**:

```bash

# Test PostgreSQL connection
psql -h localhost -U username -d ai_jobs

# Check connection string format

# DB_URL=postgresql://username:password@host:port/database

# Verify PostgreSQL service is running
sudo systemctl status postgresql
```

## 🖥️ UI and Display Issues

### Streamlit Problems

**Symptom**: App won't start, white screen, layout issues

**Solutions**:

1. **Port already in use**

   ```bash
   # Find process using port 8501
   lsof -i :8501
   
   # Kill the process
   kill -9 <PID>
   
   # Or use different port
   uv run streamlit run app.py --server.port=8502
   ```

2. **Browser caching issues**

   ```bash
   # Clear browser cache and reload
   # Or use incognito/private mode
   
   # Hard refresh: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
   ```

3. **CSS/styling problems**

   ```bash
   # Restart app to reload CSS
   # CSS is embedded in app.py, so restart fixes most issues
   ```

### Display and Formatting Issues

**Symptom**: Jobs not showing, layout broken on mobile

**Solutions**:

1. **No jobs displayed**

   ```python
   # Check if jobs exist in database
   # Run in Python console:
   from models import Session, JobSQL
   session = Session()
   count = session.query(JobSQL).count()
   print(f"Jobs in database: {count}")
   ```

2. **Mobile layout issues**
   - Use card view instead of list view on mobile
   - The CSS includes responsive breakpoints
   - Try refreshing the page

3. **Filtering not working**
   - Clear all filters and try again
   - Check that global filters aren't too restrictive
   - Verify companies are marked as active

## ⚡ Performance Issues

### Slow Scraping

**Symptom**: Scraping takes >2 minutes, timeouts

**Diagnostic Steps**:

```bash

# Check cache hit rate in logs

# Look for: "Cache hit rate: X%"

# Check network speed
speedtest-cli

# Monitor resource usage during scraping
top -p $(pgrep -f python)
```

**Solutions**:

1. **Low cache hit rate (<50%)**

   ```bash
   # Check cache directory exists
   ls -la cache/
   
   # Ensure cache files are being created
   # Should see .json files after first run
   ```

2. **Network latency**
   - Run scraping during off-peak hours
   - Consider using faster internet connection
   - Some company sites are naturally slower

3. **Resource constraints**

   ```bash
   # Check available memory
   free -h
   
   # Reduce concurrent operations if memory-limited
   # The app processes companies sequentially to manage resources
   ```

### High Memory Usage

**Symptom**: App uses >1GB RAM, system becomes slow

**Solutions**:

1. **Too many jobs in database**

   ```sql
   -- Clean old jobs (run in sqlite3 jobs.db)
   DELETE FROM jobs WHERE last_seen < date('now', '-30 days');
   VACUUM;
   ```

2. **Large job descriptions**
   - The app truncates descriptions to 1000 characters
   - If you modified this limit, consider reducing it

3. **Memory leaks (rare)**

   ```bash
   # Restart the application
   pkill -f streamlit
   uv run streamlit run app.py
   ```

## 🐛 Common Error Messages

### "No module named 'crawl4ai'"

**Fix**:

```bash
uv sync

# If that fails:
uv pip install crawl4ai==0.7.2
```

### "AttributeError: 'NoneType' object has no attribute..."

**Cause**: Usually database connection or missing data

**Fix**:

```bash

# Re-initialize database
uv run python seed.py

# Check database has data
sqlite3 jobs.db "SELECT COUNT(*) FROM companies;"
```

### "ValidationError: 1 validation error for JobPydantic"

**Cause**: Invalid job data (usually malformed URLs)

**Fix**:

- This is expected and handled automatically

- Invalid jobs are skipped with a warning

- Check logs for specific validation errors

### "OpenAI API Error: Rate limit exceeded"

**Fix**:

```bash

# Wait and retry (usually 1 minute)

# Or add billing to your OpenAI account

# The app will fallback to CSS extraction
```

### "Database is locked"

**Fix**:

```bash

# Close all instances
pkill -f streamlit
pkill -f "python.*scraper.py"

# Wait 5 seconds, then restart
uv run streamlit run app.py
```

### "Port 8501 is already in use"

**Fix**:

```bash

# Kill existing process
lsof -i :8501
kill -9 <PID>

# Or use different port
uv run streamlit run app.py --server.port=8502
```

## 🐳 Docker Issues

### Container Won't Start

**Symptoms**: Docker build fails, container exits immediately

**Solutions**:

1. **Build issues**

   ```bash
   # Clean build
   docker-compose down
   docker system prune -f
   docker-compose up --build
   ```

2. **Permission issues**

   ```bash
   # Fix volume permissions
   sudo chown -R $(id -u):$(id -g) .
   
   # Or use named volumes in docker-compose.yml
   ```

3. **Port conflicts**

   ```bash
   # Change port in docker-compose.yml
   ports:
     - "8502:8501"  # Use port 8502 instead
   ```

### Container Performance Issues

**Solutions**:

```bash

# Allocate more resources to Docker

# Docker Desktop: Settings > Resources

# Check container resource usage
docker stats

# Use multi-stage build to reduce image size

# (Already implemented in current Dockerfile)
```

## 📊 Data Issues

### No Jobs Found After Scraping

**Diagnostic Steps**:

```bash

# Check if scraping completed

# Look for "Session Summary" in logs

# Verify companies are active
sqlite3 jobs.db "SELECT name, active FROM companies;"

# Check for extraction errors in logs
```

**Solutions**:

1. **All companies inactive**

   ```sql
   -- Activate companies (run in sqlite3 jobs.db)
   UPDATE companies SET active = 1;
   ```

2. **Relevance filtering too strict**

   ```python
   # Temporarily modify regex in scraper.py for testing
   # RELEVANT_KEYWORDS = re.compile(r"Engineer", re.I)  # Broader match
   ```

3. **Website structure changes**

   ```bash
   # Clear cache to force re-extraction
   rm -rf cache/
   ```

### Duplicate Jobs Appearing

**Cause**: Link-based deduplication failing

**Fix**:

```sql

-- Remove duplicates manually (run in sqlite3 jobs.db)
DELETE FROM jobs WHERE id NOT IN (
    SELECT MIN(id) FROM jobs GROUP BY link
);
```

### Jobs Not Updating

**Cause**: Hash-based change detection not working

**Solutions**:

```bash

# Force update by clearing hashes
sqlite3 jobs.db "UPDATE jobs SET hash = NULL;"
```

## 📝 Logging and Debugging

### Enable Debug Logging

```python

# Temporarily modify scraper.py
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

### Common Log Messages

**"Using cached schema for {company}"** ✅ Good

- Indicates caching is working

**"LLM failed for {company}: ... CSS fallback"** ⚠️ Warning  

- LLM extraction failed, using basic CSS

- Usually still works but may miss some jobs

**"All attempts failed for {company}"** ❌ Error

- Company scraping completely failed

- Check network connectivity and website accessibility

**"Job validation failed"** ⚠️ Warning

- Invalid job data found and skipped

- Normal operation, some sites have malformed data

## 🆘 Getting Further Help

### Before Asking for Help

1. **Check logs** for specific error messages
2. **Try the diagnostic commands** above
3. **Search existing issues** on GitHub
4. **Try a clean install** in a new directory

### Information to Include

When reporting issues, include:

```bash

# System information
uv --version
python --version
cat /etc/os-release  # Linux
sw_vers  # macOS

# Application state
ls -la jobs.db cache/
sqlite3 jobs.db "SELECT COUNT(*) FROM jobs, companies;"

# Error logs (last 20 lines)
uv run streamlit run app.py 2>&1 | tail -20
```

### Support Channels

1. **GitHub Issues**: [Create an issue](https://github.com/BjornMelin/ai-job-scraper/issues)
2. **Check existing documentation**:
   - [Getting Started](GETTING_STARTED.md)
   - [User Guide](USER_GUIDE.md)
   - [Developer Guide](DEVELOPER_GUIDE.md)

## 📋 FAQ

### Q: Why is scraping slow on first run?

**A**: The first run has no cache, so it uses LLM extraction (~45-90s). Subsequent runs are much faster (15-45s) due to caching.

### Q: Can I run this without an OpenAI API key?

**A**: Yes! The app will use CSS-based extraction as a fallback. You'll get fewer jobs but it still works.

### Q: How often should I scrape?

**A**: Daily or weekly. Companies don't post new jobs constantly, so hourly scraping isn't necessary.

### Q: Why aren't remote jobs showing up?

**A**: Make sure you're not filtering by location in the global filters. Search for "remote" in the keyword field.

### Q: Can I add my own companies?

**A**: Yes! Use the "Manage Companies" section in the sidebar, or edit `seed.py` and re-run it.

### Q: The UI looks broken on mobile

**A**: Use Card view instead of List view on mobile devices. The responsive CSS works better with cards.

### Q: How do I backup my data?

**A**: Copy the `jobs.db` file and export CSV from each tab. The database contains all your data.

### Q: Can I deploy this to a server?

**A**: Yes! See the [Deployment Guide](DEPLOYMENT.md) for production setup instructions.

### Q: Why do some companies show no jobs?

**A**: Companies may not have AI/ML positions currently open, or their website structure may have changed. Check the logs for specific errors.

This troubleshooting guide covers the most common issues. For additional help, consult the other documentation files or open a GitHub issue with detailed information about your problem.
