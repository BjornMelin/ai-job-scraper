# 🚀 Deployment Guide: AI Job Scraper

This guide provides comprehensive strategies for deploying the AI Job Scraper in a production environment.

## 🐳 Docker Deployment (Recommended)

Using Docker is the recommended method for a clean, repeatable, and secure deployment. The project includes a multi-stage `Dockerfile` and a `docker-compose.yml` for this purpose.

### System Requirements - Docker Deployment

* A host machine with Docker and Docker Compose installed.

* **Memory**: 2GB RAM minimum, 4GB recommended for analytics workloads
* **CPU**: 1 vCPU minimum, 2+ vCPUs recommended for DuckDB analytics
* **Storage**: 10GB minimum (SQLite database + DuckDB temporary files)
* **Dependencies**: DuckDB is automatically installed via pyproject.toml

### Deployment Steps

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/BjornMelin/ai-job-scraper.git
    cd ai-job-scraper
    ```

2. **Configure Environment Variables:**
    Copy the `.env.example` file to `.env` and populate it with your production secrets, such as your `OPENAI_API_KEY` and `GROQ_API_KEY`.

    ```bash
    cp .env.example .env
    # nano .env
    ```

    The `docker-compose.yml` is configured to automatically load this `.env` file.

3. **Build and Run the Container:**
    Use Docker Compose to build the image and start the service. The `-d` flag runs it in detached mode.

    ```bash
    docker-compose up --build -d
    ```

4. **Initialize the Database:**
    The first time you deploy, you need to run the database seeder inside the running container. This creates the SQLite database that DuckDB will scan directly.

    ```bash
    docker-compose exec app uv run python -m src.seed seed
    ```

    **DuckDB Analytics Setup**: No additional setup required - DuckDB's sqlite_scanner extension is automatically loaded and configured to scan the SQLite database directly.

5. **Verify the Application:**
    The application should now be accessible at `http://localhost:8501`. You can check the status and logs of the running container:

    ```bash
    docker-compose ps
    docker-compose logs -f app
    ```

### Persistent Data

The `docker-compose.yml` file is configured to use a Docker volume (`dbdata`) to persist:

* **Primary Database**: SQLite (`jobs.db`) for all transactional operations
* **Cost Tracking**: SQLite (`costs.db`) for budget monitoring
* **Analytics Cache**: DuckDB temporary files (automatically managed)

**Architecture**: DuckDB uses sqlite_scanner to read SQLite files directly - no separate DuckDB database file is created or maintained. This ensures your data is safe even if you remove and recreate the container.

## 🖥️ Local Production Setup (Without Docker)

For personal use or deployment on a single server without Docker.

### System Requirements - Local Production Setup

* Ubuntu 22.04+ / Debian 11+

* Python 3.12+

* `uv` package manager

* `systemd` for service management

* (Optional) `nginx` for reverse proxying

### Installation Steps

1. **Install Dependencies & Clone Repo:**
    Follow the local installation steps in the [Getting Started Guide](./docs/user/getting-started.md). Ensure you have installed `uv` and cloned the repository.

2. **Install Application Dependencies:**

    ```bash
    uv sync
    ```

3. **Initialize Database:**

    ```bash
    uv run python -m src.seed seed
    ```

4. **Create a `systemd` Service:**
    Create a service file to manage the application process and ensure it restarts on failure or reboot.

    ```bash
    sudo tee /etc/systemd/system/ai-job-scraper.service > /dev/null <<EOF
    [Unit]
    Description=AI Job Scraper Streamlit App
    After=network.target

    [Service]
    Type=simple
    User=<your_username>
    Group=<your_group>
    WorkingDirectory=<path_to_ai-job-scraper_repo>
    ExecStart=$(which streamlit) run src/main.py --server.port=8501 --server.address=127.0.0.1
    Restart=on-failure
    RestartSec=5s

    [Install]
    WantedBy=multi-user.target
    EOF
    ```

    **Note:** Replace `<your_username>`, `<your_group>`, and `<path_to_ai-job-scraper_repo>` with your actual values.

5. **Enable and Start the Service:**

    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable ai-job-scraper.service
    sudo systemctl start ai-job-scraper.service
    sudo systemctl status ai-job-scraper.service
    ```

6. **(Optional) Configure Nginx as a Reverse Proxy:**
    Using Nginx allows you to easily add SSL/TLS, custom domains, and rate limiting. A basic configuration looks like this:

    ```nginx
    # /etc/nginx/sites-available/ai-job-scraper
    server {
        listen 80;
        server_name your-domain.com;

        location / {
            proxy_pass http://127.0.0.1:8501;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
        }
    }
    ```

## ☁️ Cloud Deployment

The containerized setup can be deployed to any cloud provider that supports Docker containers.

* **AWS:** Use Amazon ECS with Fargate for a serverless container deployment.

* **Google Cloud:** Use Google Cloud Run for a fully managed, scalable deployment.

* **DigitalOcean:** Use the App Platform for a simple, Git-based deployment workflow.

When deploying to the cloud, the current SQLite + DuckDB architecture is optimized for single-user deployments up to 500K+ records. For multi-user production environments, consider migrating to PostgreSQL with DuckDB analytics:

```bash
# Optional: PostgreSQL for multi-user production
DB_URL=postgresql://user:pass@host:port/dbname
```

**Current Architecture Benefits**:

* SQLite handles primary operations (5-15ms queries)
* DuckDB sqlite_scanner provides zero-ETL analytics
* Automatic scaling when p95 query latency >500ms
* No separate database file maintenance required

## 🔒 Security Hardening

* **Run as Non-Root User:** The provided `Dockerfile` already creates and uses a non-root `appuser` for enhanced security.

* **Manage Secrets:** Never hardcode API keys. Use the `.env` file for Docker/local deployments or your cloud provider's secret manager (e.g., AWS Secrets Manager, Google Secret Manager).

* **Use a Reverse Proxy:** Always place the application behind a reverse proxy like Nginx or a cloud load balancer to handle SSL/TLS termination and provide an extra layer of security.

* **Firewall:** Configure a firewall to only allow traffic on necessary ports (e.g., 80 for HTTP, 443 for HTTPS).

## 📊 Analytics & Monitoring Configuration

### DuckDB Analytics Engine

The application uses DuckDB with sqlite_scanner for high-performance analytics:

```bash
# DuckDB is automatically configured - no manual setup required
# sqlite_scanner extension loads automatically
# Direct SQLite scanning eliminates ETL processes
```

**Analytics Features Available**:

* Job market trend analysis
* Company hiring metrics  
* Salary range analytics
* Real-time cost monitoring ($50 monthly budget)

### Performance Characteristics

* **SQLite Operations**: 5-15ms queries for 1K jobs, 50-300ms for 500K jobs
* **DuckDB Analytics**: Activates automatically when p95 latency >500ms
* **Cost Tracking**: Real-time budget monitoring with alerts at 80% and 100%
* **Memory Usage**: DuckDB creates temporary files, automatically cleaned up

### Environment Variables

```bash
# Optional analytics configuration
ENABLE_ANALYTICS=true           # Enable analytics dashboard
COST_BUDGET_USD=50             # Monthly budget limit
DUCKDB_MEMORY_LIMIT=512MB      # DuckDB memory limit (optional)
```

### Troubleshooting Analytics

```bash
# Check DuckDB status
docker-compose exec app uv run python -c "import duckdb; print('DuckDB available')"

# Verify sqlite_scanner
docker-compose exec app uv run python -c "
import duckdb
conn = duckdb.connect(':memory:')
conn.execute('INSTALL sqlite_scanner')
conn.execute('LOAD sqlite_scanner')
print('sqlite_scanner ready')
"

# Check database files
docker-compose exec app ls -la *.db
```
