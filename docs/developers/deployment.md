# 🚀 Deployment Guide: AI Job Scraper

This guide provides comprehensive strategies for deploying the AI Job Scraper in a production environment.

## 🐳 Docker Deployment (Recommended)

Using Docker is the recommended method for a clean, repeatable, and secure deployment. The project includes a multi-stage `Dockerfile` and a `docker-compose.yml` for this purpose.

### System Requirements - Docker Deployment

* A host machine with Docker and Docker Compose installed.

* 2GB RAM, 1 vCPU, 10GB storage minimum.

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
    The first time you deploy, you need to run the database seeder inside the running container.

    ```bash
    docker-compose exec app uv run python -m src.seed seed
    ```

5. **Verify the Application:**
    The application should now be accessible at `http://localhost:8501`. You can check the status and logs of the running container:

    ```bash
    docker-compose ps
    docker-compose logs -f app
    ```

### Persistent Data

The `docker-compose.yml` file is configured to use a Docker volume (`dbdata`) to persist the SQLite database (`jobs.db`) outside the container. This ensures your data is safe even if you remove and recreate the container.

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

When deploying to the cloud, it is highly recommended to switch the database backend from SQLite to a managed PostgreSQL instance for better performance and reliability. You can do this by setting the `DB_URL` environment variable.

## 🔒 Security Hardening

* **Run as Non-Root User:** The provided `Dockerfile` already creates and uses a non-root `appuser` for enhanced security.

* **Manage Secrets:** Never hardcode API keys. Use the `.env` file for Docker/local deployments or your cloud provider's secret manager (e.g., AWS Secrets Manager, Google Secret Manager).

* **Use a Reverse Proxy:** Always place the application behind a reverse proxy like Nginx or a cloud load balancer to handle SSL/TLS termination and provide an extra layer of security.

* **Firewall:** Configure a firewall to only allow traffic on necessary ports (e.g., 80 for HTTP, 443 for HTTPS).
