#!/bin/bash
# vLLM Server Startup Script
# Based on ADR-010 specifications for production deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_FILE="${PROJECT_ROOT}/logs/vllm/startup.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

# Create log directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

log "Starting vLLM Server for AI Job Scraper..."

# Check prerequisites
log "Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker not found. Please install Docker first."
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    error "Docker Compose not found. Please install Docker Compose first."
fi

# Check NVIDIA Docker runtime
if ! docker info | grep -q nvidia; then
    warning "NVIDIA Docker runtime not detected. GPU support may not work."
fi

# Check GPU availability
if ! nvidia-smi &> /dev/null; then
    warning "nvidia-smi not available. GPU support may not work."
else
    log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
fi

# Set default API key if not provided
if [ -z "$VLLM_API_KEY" ]; then
    export VLLM_API_KEY="vllm-server-$(date +%s)"
    warning "VLLM_API_KEY not set. Using generated key: $VLLM_API_KEY"
fi

# Change to project root
cd "$PROJECT_ROOT"

# Start vLLM server
log "Starting vLLM server with Docker Compose..."
docker-compose -f docker-compose.vllm.yml up -d

# Wait for server to start
log "Waiting for vLLM server to be ready..."
TIMEOUT=300
COUNTER=0

while [ $COUNTER -lt $TIMEOUT ]; do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        break
    fi
    sleep 5
    COUNTER=$((COUNTER + 5))
    log "Waiting... ($COUNTER/${TIMEOUT}s)"
done

if [ $COUNTER -ge $TIMEOUT ]; then
    error "vLLM server failed to start within ${TIMEOUT} seconds"
fi

success "vLLM server is ready!"

# Verify API endpoints
log "Verifying API endpoints..."

# Check models endpoint
if curl -sf -H "Authorization: Bearer $VLLM_API_KEY" http://localhost:8000/v1/models > /dev/null; then
    success "Models endpoint is accessible"
else
    warning "Models endpoint is not accessible"
fi

# Show server info
log "vLLM Server Information:"
echo "  - Server URL: http://localhost:8000"
echo "  - API Key: $VLLM_API_KEY"
echo "  - Health Check: http://localhost:8000/health"
echo "  - Models API: http://localhost:8000/v1/models"
echo "  - OpenAI Compatible: http://localhost:8000/v1/chat/completions"

# Show container status
log "Container Status:"
docker-compose -f docker-compose.vllm.yml ps

# Show recent logs
log "Recent server logs:"
docker-compose -f docker-compose.vllm.yml logs --tail=20 vllm-server

success "vLLM server startup complete!"
log "Use 'docker-compose -f docker-compose.vllm.yml logs -f vllm-server' to follow logs"
log "Use 'scripts/stop_vllm.sh' to stop the server"