#!/bin/bash
# vLLM Server Stop Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log "Stopping vLLM Server..."

# Change to project root
cd "$PROJECT_ROOT"

# Check if containers are running
if ! docker-compose -f docker-compose.vllm.yml ps --services --filter "status=running" | grep -q .; then
    warning "No running vLLM containers found"
    exit 0
fi

# Stop containers gracefully
log "Stopping containers gracefully..."
docker-compose -f docker-compose.vllm.yml stop

# Wait a moment for graceful shutdown
sleep 5

# Remove containers
log "Removing containers..."
docker-compose -f docker-compose.vllm.yml down

# Check if containers are fully stopped
if docker-compose -f docker-compose.vllm.yml ps --services --filter "status=running" | grep -q .; then
    warning "Some containers are still running. Forcing shutdown..."
    docker-compose -f docker-compose.vllm.yml down --remove-orphans
fi

success "vLLM server stopped successfully"

# Show cleanup options
echo ""
echo "Optional cleanup commands:"
echo "  - Remove volumes: docker-compose -f docker-compose.vllm.yml down -v"
echo "  - Remove images: docker-compose -f docker-compose.vllm.yml down --rmi all"
echo "  - Clean up logs: rm -rf logs/vllm/*"