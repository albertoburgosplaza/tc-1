#!/bin/bash

# setup.sh - Automated setup script for RAG Chatbot
# Author: Generated for turingchallenge-reto-1
# Version: 1.0.0

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration defaults
DEFAULT_TIMEOUT=30
DOCS_FOLDER="./docs"
LOG_FILE="setup.log"

# Flags
SILENT_MODE=false
SKIP_INGEST=false
FORCE_REINSTALL=false

# Helper functions
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
    log "INFO: $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
    log "SUCCESS: $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
    log "WARNING: $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    log "ERROR: $1"
}

show_help() {
    cat << EOF
Setup Script for RAG Chatbot with Docker Compose

USAGE:
    ./setup.sh [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -s, --silent            Silent mode (minimal output)
    -t, --timeout SECONDS   Timeout for service health checks (default: 30)
    -f, --force             Force reinstall (removes existing containers/volumes)
    --skip-ingest          Skip document ingestion
    --no-docs              Don't create docs folder

REQUIREMENTS:
    GOOGLE_API_KEY          Required environment variable for Gemini AI

EXAMPLES:
    ./setup.sh                              # Standard setup
    ./setup.sh -s                          # Silent setup
    ./setup.sh -f                          # Force clean reinstall

EOF
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -s|--silent)
                SILENT_MODE=true
                shift
                ;;
            -t|--timeout)
                DEFAULT_TIMEOUT="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_REINSTALL=true
                shift
                ;;
            --skip-ingest)
                SKIP_INGEST=true
                shift
                ;;
            --no-docs)
                DOCS_FOLDER=""
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

check_system_requirements() {
    print_status "Checking system requirements..."
    
    # Check if running on Windows/WSL
    if [[ -f /proc/version ]] && grep -q Microsoft /proc/version; then
        print_status "Detected WSL environment"
    fi
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df . | awk 'NR==2 {print $4}')
    REQUIRED_SPACE=10485760  # 10GB in KB
    
    if [[ $AVAILABLE_SPACE -lt $REQUIRED_SPACE ]]; then
        print_error "Insufficient disk space. Required: 10GB, Available: $(($AVAILABLE_SPACE/1024/1024))GB"
        exit 1
    fi
    
    # Check available memory (minimum 4GB)
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
    if [[ $AVAILABLE_MEMORY -lt 4096 ]]; then
        print_warning "Low available memory: ${AVAILABLE_MEMORY}MB. Recommended: 4GB+"
    fi
    
    print_success "System requirements check passed"
}

check_docker() {
    print_status "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        echo "Please install Docker from: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker --version &> /dev/null; then
        print_error "Docker is not running or accessible"
        print_status "Try: sudo systemctl start docker"
        exit 1
    fi
    
    # Check Docker version (minimum 20.10)
    DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+' | head -1)
    REQUIRED_VERSION="20.10"
    
    if ! printf '%s\n%s\n' "$REQUIRED_VERSION" "$DOCKER_VERSION" | sort -V -C; then
        print_error "Docker version $DOCKER_VERSION is too old. Required: $REQUIRED_VERSION+"
        exit 1
    fi
    
    print_success "Docker $DOCKER_VERSION detected"
}

check_docker_compose() {
    print_status "Checking Docker Compose installation..."
    
    # Check for docker compose (new) or docker-compose (legacy)
    if command -v "docker compose" &> /dev/null; then
        COMPOSE_CMD="docker compose"
    elif command -v "docker-compose" &> /dev/null; then
        COMPOSE_CMD="docker-compose"
        print_warning "Using legacy docker-compose. Consider upgrading to 'docker compose'"
    else
        print_error "Docker Compose is not installed"
        echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check Docker Compose version (minimum 2.0)
    COMPOSE_VERSION=$($COMPOSE_CMD --version | grep -oP '\d+\.\d+' | head -1)
    REQUIRED_COMPOSE_VERSION="2.0"
    
    if ! printf '%s\n%s\n' "$REQUIRED_COMPOSE_VERSION" "$COMPOSE_VERSION" | sort -V -C; then
        print_warning "Docker Compose version $COMPOSE_VERSION detected. Recommended: $REQUIRED_COMPOSE_VERSION+"
    fi
    
    print_success "Docker Compose $COMPOSE_VERSION detected"
}

setup_directories() {
    if [[ -n "$DOCS_FOLDER" ]]; then
        print_status "Setting up directories..."
        
        if [[ ! -d "$DOCS_FOLDER" ]]; then
            mkdir -p "$DOCS_FOLDER"
            print_success "Created $DOCS_FOLDER directory"
            
            # Add sample README if folder is empty
            cat > "$DOCS_FOLDER/README.md" << 'EOF'
# Documents Folder

Add your PDF documents here for ingestion into the RAG system.

## Supported formats:
- PDF files (.pdf)

## Limitations:
- Maximum file size: 100MB per PDF
- Minimum content: 10 characters per document

## Usage:
1. Copy your PDF files to this folder
2. Run: `docker compose run --rm ingest`
3. Ask questions about your documents in the web interface

EOF
            print_status "Added README.md to docs folder with usage instructions"
        else
            print_status "$DOCS_FOLDER directory already exists"
        fi
        
        # Check if there are any PDFs
        PDF_COUNT=$(find "$DOCS_FOLDER" -name "*.pdf" -type f 2>/dev/null | wc -l)
        if [[ $PDF_COUNT -eq 0 ]]; then
            print_warning "No PDF files found in $DOCS_FOLDER"
            print_status "Add PDF files to $DOCS_FOLDER before running ingestion"
            SKIP_INGEST=true
        else
            print_success "Found $PDF_COUNT PDF file(s) in $DOCS_FOLDER"
        fi
    fi
}

cleanup_existing() {
    if [[ "$FORCE_REINSTALL" == true ]]; then
        print_status "Force reinstall requested - cleaning up existing installation..."
        
        # Stop and remove containers
        $COMPOSE_CMD down --remove-orphans 2>/dev/null || true
        
        # Remove volumes
        $COMPOSE_CMD down -v 2>/dev/null || true
        
        # Remove any dangling images
        docker image prune -f 2>/dev/null || true
        
        print_success "Cleanup completed"
    fi
}

build_and_start_services() {
    print_status "Building and starting services..."
    
    # Pull latest images first
    print_status "Pulling latest images..."
    $COMPOSE_CMD pull qdrant 2>/dev/null || true
    
    # Build and start base services
    print_status "Building custom services (this may take a few minutes)..."
    if [[ "$SILENT_MODE" == true ]]; then
        $COMPOSE_CMD up -d --build qdrant pyexec app > /dev/null 2>&1
    else
        $COMPOSE_CMD up -d --build qdrant pyexec app
    fi
    
    print_success "Services started"
}

wait_for_services() {
    print_status "Waiting for services to become healthy..."
    
    local max_attempts=$((DEFAULT_TIMEOUT * 2))  # 30 seconds = 60 attempts
    local attempt=1
    local services=("qdrant" "pyexec")
    
    while [[ $attempt -le $max_attempts ]]; do
        local all_healthy=true
        
        for service in "${services[@]}"; do
            local health_status
            health_status=$(docker inspect --format='{{.State.Health.Status}}' "$service" 2>/dev/null || echo "unhealthy")
            
            if [[ "$health_status" != "healthy" ]]; then
                all_healthy=false
                break
            fi
        done
        
        if [[ "$all_healthy" == true ]]; then
            print_success "All services are healthy"
            return 0
        fi
        
        if [[ $((attempt % 10)) -eq 0 ]]; then
            print_status "Still waiting for services... (${attempt}/2 seconds elapsed)"
        fi
        
        sleep 0.5
        ((attempt++))
    done
    
    print_error "Services did not become healthy within $DEFAULT_TIMEOUT seconds"
    print_status "Checking service status:"
    $COMPOSE_CMD ps
    
    print_status "Recent logs:"
    $COMPOSE_CMD logs --tail=10 qdrant pyexec app
    
    return 1
}

check_google_api_key() {
    print_status "Checking Google API key..."
    
    if [[ -z "$GOOGLE_API_KEY" ]]; then
        print_error "GOOGLE_API_KEY environment variable is not set"
        print_status "Please set your Google API key:"
        print_status "export GOOGLE_API_KEY='your_api_key_here'"
        print_status "Or add it to your .env file"
        return 1
    fi
    
    print_success "Google API key is configured"
    return 0
}

run_ingestion() {
    if [[ "$SKIP_INGEST" == true ]]; then
        print_status "Skipping document ingestion as requested"
        return 0
    fi
    
    print_status "Running document ingestion..."
    
    # Check if Qdrant is ready
    if ! docker exec qdrant wget --quiet --tries=1 --spider http://localhost:6333/healthz 2>/dev/null; then
        print_error "Qdrant is not ready for ingestion"
        return 1
    fi
    
    # Run ingestion
    if [[ "$SILENT_MODE" == true ]]; then
        $COMPOSE_CMD run --rm ingest > /dev/null 2>&1
    else
        $COMPOSE_CMD run --rm ingest
    fi
    
    if [[ $? -eq 0 ]]; then
        print_success "Document ingestion completed"
    else
        print_error "Document ingestion failed"
        return 1
    fi
}

wait_for_app() {
    print_status "Waiting for application to be ready..."
    
    local max_attempts=60
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8080/health >/dev/null 2>&1; then
            print_success "Application is ready"
            return 0
        fi
        
        if [[ $((attempt % 10)) -eq 0 ]]; then
            print_status "Still waiting for application... (${attempt} seconds elapsed)"
        fi
        
        sleep 1
        ((attempt++))
    done
    
    print_error "Application did not become ready within 60 seconds"
    return 1
}

show_final_status() {
    print_status "Setup completed! Showing final status..."
    
    echo ""
    echo "========================================="
    echo "         RAG CHATBOT SETUP COMPLETE     "
    echo "========================================="
    echo ""
    
    # Show service status
    echo "SERVICE STATUS:"
    $COMPOSE_CMD ps
    echo ""
    
    # Show access URLs
    echo "ACCESS URLS:"
    echo "  🌐 Web Interface:    http://localhost:7860"
    echo "  🔍 Health Check:     http://localhost:8080/health"
    echo "  📊 Qdrant Dashboard: http://localhost:6333/dashboard"
    echo ""
    
    # Show LLM information
    echo "LLM PROVIDER:"
    echo "  🤖 Google Gemini 2.5 Flash Lite (API-based)"
    echo "  📊 Embeddings: models/embedding-001"
    echo ""
    
    # Show document count
    if [[ -d "$DOCS_FOLDER" ]]; then
        PDF_COUNT=$(find "$DOCS_FOLDER" -name "*.pdf" -type f 2>/dev/null | wc -l)
        echo "DOCUMENTS:"
        echo "  📁 Documents folder: $DOCS_FOLDER"
        echo "  📄 PDF files found: $PDF_COUNT"
        echo ""
    fi
    
    # Show usage instructions
    echo "USAGE:"
    echo "  1. Open http://localhost:7860 in your browser"
    echo "  2. Ask questions about your documents"
    echo "  3. Use 'python: expression' for calculations"
    echo "  4. Run 'docker compose logs -f app' to see logs"
    echo ""
    
    # Show management commands
    echo "MANAGEMENT COMMANDS:"
    echo "  Stop:     docker compose down"
    echo "  Restart:  docker compose restart"
    echo "  Logs:     docker compose logs -f [service]"
    echo "  Update:   git pull && ./setup.sh -f"
    echo ""
    
    # Estimate setup time
    local end_time=$(date +%s)
    local elapsed_time=$((end_time - start_time))
    local minutes=$((elapsed_time / 60))
    local seconds=$((elapsed_time % 60))
    
    echo "SETUP TIME: ${minutes}m ${seconds}s"
    echo "========================================="
}

# Main execution
main() {
    # Record start time
    start_time=$(date +%s)
    
    # Initialize log file
    echo "Setup started at $(date)" > "$LOG_FILE"
    
    print_status "Starting RAG Chatbot setup..."
    print_status "Log file: $LOG_FILE"
    
    # Parse command line arguments
    parse_arguments "$@"
    
    # Run setup steps
    check_system_requirements
    check_docker
    check_docker_compose
    check_google_api_key || exit 1
    setup_directories
    cleanup_existing
    build_and_start_services
    
    if ! wait_for_services; then
        print_error "Setup failed: Services did not start properly"
        exit 1
    fi
    
    run_ingestion
    
    if ! wait_for_app; then
        print_error "Setup failed: Application is not responding"
        exit 1
    fi
    
    show_final_status
    
    log "Setup completed successfully in $(($(date +%s) - start_time)) seconds"
}

# Handle interruption
trap 'print_error "Setup interrupted"; exit 1' INT TERM

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi