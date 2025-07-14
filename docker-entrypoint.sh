#!/bin/bash
# ============================================================================
# Docker Entrypoint Script for ML Enhanced Router with CSP Network Integration
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default values
: ${DATABASE_URL:="sqlite:///ml_router_network.db"}
: ${CSP_NETWORK_ENABLED:="true"}
: ${ML_ROUTER_ENABLED:="true"}
: ${LOG_LEVEL:="INFO"}
: ${WAIT_FOR_SERVICES:="true"}
: ${WAIT_TIMEOUT:="60"}

log_info "Starting ML Enhanced Router with CSP Network Integration"
log_info "Container ID: $(hostname)"
log_info "Python version: $(python --version)"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

wait_for_service() {
    local host=$1
    local port=$2
    local service_name=$3
    local timeout=${4:-30}
    
    log_info "Waiting for $service_name at $host:$port..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" >/dev/null 2>&1; then
            log_success "$service_name is available"
            return 0
        fi
        
        if [ $i -eq $timeout ]; then
            log_error "Timeout waiting for $service_name at $host:$port"
            return 1
        fi
        
        sleep 1
    done
}

check_database_connection() {
    log_info "Checking database connection..."
    
    # Extract database type from URL
    if [[ $DATABASE_URL == sqlite* ]]; then
        log_info "Using SQLite database"
        # Create database directory if it doesn't exist
        db_dir=$(dirname "${DATABASE_URL#sqlite:///}")
        mkdir -p "$db_dir"
        return 0
    elif [[ $DATABASE_URL == postgresql* ]]; then
        log_info "Using PostgreSQL database"
        # Extract host and port from PostgreSQL URL
        # Format: postgresql://user:pass@host:port/db
        local db_host=$(echo $DATABASE_URL | sed -n 's|postgresql://[^@]*@\([^:]*\):.*|\1|p')
        local db_port=$(echo $DATABASE_URL | sed -n 's|postgresql://[^@]*@[^:]*:\([0-9]*\)/.*|\1|p')
        
        if [ -n "$db_host" ] && [ -n "$db_port" ]; then
            wait_for_service "$db_host" "$db_port" "PostgreSQL" 30
        else
            log_warning "Could not parse PostgreSQL URL for connection check"
        fi
    else
        log_warning "Unknown database type in DATABASE_URL"
    fi
}

check_redis_connection() {
    if [ -n "$REDIS_URL" ]; then
        log_info "Checking Redis connection..."
        # Extract host and port from Redis URL
        # Format: redis://host:port/db
        local redis_host=$(echo $REDIS_URL | sed -n 's|redis://\([^:]*\):.*|\1|p')
        local redis_port=$(echo $REDIS_URL | sed -n 's|redis://[^:]*:\([0-9]*\)/.*|\1|p')
        
        if [ -n "$redis_host" ] && [ -n "$redis_port" ]; then
            wait_for_service "$redis_host" "$redis_port" "Redis" 15
        else
            log_warning "Could not parse Redis URL for connection check"
        fi
    else
        log_info "Redis not configured, skipping connection check"
    fi
}

setup_directories() {
    log_info "Setting up application directories..."
    
    # Create necessary directories
    mkdir -p /app/ml_router_network_data
    mkdir -p /app/logs
    mkdir -p /app/backups
    mkdir -p /app/models
    mkdir -p /app/config
    
    log_success "Directories created"
}

validate_configuration() {
    log_info "Validating configuration..."
    
    # Check required environment variables
    local missing_vars=()
    
    if [ -z "$DATABASE_URL" ]; then
        missing_vars+=("DATABASE_URL")
    fi
    
    if [ -z "$SESSION_SECRET" ]; then
        log_warning "SESSION_SECRET not set, using default (not recommended for production)"
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi
    
    # Validate ports
    if [ -n "$CSP_NETWORK_PORT" ]; then
        if [ "$CSP_NETWORK_PORT" -lt 1024 ] || [ "$CSP_NETWORK_PORT" -gt 65535 ]; then
            log_error "CSP_NETWORK_PORT must be between 1024 and 65535"
            return 1
        fi
    fi
    
    if [ -n "$ML_ROUTER_PORT" ]; then
        if [ "$ML_ROUTER_PORT" -lt 1024 ] || [ "$ML_ROUTER_PORT" -gt 65535 ]; then
            log_error "ML_ROUTER_PORT must be between 1024 and 65535"
            return 1
        fi
    fi
    
    log_success "Configuration validation passed"
}

initialize_database() {
    log_info "Initializing database..."
    
    # Run database initialization if needed
    if command -v flask >/dev/null 2>&1; then
        log_info "Running Flask database initialization..."
        flask db upgrade 2>/dev/null || log_warning "Flask db upgrade failed or not needed"
    fi
    
    log_success "Database initialization completed"
}

check_ai_providers() {
    log_info "Checking AI provider configurations..."
    
    local configured_providers=0
    
    if [ -n "$OPENAI_API_KEY" ]; then
        log_info "OpenAI API key configured"
        configured_providers=$((configured_providers + 1))
    fi
    
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        log_info "Anthropic API key configured