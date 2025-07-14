#!/bin/bash
# ============================================================================
# ML Enhanced Router with CSP Network Integration - Startup Script
# ============================================================================

set -e  # Exit on any error

# ============================================================================
# CONFIGURATION
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Default configuration
CSP_NETWORK_ENABLED=true
ML_ROUTER_ENABLED=true
WAIT_FOR_NETWORK=10
LOG_LEVEL="INFO"
ENVIRONMENT="development"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

log() {
    echo -e "${CYAN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

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

log_header() {
    echo -e "\n${PURPLE}============================================================================${NC}"
    echo -e "${WHITE}$1${NC}"
    echo -e "${PURPLE}============================================================================${NC}"
}

check_dependencies() {
    log_info "Checking dependencies..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    local python_version=$(python3 --version | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 is required but not installed"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        log_warning "No virtual environment found. Consider creating one with: python3 -m venv venv"
    fi
    
    log_success "Dependencies check completed"
}

install_requirements() {
    log_info "Installing/updating requirements..."
    
    if [ -f "requirements_integrated.txt" ]; then
        pip3 install -r requirements_integrated.txt
        log_success "Requirements installed successfully"
    else
        log_warning "requirements_integrated.txt not found, skipping package installation"
    fi
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create necessary directories
    mkdir -p logs
    mkdir -p data
    mkdir -p ml_router_network_data
    mkdir -p backups
    mkdir -p config_backups
    
    # Copy environment file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            log_info "Created .env file from .env.example"
            log_warning "Please edit .env file with your configuration before running"
        else
            log_warning "No .env file found. Creating minimal configuration..."
            cat > .env << EOF
# Minimal configuration for ML Router with CSP Network
DATABASE_URL=sqlite:///ml_router_network.db
SESSION_SECRET=change-this-secret-in-production
CSP_NETWORK_ENABLED=true
ML_ROUTER_ENABLED=true
LOG_LEVEL=INFO
EOF
        fi
    fi
    
    log_success "Environment setup completed"
}

check_ports() {
    log_info "Checking port availability..."
    
    local ports=(5000 30405 30301)
    local unavailable_ports=()
    
    for port in "${ports[@]}"; do
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            unavailable_ports+=($port)
        elif ss -tuln 2>/dev/null | grep -q ":$port "; then
            unavailable_ports+=($port)
        fi
    done
    
    if [ ${#unavailable_ports[@]} -gt 0 ]; then
        log_warning "The following ports are already in use: ${unavailable_ports[*]}"
        log_warning "You may need to stop other services or change port configuration"
    else
        log_success "All required ports are available"
    fi
}

start_csp_network() {
    if [ "$CSP_NETWORK_ENABLED" != "true" ]; then
        log_info "CSP Network disabled, skipping..."
        return 0
    fi
    
    log_info "Starting Enhanced CSP Network..."
    
    # Check if network_startup.py exists
    if [ -f "network_startup.py" ]; then
        log_info "Using network_startup.py"
        python3 network_startup.py --local-port 30301 --node-name csp-base-node --log-level "$LOG_LEVEL" &
        CSP_NETWORK_PID=$!
        echo $CSP_NETWORK_PID > csp_network.pid
    elif [ -f "enhanced_csp/network/main.py" ]; then
        log_info "Using enhanced_csp.network.main"
        python3 -m enhanced_csp.network.main --local-port 30301 --node-name csp-base-node &
        CSP_NETWORK_PID=$!
        echo $CSP_NETWORK_PID > csp_network.pid
    else
        log_error "No CSP Network startup script found"
        log_error "Expected: network_startup.py or enhanced_csp/network/main.py"
        return 1
    fi
    
    log_info "CSP Network starting with PID: $CSP_NETWORK_PID"
    
    # Wait for network to initialize
    log_info "Waiting ${WAIT_FOR_NETWORK}s for CSP Network to initialize..."
    sleep $WAIT_FOR_NETWORK
    
    # Check if process is still running
    if ! kill -0 $CSP_NETWORK_PID 2>/dev/null; then
        log_error "CSP Network failed to start"
        return 1
    fi
    
    log_success "CSP Network started successfully"
    return 0
}

start_ml_router() {
    if [ "$ML_ROUTER_ENABLED" != "true" ]; then
        log_info "ML Router disabled, skipping..."
        return 0
    fi
    
    log_info "Starting ML Enhanced Router with Network Integration..."
    
    # Check if ml_router_network.py exists
    if [ -f "ml_router_network.py" ]; then
        python3 ml_router_network.py &
        ML_ROUTER_PID=$!
        echo $ML_ROUTER_PID > ml_router.pid
        log_info "ML Router starting with PID: $ML_ROUTER_PID"
    else
        log_error "ml_router_network.py not found"
        return 1
    fi
    
    # Wait for ML Router to initialize
    log_info "Waiting 5s for ML Router to initialize..."
    sleep 5
    
    # Check if process is still running
    if ! kill -0 $ML_ROUTER_PID 2>/dev/null; then
        log_error "ML Router failed to start"
        return 1
    fi
    
    log_success "ML Router started successfully"
    return 0
}

check_system_status() {
    log_info "Checking system status..."
    
    # Check CSP Network
    if [ -f "csp_network.pid" ]; then
        local csp_pid=$(cat csp_network.pid)
        if kill -0 $csp_pid 2>/dev/null; then
            log_success "CSP Network is running (PID: $csp_pid)"
        else
            log_error "CSP Network process not found"
        fi
    fi
    
    # Check ML Router
    if [ -f "ml_router.pid" ]; then
        local ml_pid=$(cat ml_router.pid)
        if kill -0 $ml_pid 2>/dev/null; then
            log_success "ML Router is running (PID: $ml_pid)"
        else
            log_error "ML Router process not found"
        fi
    fi
    
    # Test HTTP endpoints
    log_info "Testing HTTP endpoints..."
    
    # Test ML Router health
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        log_success "ML Router HTTP endpoint is responding"
    else
        log_warning "ML Router HTTP endpoint is not responding"
    fi
    
    # Test network status
    if curl -s http://localhost:5000/api/network/status > /dev/null 2>&1; then
        log_success "Network integration endpoint is responding"
    else
        log_warning "Network integration endpoint is not responding"
    fi
}

stop_system() {
    log_info "Stopping integrated system..."
    
    # Stop ML Router
    if [ -f "ml_router.pid" ]; then
        local ml_pid=$(cat ml_router.pid)
        if kill -0 $ml_pid 2>/dev/null; then
            log_info "Stopping ML Router (PID: $ml_pid)..."
            kill $ml_pid
            sleep 2
            if kill -0 $ml_pid 2>/dev/null; then
                log_warning "Force killing ML Router..."
                kill -9 $ml_pid
            fi
        fi
        rm -f ml_router.pid
    fi
    
    # Stop CSP Network
    if [ -f "csp_network.pid" ]; then
        local csp_pid=$(cat csp_network.pid)
        if kill -0 $csp_pid 2>/dev/null; then
            log_info "Stopping CSP Network (PID: $csp_pid)..."
            kill $csp_pid
            sleep 2
            if kill -0 $csp_pid 2>/dev/null; then
                log_warning "Force killing CSP Network..."
                kill -9 $csp_pid
            fi
        fi
        rm -f csp_network.pid
    fi
    
    log_success "System stopped"
}

show_status() {
    log_header "SYSTEM STATUS"
    check_system_status
    
    echo ""
    log_info "Access URLs:"
    echo "  🌐 ML Router Dashboard: http://localhost:5000"
    echo "  🔍 Network Dashboard: http://localhost:5000/network-dashboard"
    echo "  📊 Network Status API: http://localhost:5000/api/network/status"
    echo "  ❤️  Health Check: http://localhost:5000/health"
    echo ""
}

show_logs() {
    log_info "Recent logs:"
    echo ""
    
    if [ -f "ml_router_network.log" ]; then
        echo "=== ML Router Logs (last 20 lines) ==="
        tail -20 ml_router_network.log
        echo ""
    fi
    
    if [ -f "csp_network.log" ]; then
        echo "=== CSP Network Logs (last 20 lines) ==="
        tail -20 csp_network.log
        echo ""
    fi
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -f *.pid
    rm -f *.log
    rm -rf __pycache__
    rm -rf .pytest_cache
    log_success "Cleanup completed"
}

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

cleanup_on_exit() {
    log_info "Received exit signal, cleaning up..."
    stop_system
    exit 0
}

trap cleanup_on_exit SIGINT SIGTERM

# ============================================================================
# COMMAND LINE ARGUMENT PARSING
# ============================================================================

show_help() {
    cat << EOF
ML Enhanced Router with CSP Network Integration - Startup Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    start       Start the integrated system (default)
    stop        Stop the integrated system
    restart     Restart the integrated system
    status      Show system status
    logs        Show recent logs
    install     Install dependencies
    setup       Setup environment
    cleanup     Clean up temporary files
    health      Run health checks

Options:
    --no-network        Disable CSP Network
    --no-ml-router      Disable ML Router
    --wait-time=N       Wait time for network initialization (default: 10)
    --log-level=LEVEL   Log level (DEBUG, INFO, WARNING, ERROR)
    --environment=ENV   Environment (development, staging, production)
    --help              Show this help message

Examples:
    $0 start                    # Start both CSP Network and ML Router
    $0 start --no-network       # Start only ML Router
    $0 start --wait-time=15     # Start with 15s network wait time
    $0 status                   # Show current system status
    $0 logs                     # Show recent logs

Environment Variables:
    CSP_NETWORK_ENABLED=true/false
    ML_ROUTER_ENABLED=true/false
    LOG_LEVEL=INFO
    ENVIRONMENT=development

For more information, see the integration documentation.
EOF
}

# Parse command line arguments
COMMAND="start"
while [[ $# -gt 0 ]]; do
    case $1 in
        start|stop|restart|status|logs|install|setup|cleanup|health)
            COMMAND="$1"
            shift
            ;;
        --no-network)
            CSP_NETWORK_ENABLED=false
            shift
            ;;
        --no-ml-router)
            ML_ROUTER_ENABLED=false
            shift
            ;;
        --wait-time=*)
            WAIT_FOR_NETWORK="${1#*=}"
            shift
            ;;
        --log-level=*)
            LOG_LEVEL="${1#*=}"
            shift
            ;;
        --environment=*)
            ENVIRONMENT="${1#*=}"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# ============================================================================
# MAIN EXECUTION
# ============================================================================

main() {
    log_header "ML ENHANCED ROUTER WITH CSP NETWORK INTEGRATION"
    
    case $COMMAND in
        start)
            log_info "Starting integrated system..."
            check_dependencies
            setup_environment
            check_ports
            
            if start_csp_network && start_ml_router; then
                log_success "Integrated system started successfully!"
                show_status
                
                log_info "System is running. Press Ctrl+C to stop."
                
                # Keep script running and handle signals
                while true; do
                    sleep 10
                    # Check if processes are still running
                    if [ "$CSP_NETWORK_ENABLED" = "true" ] && [ -f "csp_network.pid" ]; then
                        local csp_pid=$(cat csp_network.pid)
                        if ! kill -0 $csp_pid 2>/dev/null; then
                            log_error "CSP Network process died unexpectedly"
                            break
                        fi
                    fi
                    
                    if [ "$ML_ROUTER_ENABLED" = "true" ] && [ -f "ml_router.pid" ]; then
                        local ml_pid=$(cat ml_router.pid)
                        if ! kill -0 $ml_pid 2>/dev/null; then
                            log_error "ML Router process died unexpectedly"
                            break
                        fi
                    fi
                done
            else
                log_error "Failed to start integrated system"
                stop_system
                exit 1
            fi
            ;;
        stop)
            stop_system
            ;;
        restart)
            log_info "Restarting integrated system..."
            stop_system
            sleep 2
            $0 start
            ;;
        status)
            show_status
            ;;
        logs)
            show_logs
            ;;
        install)
            check_dependencies
            install_requirements
            ;;
        setup)
            setup_environment
            log_success "Environment setup completed"
            ;;
        cleanup)
            cleanup
            ;;
        health)
            log_header "HEALTH CHECK"
            check_dependencies
            check_ports
            check_system_status
            ;;
        *)
            log_error "Unknown command: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"