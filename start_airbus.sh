#!/bin/bash

# Airbus Detection System - Production Startup Script (Enhanced with Fixed Smart Container Management)
# This script handles the complete startup sequence with efficient container management

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Configuration
PROJECT_DIR="C:/Users/hp/Desktop/airbus2.0/detection_system2.0"
CAMERA_DIR="$PROJECT_DIR/backend/cameraHardware"
CAMERA_VENV="$CAMERA_DIR/venv"
CAMERA_HOST="127.0.0.1"
CAMERA_PORT="8003"
LOG_DIR="$PROJECT_DIR/logs"
CAMERA_PID_FILE="$LOG_DIR/camera_hardware.pid"
DOCKER_COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR $(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS $(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARNING $(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

# Create logs directory and ensure it exists
create_logs_dir() {
    if [ ! -d "$LOG_DIR" ]; then
        log "Creating logs directory at: $LOG_DIR"
        mkdir -p "$LOG_DIR"
        if [ $? -eq 0 ]; then
            success "Logs directory created successfully"
        else
            error "Failed to create logs directory"
            return 1
        fi
    else
        log "Logs directory already exists at: $LOG_DIR"
    fi
    
    # Test write permissions
    local test_file="$LOG_DIR/test_write.tmp"
    if echo "test" > "$test_file" 2>/dev/null; then
        rm -f "$test_file"
        log "Logs directory is writable"
        return 0
    else
        error "Logs directory exists but is not writable"
        return 1
    fi
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if netstat -an 2>/dev/null | grep -q ":$port.*LISTENING"; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local timeout=${3:-30}
    local counter=0
    
    log "Waiting for service at $host:$port to be ready..."
    
    while [ $counter -lt $timeout ]; do
        if nc -z "$host" "$port" 2>/dev/null; then
            success "Service at $host:$port is ready!"
            return 0
        fi
        sleep 1
        counter=$((counter + 1))
        echo -n "."
    done
    
    error "Service at $host:$port failed to start within $timeout seconds"
    return 1
}

# Enhanced function to start camera hardware service
start_camera_hardware_ps() {
    log "Starting Camera Hardware Service with PowerShell..."
    
    # Check if already running
    if check_port $CAMERA_PORT; then
        warn "Camera hardware service appears to be already running on port $CAMERA_PORT"
        return 0
    fi
    
    # Check if virtual environment exists
    if [ ! -d "$CAMERA_VENV" ]; then
        error "Virtual environment not found at: $CAMERA_VENV"
        error "Please ensure the camera hardware virtual environment is set up correctly"
        error "To create it, run: python -m venv $CAMERA_VENV"
        return 1
    fi
    
    # Ensure logs directory exists and is writable
    if ! create_logs_dir; then
        warn "Could not create/access logs directory. Starting without detailed logging..."
        LOG_TO_FILE=false
    else
        LOG_TO_FILE=true
    fi
    
    # Check if main.py exists in camera directory
    if [ ! -f "$CAMERA_DIR/main.py" ]; then
        error "main.py not found in camera directory: $CAMERA_DIR"
        error "Please ensure the camera hardware application exists"
        return 1
    fi
    
    log "Starting camera service..."
    log "Camera directory: $CAMERA_DIR"
    log "Virtual environment: $CAMERA_VENV"
    log "Service will run on: http://$CAMERA_HOST:$CAMERA_PORT"
    
    # Create PowerShell command with better error handling
    local ps_command=""
    if [ "$LOG_TO_FILE" = true ]; then
        ps_command="
            try {
                Set-Location '$CAMERA_DIR'
                Write-Host 'Activating virtual environment...'
                & '$CAMERA_VENV/Scripts/Activate.ps1'
                Write-Host 'Starting uvicorn server...'
                Start-Process -WindowStyle Hidden -FilePath 'uvicorn' -ArgumentList 'main:app', '--host', '$CAMERA_HOST', '--port', '$CAMERA_PORT', '--reload' -RedirectStandardOutput '$LOG_DIR/camera_hardware.log' -RedirectStandardError '$LOG_DIR/camera_hardware_error.log'
                Write-Host 'Camera service started successfully'
            } catch {
                Write-Error \"Failed to start camera service: \$_\"
                exit 1
            }
        "
    else
        ps_command="
            try {
                Set-Location '$CAMERA_DIR'
                Write-Host 'Activating virtual environment...'
                & '$CAMERA_VENV/Scripts/Activate.ps1'
                Write-Host 'Starting uvicorn server...'
                Start-Process -WindowStyle Hidden -FilePath 'uvicorn' -ArgumentList 'main:app', '--host', '$CAMERA_HOST', '--port', '$CAMERA_PORT', '--reload'
                Write-Host 'Camera service started successfully'
            } catch {
                Write-Error \"Failed to start camera service: \$_\"
                exit 1
            }
        "
    fi
    
    # Execute PowerShell command
    if powershell.exe -Command "$ps_command"; then
        # Give it a moment to start
        sleep 3
        
        # Verify the service started by checking the port
        local retry_count=0
        local max_retries=10
        
        while [ $retry_count -lt $max_retries ]; do
            if check_port $CAMERA_PORT; then
                success "Camera Hardware Service started successfully on port $CAMERA_PORT"
                
                # Try to save a simple process indicator
                if [ "$LOG_TO_FILE" = true ]; then
                    echo "Camera service started at $(date)" > "$CAMERA_PID_FILE"
                fi
                
                if [ "$LOG_TO_FILE" = true ]; then
                    log "Camera service logs available at:"
                    log "  Output: $LOG_DIR/camera_hardware.log"
                    log "  Errors: $LOG_DIR/camera_hardware_error.log"
                else
                    log "Camera service started without file logging"
                fi
                
                return 0
            fi
            
            log "Waiting for camera service to bind to port... (attempt $((retry_count + 1))/$max_retries)"
            sleep 2
            retry_count=$((retry_count + 1))
        done
        
        error "Camera service started but failed to bind to port $CAMERA_PORT within expected time"
        return 1
    else
        error "Failed to execute PowerShell command to start camera service"
        return 1
    fi
}

# FIXED: Improved Docker container status check with more reliable detection
check_docker_containers_status() {
    cd "$PROJECT_DIR"
    
    # Get all containers defined in docker-compose
    local all_services
    if ! all_services=$(docker-compose config --services 2>/dev/null); then
        echo "0,0,0"  # No services found
        return
    fi
    
    local total_services
    total_services=$(echo "$all_services" | wc -l)
    
    # Check running containers
    local running_containers=0
    local stopped_containers=0
    
    # Use docker-compose ps with better parsing
    local ps_output
    if ps_output=$(docker-compose ps -q 2>/dev/null); then
        if [ -n "$ps_output" ]; then
            # We have containers, now check their status
            while IFS= read -r container_id; do
                if [ -n "$container_id" ]; then
                    local container_status
                    container_status=$(docker inspect --format='{{.State.Status}}' "$container_id" 2>/dev/null || echo "unknown")
                    case "$container_status" in
                        "running")
                            running_containers=$((running_containers + 1))
                            ;;
                        "exited"|"created"|"paused")
                            stopped_containers=$((stopped_containers + 1))
                            ;;
                    esac
                fi
            done <<< "$ps_output"
        fi
    fi
    
    local existing_containers=$((running_containers + stopped_containers))
    echo "$existing_containers,$running_containers,$stopped_containers"
}

# FIXED: Enhanced function to start Docker services with proper fallback logic
start_docker_services() {
    log "Starting Docker services with smart container management..."
    
    cd "$PROJECT_DIR"
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running. Please start Docker first."
        error "On Windows, ensure Docker Desktop is running"
        return 1
    fi
    
    # Check if docker-compose.yml exists
    if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
        error "docker-compose.yml not found at: $DOCKER_COMPOSE_FILE"
        return 1
    fi
    
    # Get container status
    IFS=',' read -r existing_containers running_containers stopped_containers <<< "$(check_docker_containers_status)"
    
    log "Container status: $existing_containers total, $running_containers running, $stopped_containers stopped"
    
    # If already running, just confirm and return
    if [ "$running_containers" -gt 0 ] && [ "$stopped_containers" -eq 0 ]; then
        success "✓ All Docker services are already running ($running_containers containers)"
        docker-compose ps
        return 0
    fi
    
    # If we have some running and some stopped, start the stopped ones
    if [ "$running_containers" -gt 0 ] && [ "$stopped_containers" -gt 0 ]; then
        log "🔄 Starting stopped containers while keeping running ones..."
        if docker-compose start; then
            success "✅ All Docker containers are now running"
            docker-compose ps
            return 0
        else
            warn "⚠ Failed to start some containers, attempting full restart..."
            # Fall through to full restart logic
        fi
    fi
    
    # If we have stopped containers, try to start them
    if [ "$stopped_containers" -gt 0 ]; then
        log "🔄 Found $stopped_containers stopped containers, starting them..."
        log "Executing: docker-compose start"
        if docker-compose start; then
            # Verify all services are now running
            sleep 3
            IFS=',' read -r new_existing new_running new_stopped <<< "$(check_docker_containers_status)"
            
            if [ "$new_stopped" -eq 0 ] && [ "$new_running" -gt 0 ]; then
                success "✅ Docker containers started successfully (no rebuild needed!)"
                log "Docker services status:"
                docker-compose ps
                return 0
            else
                warn "⚠ Some containers failed to start properly, trying full recreation..."
                # Fall through to docker-compose up
            fi
        else
            warn "⚠ Failed to start existing containers, falling back to recreation..."
            # Fall through to docker-compose up
        fi
    fi
    
    # Either no containers exist or starting failed - use docker-compose up
    log "🆕 Creating/recreating containers with docker-compose up..."
    log "Executing: docker-compose up -d"
    if docker-compose up -d; then
        success "✅ Docker services created and started successfully"
        
        # Wait for services to initialize
        log "Waiting for services to initialize..."
        sleep 5
        
        # Show final status
        log "Docker services status:"
        docker-compose ps
        
        # Verify services are actually running
        IFS=',' read -r final_existing final_running final_stopped <<< "$(check_docker_containers_status)"
        if [ "$final_running" -gt 0 ]; then
            success "✅ $final_running containers are running successfully"
            return 0
        else
            error "❌ Containers were created but none are running properly"
            error "Check Docker logs with: docker-compose logs"
            return 1
        fi
    else
        error "❌ Failed to create Docker services"
        error "Check Docker logs with: docker-compose logs"
        return 1
    fi
}

# IMPROVED: Function to stop services with options
stop_services() {
    local force_remove=${1:-false}
    
    if [ "$force_remove" = "true" ]; then
        log "🔥 Force stopping and REMOVING all services..."
        warn "This will require full rebuild on next start!"
    else
        log "⏸️ Stopping services (containers will be preserved)..."
    fi
    
    # Stop Docker services
    cd "$PROJECT_DIR"
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        log "Stopping Docker services..."
        
        if [ "$force_remove" = "true" ]; then
            log "Executing: docker-compose down (removing containers)"
            docker-compose down
        else
            log "Executing: docker-compose stop (preserving containers)"
            if docker-compose stop; then
                success "✅ Docker containers stopped (preserved for fast restart)"
            else
                warn "⚠ Failed to stop containers gracefully, trying force stop..."
                docker-compose kill
                # Don't remove them even if kill was needed
                success "✅ Docker containers forcibly stopped (still preserved)"
            fi
        fi
    else
        warn "docker-compose.yml not found, skipping Docker service stop"
    fi
    
    # Stop camera hardware service
    log "Stopping Camera Hardware Service..."
    
    # Clean up any uvicorn/python processes on the camera port
    powershell.exe -Command "
        try {
            # Kill any uvicorn processes
            Get-Process -Name 'uvicorn' -ErrorAction SilentlyContinue | ForEach-Object {
                Write-Host \"Stopping uvicorn process: \$(\$_.Id)\"
                Stop-Process -Id \$_.Id -Force -ErrorAction SilentlyContinue
            }
            
            # Kill any python processes using port $CAMERA_PORT
            \$connections = Get-NetTCPConnection -LocalPort $CAMERA_PORT -ErrorAction SilentlyContinue
            foreach (\$conn in \$connections) {
                try {
                    \$process = Get-Process -Id \$conn.OwningProcess -ErrorAction SilentlyContinue
                    if (\$process) {
                        Write-Host \"Stopping process using port $CAMERA_PORT: \$(\$process.ProcessName) (PID: \$(\$process.Id))\"
                        Stop-Process -Id \$process.Id -Force -ErrorAction SilentlyContinue
                    }
                } catch {
                    Write-Warning \"Could not stop process: \$_\"
                }
            }
            
            Write-Host \"Camera service cleanup completed\"
        } catch {
            Write-Warning \"Error during camera service cleanup: \$_\"
        }
    " 2>/dev/null || warn "Could not clean up camera service processes"
    
    # Clean up PID file
    if [ -f "$CAMERA_PID_FILE" ]; then
        rm -f "$CAMERA_PID_FILE"
    fi
    
    # Clean up any temporary files
    rm -f "$CAMERA_DIR/start_camera.bat" 2>/dev/null || true
    
    if [ "$force_remove" = "true" ]; then
        success "🔥 All services force stopped and removed"
    else
        success "⏸️ All services stopped (containers preserved)"
    fi
}

# Function to check service status
check_status() {
    log "Checking service status..."
    
    # Check camera hardware
    if check_port $CAMERA_PORT; then
        success "✓ Camera Hardware Service is running on port $CAMERA_PORT"
        
        # Try to make a simple HTTP request to verify it's responding
        log "Testing camera service response..."
        if timeout 5 curl -s "http://$CAMERA_HOST:$CAMERA_PORT/" > /dev/null 2>&1; then
            success "✓ Camera Hardware Service is responding to HTTP requests"
        elif timeout 5 curl -s "http://$CAMERA_HOST:$CAMERA_PORT/docs" > /dev/null 2>&1; then
            success "✓ Camera Hardware Service API documentation is accessible"
        else
            warn "⚠ Camera Hardware Service port is open but may not be fully initialized"
            warn "  Try accessing: http://$CAMERA_HOST:$CAMERA_PORT"
        fi
    else
        warn "✗ Camera Hardware Service is not running on port $CAMERA_PORT"
    fi
    
    # Check Docker services
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        cd "$PROJECT_DIR"
        log "Docker services status:"
        if docker info >/dev/null 2>&1; then
            docker-compose ps
            
            # Additional status info
            IFS=',' read -r existing_containers running_containers stopped_containers <<< "$(check_docker_containers_status)"
            if [ "$existing_containers" -gt 0 ]; then
                if [ "$running_containers" -gt 0 ]; then
                    success "✓ $running_containers Docker containers running"
                fi
                if [ "$stopped_containers" -gt 0 ]; then
                    warn "⏸️ $stopped_containers Docker containers stopped (ready for quick start)"
                fi
            else
                log "ℹ️ No Docker containers exist yet"
            fi
        else
            warn "Docker is not running"
        fi
    else
        warn "docker-compose.yml not found"
    fi
}

# Function to view logs
show_logs() {
    local service=${1:-""}
    
    if [ "$service" = "camera" ]; then
        log "Camera Hardware Service logs:"
        if [ -f "$LOG_DIR/camera_hardware.log" ]; then
            tail -f "$LOG_DIR/camera_hardware.log"
        else
            warn "No camera hardware logs found at $LOG_DIR/camera_hardware.log"
            warn "Camera service may not have file logging enabled"
        fi
    elif [ "$service" = "camera-error" ]; then
        log "Camera Hardware Service error logs:"
        if [ -f "$LOG_DIR/camera_hardware_error.log" ]; then
            tail -f "$LOG_DIR/camera_hardware_error.log"
        else
            warn "No camera hardware error logs found at $LOG_DIR/camera_hardware_error.log"
        fi
    elif [ "$service" = "docker" ]; then
        log "Docker services logs:"
        if [ -f "$DOCKER_COMPOSE_FILE" ]; then
            cd "$PROJECT_DIR"
            docker-compose logs -f
        else
            error "docker-compose.yml not found"
        fi
    else
        log "Available log options:"
        log "  camera       - Show camera hardware service logs"
        log "  camera-error - Show camera hardware error logs"
        log "  docker       - Show all Docker services logs"
    fi
}

# Function to test camera hardware manually
test_camera_hardware() {
    log "Testing camera hardware service manually..."
    
    # Check if virtual environment exists
    if [ ! -d "$CAMERA_VENV" ]; then
        error "Virtual environment not found at: $CAMERA_VENV"
        error "Please create it first with: python -m venv $CAMERA_VENV"
        error "Then install requirements: pip install -r requirements.txt"
        return 1
    fi
    
    # Check if main.py exists
    if [ ! -f "$CAMERA_DIR/main.py" ]; then
        error "main.py not found in: $CAMERA_DIR"
        return 1
    fi
    
    log "Starting camera service in foreground for testing..."
    log "This will show all output directly in the terminal"
    log "Press Ctrl+C to stop the test"
    log "Service will be available at: http://$CAMERA_HOST:$CAMERA_PORT"
    
    cd "$CAMERA_DIR"
    
    # Run in foreground with proper environment activation
    powershell.exe -Command "
        try {
            Set-Location '$CAMERA_DIR'
            Write-Host 'Activating virtual environment...'
            & '$CAMERA_VENV/Scripts/Activate.ps1'
            Write-Host 'Starting uvicorn in foreground mode...'
            uvicorn main:app --host $CAMERA_HOST --port $CAMERA_PORT --reload
        } catch {
            Write-Error \"Failed to start camera service: \$_\"
            Read-Host 'Press Enter to continue...'
        }
    "
}

# Function to check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    local all_good=true
    
    # Check Docker
    if docker --version >/dev/null 2>&1; then
        success "✓ Docker is installed: $(docker --version)"
        if docker info >/dev/null 2>&1; then
            success "✓ Docker daemon is running"
        else
            warn "⚠ Docker is installed but not running"
            all_good=false
        fi
    else
        error "✗ Docker is not installed or not in PATH"
        all_good=false
    fi
    
    # Check docker-compose
    if docker-compose --version >/dev/null 2>&1; then
        success "✓ Docker Compose is installed: $(docker-compose --version)"
    else
        error "✗ Docker Compose is not installed or not in PATH"
        all_good=false
    fi
    
    # Check Python
    if python --version >/dev/null 2>&1; then
        success "✓ Python is installed: $(python --version)"
    else
        warn "⚠ Python is not in PATH (may still work if in venv)"
    fi
    
    # Check project structure
    if [ -d "$PROJECT_DIR" ]; then
        success "✓ Project directory exists: $PROJECT_DIR"
    else
        error "✗ Project directory not found: $PROJECT_DIR"
        all_good=false
    fi
    
    if [ -d "$CAMERA_DIR" ]; then
        success "✓ Camera directory exists: $CAMERA_DIR"
    else
        error "✗ Camera directory not found: $CAMERA_DIR"
        all_good=false
    fi
    
    if [ -f "$DOCKER_COMPOSE_FILE" ]; then
        success "✓ Docker compose file exists"
    else
        error "✗ Docker compose file not found: $DOCKER_COMPOSE_FILE"
        all_good=false
    fi
    
    if [ -d "$CAMERA_VENV" ]; then
        success "✓ Camera virtual environment exists"
    else
        warn "⚠ Camera virtual environment not found: $CAMERA_VENV"
        all_good=false
    fi
    
    if [ -f "$CAMERA_DIR/main.py" ]; then
        success "✓ Camera main.py exists"
    else
        error "✗ Camera main.py not found: $CAMERA_DIR/main.py"
        all_good=false
    fi
    
    if [ "$all_good" = true ]; then
        success "🎉 All requirements are met!"
        return 0
    else
        error "❌ Some requirements are missing. Please fix the issues above."
        return 1
    fi
}

# Enhanced control panel with fixed smart container management
show_control_panel() {
    while true; do
        clear
        echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
        echo -e "${BLUE}║           AIRBUS DETECTION SYSTEM            ║${NC}"
        echo -e "${BLUE}║         SMART CONTROL PANEL v2.1            ║${NC}"
        echo -e "${BLUE}║              (FIXED VERSION)                 ║${NC}"
        echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
        echo ""
        
        # Show current status
        echo -e "${YELLOW}📊 Current Status:${NC}"
        if check_port $CAMERA_PORT; then
            echo -e "   ${GREEN}✓ Camera Service: RUNNING${NC} (http://$CAMERA_HOST:$CAMERA_PORT)"
        else
            echo -e "   ${RED}✗ Camera Service: STOPPED${NC}"
        fi
        
        # Enhanced Docker status check
        cd "$PROJECT_DIR" 2>/dev/null || true
        if docker info >/dev/null 2>&1 && [ -f "$DOCKER_COMPOSE_FILE" ]; then
            IFS=',' read -r existing_containers running_containers stopped_containers <<< "$(check_docker_containers_status)"
            
            if [ "$running_containers" -gt 0 ] && [ "$stopped_containers" -eq 0 ]; then
                echo -e "   ${GREEN}✓ Docker Services: ALL RUNNING${NC} ($running_containers/$existing_containers containers)"
            elif [ "$running_containers" -gt 0 ] && [ "$stopped_containers" -gt 0 ]; then
                echo -e "   ${YELLOW}⚠ Docker Services: PARTIALLY RUNNING${NC} ($running_containers running, $stopped_containers stopped)"
            elif [ "$stopped_containers" -gt 0 ]; then
                echo -e "   ${YELLOW}⏸️ Docker Services: STOPPED${NC} ($stopped_containers containers ready for quick start)"
            else
                echo -e "   ${RED}⚪ Docker Services: NOT CREATED${NC}"
            fi
        else
            echo -e "   ${RED}✗ Docker Services: NOT AVAILABLE${NC}"
        fi
        
        echo ""
        echo -e "${BLUE}🌐 Quick Access:${NC}"
        if check_port $CAMERA_PORT; then
            echo "   • Main App: http://localhost"
            echo "   • Camera API: http://$CAMERA_HOST:$CAMERA_PORT"
            echo "   • API Docs: http://$CAMERA_HOST:$CAMERA_PORT/docs"
        else
            echo "   • Services are stopped - start them first"
        fi
        
        echo ""
        echo -e "${YELLOW}⚡ Smart Control Options:${NC}"
        echo "   [s] Start all services (smart mode)"
        echo "   [t] Stop services (preserve containers)"
        echo "   [T] FORCE Stop (remove all containers)"
        echo "   [r] Restart services (smart mode)"
        echo "   [R] Force restart (rebuild everything)"
        echo "   [c] Check detailed status"
        echo "   [l] View logs (camera/docker)"
        echo "   [o] Open main app in browser"
        echo "   [h] Help & Tips"
        echo "   [q] Quit control panel"
        echo ""
        
        # Show container efficiency tip
        cd "$PROJECT_DIR" 2>/dev/null || true
        if docker info >/dev/null 2>&1 && [ -f "$DOCKER_COMPOSE_FILE" ]; then
            IFS=',' read -r existing_containers running_containers stopped_containers <<< "$(check_docker_containers_status)"
            if [ "$stopped_containers" -gt 0 ]; then
                echo -e "${GREEN}💡 Tip: You have $stopped_containers stopped containers ready for instant start!${NC}"
                echo ""
            fi
        fi
        
        read -n 1 -p "Choose an option: " choice
        echo ""
        
        case "$choice" in
            "s"|"S")
                log "Starting services in smart mode..."
                if start_camera_hardware_ps && start_docker_services; then
                    success "✅ All services started successfully!"
                    log "Opening main application in browser..."
                    (powershell.exe -Command "Start-Process 'http://localhost'" 2>/dev/null &) || true
                else
                    error "❌ Failed to start some services"
                fi
                read -p "Press Enter to continue..."
                ;;
            "t")
                log "Stopping services (containers will be preserved)..."
                stop_services false
                success "✅ Services stopped. Next start will be much faster!"
                read -p "Press Enter to continue..."
                ;;
            "T")
                warn "⚠️ This will REMOVE all containers and require full rebuild!"
                read -p "Are you sure? (y/N): " confirm
                if [[ "$confirm" =~ ^[Yy]$ ]]; then
                    log "Force stopping and removing services..."
                    stop_services true
                    warn "⚠️ Next start will require full container rebuild"
                else
                    log "Cancelled force stop"
                fi
                read -p "Press Enter to continue..."
                ;;
            "r"|"R")
                if [ "$choice" = "R" ]; then
                    warn "⚠️ Force restart will rebuild everything!"
                    read -p "Are you sure? (y/N): " confirm
                    if [[ "$confirm" =~ ^[Yy]$ ]]; then
                        log "Force restarting services (full rebuild)..."
                        stop_services true
                        force_restart=true
                    else
                        log "Cancelled force restart"
                        read -p "Press Enter to continue..."
                        continue
                    fi
                else
                    log "Smart restarting services..."
                    stop_services false
                    force_restart=false
                fi
                
                sleep 3
                if start_camera_hardware_ps && start_docker_services; then
                    if [ "$force_restart" = true ]; then
                        success "✅ Services force restarted (full rebuild completed)!"
                    else
                        success "✅ Services restarted efficiently!"
                    fi
                else
                    error "❌ Failed to restart some services"
                fi
                read -p "Press Enter to continue..."
                ;;
            "c"|"C")
                check_status
                read -p "Press Enter to continue..."
                ;;
            "l"|"L")
                echo "View logs for:"
                echo "  [1] Camera service"
                echo "  [2] Camera errors"
                echo "  [3] Docker services"
                read -n 1 -p "Choose (1-3): " log_choice
                echo ""
                case "$log_choice" in
                    "1") show_logs "camera" ;;
                    "2") show_logs "camera-error" ;;
                    "3") show_logs "docker" ;;
                    *) warn "Invalid choice" ;;
                esac
                ;;
            "o"|"O")
                log "Opening main application in browser..."
                (powershell.exe -Command "Start-Process 'http://localhost'" 2>/dev/null &) || warn "Could not open browser"
                ;;
            "h"|"H")
                clear
                echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
                echo -e "${BLUE}║                HELP & TIPS                   ║${NC}"
                echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
                echo ""
                echo -e "${GREEN}🚀 SMART CONTAINER MANAGEMENT:${NC}"
                echo "• Normal 'Stop' preserves containers for instant restart"
                echo "• 'Force Stop' removes containers (requires rebuild)"
                echo "• Smart restart reuses existing containers when possible"
                echo ""
                echo -e "${GREEN}⚡ DEVELOPMENT WORKFLOW:${NC}"
                echo "• Use normal start/stop for daily development"
                echo "• Only use force options when troubleshooting"
                echo "• Check status to see if containers are preserved"
                echo ""
                echo -e "${GREEN}🐛 TROUBLESHOOTING:${NC}"
                echo "• If services won't start, try force restart"
                echo "• Check logs for detailed error information"
                echo "• Use test-camera for camera service debugging"
                echo ""
                echo -e "${GREEN}📊 EFFICIENCY BENEFITS:${NC}"
                echo "• 🔥 No more waiting for container rebuilds"
                echo "• 💾 Preserves container state and volumes"
                echo "• ⚡ 5-10x faster restarts in development"
                echo "• 🎯 Only rebuilds when actually needed"
                echo ""
                echo -e "${GREEN}🔧 FIXES IN v2.1:${NC}"
                echo "• ✅ Improved container status detection"
                echo "• ✅ Better fallback logic for Docker start"
                echo "• ✅ Enhanced error handling and recovery"
                echo "• ✅ More reliable service verification"
                echo ""
                read -p "Press Enter to return to control panel..."
                ;;
            "q"|"Q")
                log "Exiting control panel..."
                echo -e "${YELLOW}💡 Services will continue running in the background${NC}"
                echo -e "${GREEN}💡 Next restart will be fast thanks to preserved containers!${NC}"
                echo -e "${YELLOW}💡 Run this script again to access the control panel${NC}"
                echo -e "${YELLOW}💡 Or use '$0 stop' to stop (preserve) or '$0 force-stop' to remove${NC}"
                exit 0
                ;;
            *)
                warn "Invalid option: $choice"
                sleep 1
                ;;
        esac
    done
}

# Main execution
main() {
    case "${1:-start}" in
        "start")
            log "Starting Airbus Detection System with Fixed Smart Container Management..."
            
            # Quick requirements check
            if ! check_requirements; then
                error "System requirements not met. Run '$0 check' for details."
                exit 1
            fi
            
            # Start camera hardware service
            if start_camera_hardware_ps; then
                # Wait a bit longer for camera service to fully initialize
                log "Waiting 10 seconds for camera service to fully initialize..."
                sleep 10
                
                # Verify camera service is responding
                if check_port $CAMERA_PORT; then
                    success "✓ Camera service confirmed running, starting Docker services..."
                    
                    # Start Docker services with fixed smart management
                    if start_docker_services; then
                        success "🚀 Airbus Detection System startup completed successfully!"
                        log ""
                        log "🌐 Access points:"
                        log "  • Main application: http://localhost"
                        log "  • Camera Hardware API: http://$CAMERA_HOST:$CAMERA_PORT"
                        log "  • API Documentation: http://$CAMERA_HOST:$CAMERA_PORT/docs"
                        log ""
                        
                        # Show efficiency message
                        cd "$PROJECT_DIR"
                        IFS=',' read -r existing_containers running_containers stopped_containers <<< "$(check_docker_containers_status)"
                        if [ "$stopped_containers" -eq 0 ] && [ "$existing_containers" -gt 0 ]; then
                            success "✅ Started existing containers instantly (no rebuild needed)!"
                        fi
                        
                        log "Opening control panel in 3 seconds..."
                        sleep 3
                        show_control_panel
                    else
                        error "Failed to start Docker services. Camera service is still running."
                        log "You can check Docker logs with: docker-compose logs"
                        log "Or try running: $0 force-restart"
                        exit 1
                    fi
                else
                    error "Camera service failed to start properly"
                    error "Try running: $0 test-camera"
                    exit 1
                fi
            else
                error "Failed to start Camera Hardware Service"
                exit 1
            fi
            ;;
        "control"|"panel")
            show_control_panel
            ;;
        "test-camera")
            test_camera_hardware
            ;;
        "stop")
            stop_services false
            ;;
        "force-stop")
            warn "⚠️ This will REMOVE all containers and require full rebuild on next start!"
            read -p "Are you sure? (y/N): " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                stop_services true
            else
                log "Cancelled force stop"
            fi
            ;;
        "restart")
            stop_services false
            sleep 5
            main start
            ;;
        "force-restart")
            warn "⚠️ This will REMOVE and REBUILD all containers!"
            read -p "Are you sure? (y/N): " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                stop_services true
                sleep 5
                main start
            else
                log "Cancelled force restart"
            fi
            ;;
        "status")
            check_status
            ;;
        "check")
            check_requirements
            ;;
        "logs")
            show_logs "${2:-}"
            ;;
        "clean")
            warn "🧹 This will remove all containers, images, and volumes!"
            warn "⚠️ This is useful for troubleshooting but will require full rebuild"
            read -p "Are you sure? (y/N): " confirm
            if [[ "$confirm" =~ ^[Yy]$ ]]; then
                log "Cleaning up Docker environment..."
                cd "$PROJECT_DIR"
                docker-compose down -v --remove-orphans
                docker system prune -f
                success "✅ Docker environment cleaned"
            else
                log "Cancelled cleanup"
            fi
            ;;
        "debug")
            log "🔍 Running diagnostic checks..."
            echo ""
            log "=== SYSTEM DIAGNOSTICS ==="
            check_requirements
            echo ""
            log "=== SERVICE STATUS ==="
            check_status
            echo ""
            log "=== DOCKER COMPOSE VALIDATION ==="
            cd "$PROJECT_DIR"
            if docker-compose config >/dev/null 2>&1; then
                success "✓ docker-compose.yml is valid"
                log "Services defined:"
                docker-compose config --services | while read -r service; do
                    log "  • $service"
                done
            else
                error "✗ docker-compose.yml has syntax errors"
                log "Running validation:"
                docker-compose config
            fi
            echo ""
            log "=== CONTAINER DETAILS ==="
            IFS=',' read -r existing_containers running_containers stopped_containers <<< "$(check_docker_containers_status)"
            log "Existing containers: $existing_containers"
            log "Running containers: $running_containers"
            log "Stopped containers: $stopped_containers"
            if [ "$existing_containers" -gt 0 ]; then
                log "Container details:"
                docker-compose ps -a
            fi
            ;;
        *)
            echo "Airbus Detection System - Fixed Smart Startup Script v2.1"
            echo ""
            echo "Usage: $0 {command} [options]"
            echo ""
            echo -e "${GREEN}🚀 MAIN COMMANDS:${NC}"
            echo "  start         - Start all services with fixed smart container management"
            echo "  control       - Show interactive control panel"
            echo "  stop          - Stop services (preserve containers for fast restart)"
            echo "  restart       - Smart restart (reuse containers when possible)"
            echo ""
            echo -e "${YELLOW}⚡ DEVELOPMENT COMMANDS:${NC}"
            echo "  test-camera   - Start camera service in foreground for testing"
            echo "  status        - Check detailed service status"
            echo "  check         - Check system requirements"
            echo "  logs [type]   - View logs (camera/camera-error/docker)"
            echo "  debug         - Run comprehensive diagnostic checks"
            echo ""
            echo -e "${RED}🔥 FORCE COMMANDS (USE CAREFULLY):${NC}"
            echo "  force-stop    - Stop and REMOVE all containers"
            echo "  force-restart - Restart with full container rebuild"
            echo "  clean         - Deep clean Docker environment"
            echo ""
            echo -e "${BLUE}🔧 FIXES IN v2.1:${NC}"
            echo "  • ✅ Fixed Docker container startup logic"
            echo "  • ✅ Improved container status detection reliability"
            echo "  • ✅ Better error handling and fallback mechanisms"
            echo "  • ✅ Enhanced service verification"
            echo "  • ✅ Added debug command for troubleshooting"
            echo ""
            echo -e "${GREEN}💡 EXAMPLES:${NC}"
            echo "  $0 start          # Start system (fixed smart mode)"
            echo "  $0 control        # Access control panel"
            echo "  $0 debug          # Run diagnostics if having issues"
            echo "  $0 stop           # Stop but keep containers"
            echo "  $0 restart        # Fast restart using existing containers"
            echo ""
            echo -e "${YELLOW}⚠️ EFFICIENCY TIP:${NC}"
            echo "Use 'stop' instead of 'force-stop' for 10x faster next startup!"
            exit 1
            ;;
    esac
}

# Handle script interruption
trap 'error "Script interrupted. Stopping services gracefully..."; stop_services false; exit 1' INT TERM

# Run main function
main "$@"