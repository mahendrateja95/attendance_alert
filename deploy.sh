#!/bin/bash

# Deployment script for Face Recognition Attendance System
# Usage: ./deploy.sh [build|start|stop|restart|logs|status]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Build Docker image
build() {
    print_info "Building Docker image..."
    docker-compose build --no-cache
    print_info "Build completed successfully!"
}

# Start the application
start() {
    print_info "Starting attendance alert application..."
    docker-compose up -d
    print_info "Application started successfully!"
    print_info "Access the application at: http://localhost:1111"
}

# Stop the application
stop() {
    print_info "Stopping attendance alert application..."
    docker-compose down
    print_info "Application stopped successfully!"
}

# Restart the application
restart() {
    print_info "Restarting attendance alert application..."
    docker-compose restart
    print_info "Application restarted successfully!"
}

# View logs
logs() {
    print_info "Fetching logs (press Ctrl+C to exit)..."
    docker-compose logs -f
}

# Check status
status() {
    print_info "Checking application status..."
    docker-compose ps
}

# Main script logic
check_docker

case "$1" in
    build)
        build
        ;;
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    logs)
        logs
        ;;
    status)
        status
        ;;
    *)
        echo "Usage: $0 {build|start|stop|restart|logs|status}"
        echo ""
        echo "Commands:"
        echo "  build   - Build the Docker image"
        echo "  start   - Start the application"
        echo "  stop    - Stop the application"
        echo "  restart - Restart the application"
        echo "  logs    - View application logs"
        echo "  status  - Check application status"
        exit 1
        ;;
esac

exit 0

