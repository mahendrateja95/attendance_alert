#!/bin/bash

# 🚀 Attendance Recognition System - WebRTC Deployment Script
# For Contabo/VPS deployment with simplified Docker setup

set -e

echo "🎯 Attendance Recognition System - WebRTC Deployment"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    mkdir -p users attendance_collections static
    chmod 755 users attendance_collections static
    print_success "Directories created"
}

# Stop existing containers
stop_existing() {
    print_info "Stopping any existing containers..."
    docker-compose down 2>/dev/null || true
    print_success "Stopped existing containers"
}

# Build and start the application
deploy_app() {
    print_info "Building and starting the WebRTC attendance system..."
    
    # Build the Docker image
    docker-compose build --no-cache
    
    # Start the application
    docker-compose up -d
    
    print_success "Application deployed successfully!"
}

# Check application health
check_health() {
    print_info "Checking application health..."
    sleep 10
    
    # Check if container is running
    if docker-compose ps | grep -q "Up"; then
        print_success "Container is running"
    else
        print_error "Container failed to start"
        docker-compose logs
        exit 1
    fi
    
    # Check if application responds
    if curl -f http://localhost:8082 >/dev/null 2>&1; then
        print_success "Application is responding"
    else
        print_warning "Application might be starting up. Check logs if issues persist."
    fi
}

# Display deployment information
show_info() {
    echo ""
    echo "🎉 Deployment Complete!"
    echo "======================"
    echo ""
    print_info "Application URL: http://your-server-ip:8082"
    print_warning "For production: Configure HTTPS (required for camera access)"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Set up HTTPS/SSL certificate"
    echo "2. Configure reverse proxy (nginx/apache)"
    echo "3. Test camera functionality in browser"
    echo "4. Register your first user"
    echo ""
    echo "📊 Useful Commands:"
    echo "   View logs:     docker-compose logs -f"
    echo "   Restart app:   docker-compose restart"
    echo "   Stop app:      docker-compose down"
    echo "   Update app:    git pull && docker-compose up -d --build"
    echo ""
    echo "🔧 Debug Issues:"
    echo "   Run debug script: python debug_recognition.py"
    echo ""
}

# Main deployment process
main() {
    echo "Starting deployment process..."
    echo ""
    
    check_docker
    create_directories
    stop_existing
    deploy_app
    check_health
    show_info
}

# Handle script arguments
case "${1:-deploy}" in
    "deploy")
        main
        ;;
    "stop")
        print_info "Stopping the application..."
        docker-compose down
        print_success "Application stopped"
        ;;
    "restart")
        print_info "Restarting the application..."
        docker-compose restart
        print_success "Application restarted"
        ;;
    "logs")
        print_info "Showing application logs..."
        docker-compose logs -f
        ;;
    "update")
        print_info "Updating the application..."
        git pull
        docker-compose down
        docker-compose up -d --build
        print_success "Application updated"
        ;;
    "debug")
        print_info "Running debug checks..."
        python debug_recognition.py
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|update|debug}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy the application (default)"
        echo "  stop    - Stop the application"
        echo "  restart - Restart the application"
        echo "  logs    - Show application logs"
        echo "  update  - Update from git and redeploy"
        echo "  debug   - Run debug checks"
        exit 1
        ;;
esac