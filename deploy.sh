#!/bin/bash

# Attendance Recognition System - Deployment Script
# For Contabo Server with aaPanel and Docker

set -e

echo "🚀 Starting deployment of Attendance Recognition System..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Stop existing containers if running
echo "🔄 Stopping existing containers..."
docker-compose down --remove-orphans || true

# Build and start the application
echo "🏗️ Building Docker image..."
docker-compose build --no-cache

echo "🚀 Starting the application..."
docker-compose up -d

# Wait for the application to start
echo "⏳ Waiting for application to start..."
sleep 10

# Check if the application is running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Application is running successfully!"
    echo "🌐 Application URL: http://your-server-ip:8080"
    echo "📊 To view logs: docker-compose logs -f"
    echo "🔄 To restart: docker-compose restart"
    echo "🛑 To stop: docker-compose down"
else
    echo "❌ Application failed to start. Check logs with: docker-compose logs"
    exit 1
fi
