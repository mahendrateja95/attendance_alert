# Docker Deployment Guide - Face Recognition Attendance System

## 🚀 Quick Start

### Prerequisites
- Docker installed (version 20.10+)
- Docker Compose installed (version 2.0+)
- Ports 1111 and 1122 available on your server

### Deployment Steps

#### 1. Build the Docker Image
```bash
./deploy.sh build
```
Or manually:
```bash
docker-compose build --no-cache
```

#### 2. Start the Application
```bash
./deploy.sh start
```
Or manually:
```bash
docker-compose up -d
```

#### 3. Access the Application
Open your browser and navigate to:
```
http://your-server-ip:1111
```

## 📋 Available Commands

### Using the deploy.sh script:

```bash
# Build the Docker image
./deploy.sh build

# Start the application
./deploy.sh start

# Stop the application
./deploy.sh stop

# Restart the application
./deploy.sh restart

# View logs
./deploy.sh logs

# Check application status
./deploy.sh status
```

### Manual Docker Commands:

```bash
# Build image
docker-compose build

# Start container
docker-compose up -d

# Stop container
docker-compose down

# View logs
docker-compose logs -f

# Check status
docker-compose ps

# Restart container
docker-compose restart
```

## 🔧 Configuration

### Ports
- **Port 1111**: Main Flask application (HTTP)
- **Port 1122**: Reserved for future use

### Environment Variables
You can modify these in `docker-compose.yml`:
- `FLASK_ENV`: Set to `production` for production deployment
- `PORT`: Application port (default: 1111)
- `PYTHONUNBUFFERED`: Set to 1 for immediate log output

### Volumes
The following directories are mounted as volumes to persist data:
- `./users`: User face data
- `./attendance_collections`: Attendance records
- `./face_recognition.db`: SQLite database
- `./input_files`: Input files folder
- `./output_files`: Output files folder

## 🛠️ Troubleshooting

### Check if Docker is running:
```bash
docker ps
```

### View container logs:
```bash
docker-compose logs -f attendance-app
```

### Rebuild after code changes:
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Check port availability:
```bash
sudo netstat -tulpn | grep -E '1111|1122'
# or
sudo ss -tulpn | grep -E '1111|1122'
```

### Access container shell:
```bash
docker exec -it attendance_alert_app bash
```

## 🔐 Security Notes

1. The application runs in production mode by default
2. Make sure to configure firewall rules for ports 1111 and 1122
3. Consider using a reverse proxy (nginx) with SSL/TLS for production
4. Back up the database and user folders regularly

## 🔄 Updates

To update the application after code changes:

```bash
./deploy.sh stop
./deploy.sh build
./deploy.sh start
```

## 📊 Health Check

The container includes a health check that runs every 30 seconds. Check container health:
```bash
docker inspect attendance_alert_app | grep -A 5 Health
```

## 🗄️ Data Backup

To backup your data:
```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Backup database and user data
cp face_recognition.db backups/$(date +%Y%m%d)/
cp -r users backups/$(date +%Y%m%d)/
cp -r attendance_collections backups/$(date +%Y%m%d)/
```

## 🌐 Network Configuration

If you need to connect multiple containers:
- The application uses the `attendance-network` bridge network
- Other services can connect to this network using:
  ```yaml
  networks:
    - attendance-network
  ```

