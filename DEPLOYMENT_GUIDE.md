# 🚀 Deployment Guide - Attendance Recognition System

## Deploying to Contabo Server with aaPanel using Docker

### Prerequisites
- Contabo VPS server with Ubuntu/CentOS
- aaPanel installed and configured
- Domain name (optional but recommended)
- SSH access to your server

### Step 1: Server Setup

#### 1.1 Connect to your Contabo server via SSH
```bash
ssh root@your-server-ip
```

#### 1.2 Update system packages
```bash
apt update && apt upgrade -y  # For Ubuntu
# OR
yum update -y  # For CentOS
```

#### 1.3 Install Docker if not already installed
```bash
# For Ubuntu
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Start Docker service
systemctl start docker
systemctl enable docker

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
```

### Step 2: aaPanel Configuration

#### 2.1 Login to aaPanel
- Open your browser and go to `http://your-server-ip:8888`
- Login with your aaPanel credentials

#### 2.2 Configure Security Rules
1. Go to **Security** in aaPanel
2. Add the following ports:
   - **Port 8082** (for the attendance app)
   - **Port 22** (SSH)
   - **Port 80** (HTTP)
   - **Port 443** (HTTPS)

#### 2.3 Configure Nginx (Optional - for domain setup)
1. Go to **Website** → **Site**
2. Click **Add Site**
3. Enter your domain name
4. Select **PHP version**: None (since we're using Docker)
5. After creation, edit the site configuration

### Step 3: Upload Application Files

#### 3.1 Create project directory
```bash
mkdir -p /www/wwwroot/attendance-system
cd /www/wwwroot/attendance-system
```

#### 3.2 Upload your project files
You can use one of these methods:

**Method A: Using SCP/SFTP**
```bash
# From your local machine
scp -r C:\Users\MahendraTejaKondapal\Downloads\Attendance_Alert/* root@your-server-ip:/www/wwwroot/attendance-system/
```

**Method B: Using Git (if you have a repository)**
```bash
git clone https://github.com/yourusername/attendance-system.git .
```

**Method C: Using aaPanel File Manager**
1. Go to **Files** in aaPanel
2. Navigate to `/www/wwwroot/attendance-system`
3. Upload your project files

### Step 4: Deploy the Application

#### 4.1 Navigate to project directory
```bash
cd /www/wwwroot/attendance-system
```

#### 4.2 Make deployment script executable
```bash
chmod +x deploy.sh
```

#### 4.3 Run the deployment script
```bash
./deploy.sh
```

#### 4.4 Manual deployment (if script fails)
```bash
# Stop any existing containers
docker-compose down

# Build and start
docker-compose build --no-cache
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f
```

### Step 5: Configure Reverse Proxy (Optional)

If you want to use a domain name and SSL:

#### 5.1 Edit Nginx configuration in aaPanel
1. Go to **Website** → **Site**
2. Click **Settings** for your domain
3. Go to **Config File**
4. Replace the content with:

```nginx
server {
    listen 80;
    server_name your-domain.com www.your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8082;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support (if needed)
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

#### 5.2 Enable SSL (Let's Encrypt)
1. In aaPanel, go to your site settings
2. Click **SSL** tab
3. Select **Let's Encrypt**
4. Apply for SSL certificate

### Step 6: Firewall Configuration

#### 6.1 Configure aaPanel firewall
1. Go to **Security** → **Firewall**
2. Ensure these ports are open:
   - 22 (SSH)
   - 80 (HTTP)
   - 443 (HTTPS)
   - 8082 (Application - only if not using reverse proxy)

#### 6.2 Configure system firewall (if needed)
```bash
# Ubuntu (UFW)
ufw allow 22
ufw allow 80
ufw allow 443
ufw allow 8082
ufw enable

# CentOS (firewalld)
firewall-cmd --permanent --add-port=22/tcp
firewall-cmd --permanent --add-port=80/tcp
firewall-cmd --permanent --add-port=443/tcp
firewall-cmd --permanent --add-port=8082/tcp
firewall-cmd --reload
```

### Step 7: Testing and Monitoring

#### 7.1 Test the application
- Direct access: `http://161.97.155.89:8082`
- Domain access: `http://your-domain.com` (if configured)

#### 7.2 Check application logs
```bash
cd /www/wwwroot/attendance-system
docker-compose logs -f
```

#### 7.3 Monitor resource usage
```bash
# Check Docker containers
docker ps
docker stats

# Check system resources
htop
df -h
```

### Step 8: Application Management

#### 8.1 Common commands
```bash
# View running containers
docker-compose ps

# View logs
docker-compose logs -f

# Restart application
docker-compose restart

# Stop application
docker-compose down

# Update application
git pull  # if using git
docker-compose build --no-cache
docker-compose up -d
```

#### 8.2 Backup important data
```bash
# Backup user data and attendance records
tar -czf backup-$(date +%Y%m%d).tar.gz users/ attendance_collections/
```

### Step 9: Troubleshooting

#### 9.1 Common issues and solutions

**Application not starting:**
```bash
# Check logs
docker-compose logs

# Check if port is available
netstat -tlnp | grep 8082

# Rebuild container
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

**Permission issues:**
```bash
# Fix ownership
chown -R root:root /www/wwwroot/attendance-system
chmod -R 755 /www/wwwroot/attendance-system
```

**Camera not working in Docker:**
```bash
# Add camera access to docker-compose.yml
devices:
  - /dev/video0:/dev/video0
```

### Step 10: Production Optimizations

#### 10.1 Enable container auto-restart
The docker-compose.yml already includes `restart: unless-stopped`

#### 10.2 Set up log rotation
```bash
# Edit docker-compose.yml logging section
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

#### 10.3 Schedule regular backups
```bash
# Add to crontab
crontab -e

# Add this line for daily backup at 2 AM
0 2 * * * cd /www/wwwroot/attendance-system && tar -czf backup-$(date +\%Y\%m\%d).tar.gz users/ attendance_collections/
```

### 🎉 Success!

Your Attendance Recognition System should now be running on your Contabo server!

**Access URLs:**
- Direct: `http://161.97.155.89:8082`
- Domain: `http://your-domain.com` (if configured)

**Key Features Available:**
- ✅ User registration with 300 face images
- ✅ Face recognition with progress tracking
- ✅ Multi-face recognition with name labels
- ✅ Attendance marking system
- ✅ Dropdown user selection for signin
- ✅ Real-time face detection with green/red boxes

### Support

If you encounter any issues, check:
1. Docker container logs: `docker-compose logs`
2. System logs: `journalctl -u docker`
3. aaPanel error logs in the interface
4. Firewall and port configurations
