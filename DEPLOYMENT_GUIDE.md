# 🚀 Deployment Guide - Attendance Recognition System (WebRTC Version)

## Deploying to Contabo Server with aaPanel using Docker

### 🌟 What's New - WebRTC Implementation
- ✅ **No camera device mapping needed** (`--device /dev/video0`)
- ✅ **No privileged mode required** (`--privileged`)
- ✅ **Camera runs in user's browser** via WebRTC
- ✅ **Works on ANY VPS/cloud server**
- ✅ **HTTPS required for production** (camera access)

### Prerequisites
- Contabo VPS server with Ubuntu/CentOS
- aaPanel installed and configured
- Domain name with SSL certificate (required for WebRTC camera access)
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
   - **Port 443** (HTTPS - required for WebRTC)

#### 2.3 SSL Certificate Setup (REQUIRED for WebRTC)
1. Go to **Website** → **SSL**
2. Add your domain and configure SSL certificate
3. Enable "Force HTTPS" for your domain

### Step 3: Deploy the Application

#### 3.1 Clone the repository
```bash
cd /opt
git clone https://github.com/mahendrateja95/attendance_alert.git
cd attendance_alert
```

#### 3.2 Build and run with Docker Compose
```bash
# Build and start the application (simplified - no camera mapping!)
docker-compose up -d --build

# Check if it's running
docker-compose ps
docker-compose logs -f
```

#### 3.3 Verify deployment
```bash
# Check if the container is running
docker ps

# Test the application
curl http://localhost:8082
```

### Step 4: Nginx Reverse Proxy Setup (aaPanel)

#### 4.1 Create a new website in aaPanel
1. Go to **Website** → **Add Site**
2. Domain: `your-domain.com`
3. Create the site

#### 4.2 Configure reverse proxy
1. Click on your domain → **Reverse Proxy**
2. Add proxy with these settings:
   - **Target URL**: `http://127.0.0.1:8082`
   - **Domain**: `your-domain.com`
   - **Send Host**: Yes

#### 4.3 SSL Configuration for WebRTC
1. Go to **SSL** for your domain
2. Enable SSL certificate (Let's Encrypt or upload your own)
3. Enable "Force HTTPS"
4. **This is REQUIRED for camera access in browsers**

### Step 5: Application Configuration

#### 5.1 Environment Variables
Create a `.env` file if needed:
```bash
echo "FLASK_ENV=production" > .env
echo "PORT=5000" >> .env
```

#### 5.2 Set proper permissions
```bash
chown -R 1000:1000 users attendance_collections
chmod -R 755 users attendance_collections
```

### Step 6: Testing the WebRTC Implementation

#### 6.1 Access the application
- Open: `https://your-domain.com` (HTTPS required!)
- You should see the attendance system homepage

#### 6.2 Test camera functionality
1. Click "Sign Up" to register a new user
2. Enter a name and proceed to camera capture
3. Browser will ask for camera permission - click "Allow"
4. You should see your camera feed and face detection boxes
5. Images will be captured (100 images, ~20 seconds)

#### 6.3 Test recognition
1. After registration, go to "Sign In"
2. Select your name from dropdown
3. Camera should recognize you with green box around face

### Step 7: Monitoring and Maintenance

#### 7.1 Check logs
```bash
# View application logs
docker-compose logs -f

# Check container status
docker-compose ps
```

#### 7.2 Restart application
```bash
# Restart the application
docker-compose restart

# Rebuild if code changed
docker-compose down
docker-compose up -d --build
```

#### 7.3 Backup data
```bash
# Backup user data and attendance records
tar -czf attendance_backup_$(date +%Y%m%d).tar.gz users attendance_collections
```

### Troubleshooting

#### Camera Issues
- **"Camera not working"**: Ensure HTTPS is enabled (required for WebRTC)
- **"Permission denied"**: User needs to click "Allow" in browser
- **"Camera not found"**: Check if running on HTTPS and browser supports WebRTC

#### Application Issues
```bash
# Check if application is responding
curl https://your-domain.com

# Check Docker logs
docker-compose logs attendance-app

# Restart if needed
docker-compose restart
```

#### SSL Issues
- Ensure SSL certificate is properly installed
- Force HTTPS redirect is enabled
- Check that port 443 is open in firewall

### Performance Optimization

#### 7.1 Resource allocation
```bash
# Check resource usage
docker stats

# Adjust resources if needed in docker-compose.yml
```

#### 7.2 Image optimization
- The new system captures only 100 images (vs 300 previously)
- Faster capture rate for better user experience
- Optimized face recognition for VPS deployment

### Security Considerations

1. **HTTPS**: Mandatory for WebRTC camera access
2. **Firewall**: Only necessary ports should be open
3. **Regular updates**: Keep Docker images and system updated
4. **Backup**: Regular backups of user data and attendance records

### Support

If you encounter issues:
1. Check the logs: `docker-compose logs -f`
2. Verify HTTPS is working
3. Ensure camera permissions are granted in browser
4. Test on different browsers (Chrome, Firefox, Safari, Edge)

---

## 🎉 Success Indicators

- ✅ Application accessible via HTTPS
- ✅ Camera permission prompt appears in browser
- ✅ Face detection boxes appear during capture
- ✅ User registration completes in ~20 seconds
- ✅ Face recognition works during sign-in
- ✅ Attendance marking functions properly

Your WebRTC-based attendance system is now deployed and working on your Contabo server!