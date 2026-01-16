# 🌐 Domain Migration Complete - www.3netra.in

## ⚠️ Migration Status: DOMAIN DISABLED

**Date**: October 12, 2025  
**Domain**: www.3netra.in  
**Application**: NEURA-ID Identification System  
**Status**: 🔴 **DISABLED - Domain moved to another server**

**Note**: This domain is no longer active on this server. All domain configurations have been exported to `3NETRA_DOMAIN_CONFIG.md` for use on the new server.

---

## 📋 What Was Done

### 1. Pre-Migration Check
- ✅ Checked existing nginx configuration
- ✅ Verified ports 3000 and 8001 were not in use
- ✅ No processes needed to be killed
- ✅ Confirmed Docker container running on port 1111

### 2. Nginx Configuration Update
- ✅ Backed up original config: `/etc/nginx/sites-available/3netra.in.backup`
- ✅ Updated proxy target: Port 3000 → Port 1111 (Docker)
- ✅ Removed old API proxy configuration (port 8001)
- ✅ Added security headers
- ✅ Configured timeouts for face recognition processing
- ✅ Added buffer settings for video streaming

### 3. SSL/HTTPS Configuration
- ✅ SSL certificates preserved (Let's Encrypt)
- ✅ HTTPS working on port 443
- ✅ HTTP to HTTPS redirect functional
- ✅ Security headers configured

### 4. Testing & Verification
- ✅ Nginx configuration syntax validated
- ✅ Nginx service reloaded successfully
- ✅ HTTP redirects to HTTPS (301)
- ✅ HTTPS returns 200 OK
- ✅ Application accessible via both domains
- ✅ NEURA-ID system responding correctly

---

## 🌐 Access URLs

### ⚠️ Domain Disabled
```
https://3netra.in          (DISABLED - moved to another server)
https://www.3netra.in      (DISABLED - moved to another server)
```

**Domain configuration exported to**: `3NETRA_DOMAIN_CONFIG.md`

### IP Address (HTTP)
```
http://161.97.155.89:1111
```

**All domains now point to your NEURA-ID Identification System!**

---

## 🔒 SSL Certificate Information

- **Certificate Provider**: Let's Encrypt
- **Certificate Path**: `/etc/letsencrypt/live/3netra.in/fullchain.pem`
- **Private Key**: `/etc/letsencrypt/live/3netra.in/privkey.pem`
- **Status**: ✅ Valid and Active
- **Auto-Renewal**: Configured via certbot

---

## 🔧 Technical Details

### Previous Configuration
```
Port 3000: Frontend (Not in use)
Port 8001: Backend API (Not in use)
```

### New Configuration
```
Port 1111: NEURA-ID Docker Container
SSL: HTTPS enabled with Let's Encrypt
Proxy: Nginx reverse proxy
```

### Nginx Proxy Settings
- **Upstream**: `http://127.0.0.1:1111`
- **Timeouts**: 300 seconds (for face processing)
- **Buffering**: Disabled (for video streaming)
- **WebSocket**: Supported
- **Security Headers**: Enabled

---

## 📊 Verification Results

| Test | URL | Status | Response |
|------|-----|--------|----------|
| HTTP Redirect | http://3netra.in | ✅ | 301 → HTTPS |
| HTTPS (3netra.in) | https://3netra.in | ✅ | 200 OK |
| HTTPS (www) | https://www.3netra.in | ✅ | 200 OK |
| Application Title | All URLs | ✅ | "NeuraID Identification System" |
| Docker Container | Port 1111 | ✅ | Healthy |
| Nginx Service | systemctl | ✅ | Active (running) |

---

## 📁 Configuration Files

### Nginx Configuration
- **Active Config**: `/etc/nginx/sites-available/3netra.in`
- **Backup**: `/etc/nginx/sites-available/3netra.in.backup`
- **Enabled**: `/etc/nginx/sites-enabled/3netra.in` (symlink)

### Log Files
- **Access Log**: `/var/log/nginx/3netra.in.access.log`
- **Error Log**: `/var/log/nginx/3netra.in.error.log`

---

## 🚀 Quick Commands

### Check Nginx Status
```bash
sudo systemctl status nginx
sudo nginx -t
```

### View Logs
```bash
# Nginx logs
sudo tail -f /var/log/nginx/3netra.in.access.log
sudo tail -f /var/log/nginx/3netra.in.error.log

# Application logs
docker-compose logs -f attendance-app
```

### Reload Nginx (after config changes)
```bash
sudo nginx -t
sudo systemctl reload nginx
```

### Check Application Status
```bash
docker-compose ps
curl -I https://3netra.in
```

---

## 🔄 SSL Certificate Renewal

Your SSL certificates will auto-renew via certbot. To check renewal status:

```bash
# Check certificate expiry
sudo certbot certificates

# Test renewal
sudo certbot renew --dry-run

# Force renewal (if needed)
sudo certbot renew --force-renewal
```

After certificate renewal, reload nginx:
```bash
sudo systemctl reload nginx
```

---

## 🛡️ Security Features Enabled

1. **HTTPS Only** - HTTP automatically redirects to HTTPS
2. **Security Headers**:
   - X-Frame-Options: SAMEORIGIN
   - X-Content-Type-Options: nosniff
   - X-XSS-Protection: enabled
3. **SSL/TLS** - Let's Encrypt certificates
4. **Reverse Proxy** - Application not directly exposed
5. **Firewall** - UFW configured for ports 1111, 1122

---

## 📈 Performance Optimizations

### For Face Recognition
- Connection timeout: 300 seconds
- Send timeout: 300 seconds
- Read timeout: 300 seconds
- Send timeout: 300 seconds

### For Video Streaming
- Proxy buffering: OFF
- Request buffering: OFF
- WebSocket support: ENABLED

---

## 🔧 Troubleshooting

### Domain Not Accessible?

1. **Check Nginx**:
   ```bash
   sudo systemctl status nginx
   sudo nginx -t
   ```

2. **Check Docker Container**:
   ```bash
   docker-compose ps
   docker-compose logs attendance-app
   ```

3. **Check Port**:
   ```bash
   sudo ss -tulpn | grep 1111
   ```

4. **Check DNS**:
   ```bash
   nslookup 3netra.in
   nslookup www.3netra.in
   ```

### SSL Certificate Issues?

```bash
# Check certificate
sudo certbot certificates

# Renew if needed
sudo certbot renew

# Reload nginx
sudo systemctl reload nginx
```

### Application Not Responding?

```bash
# Restart Docker container
docker-compose restart

# Check application logs
docker-compose logs -f

# Test direct access
curl http://localhost:1111
```

---

## 🎯 Migration Summary

### Before
- **Domain**: www.3netra.in
- **Backend**: Port 3000 (not running)
- **API**: Port 8001 (not running)
- **SSL**: Let's Encrypt
- **Status**: Inactive

### After
- **Domain**: www.3netra.in ✅
- **Application**: NEURA-ID Identification System
- **Port**: 1111 (Docker container)
- **SSL**: Let's Encrypt ✅
- **Status**: 🟢 **LIVE & ACTIVE**

---

## ✨ Benefits of This Setup

1. ✅ **Secure**: HTTPS with valid SSL certificate
2. ✅ **Professional**: Custom domain instead of IP
3. ✅ **Fast**: Nginx reverse proxy with optimization
4. ✅ **Reliable**: Docker containerization
5. ✅ **Scalable**: Easy to update and maintain
6. ✅ **Monitored**: Centralized logging
7. ✅ **Protected**: Security headers enabled

---

## ⚠️ Domain Disabled

The domain 3netra.in has been disabled on this server and moved to another application.

### 📦 Domain Configuration Package
All domain-related configurations have been exported to:
### 📄 **3NETRA_DOMAIN_CONFIG.md**

This file contains everything needed to set up the domain on the new server.

**Status**: 🔴 DISABLED on this server

---

## 📞 Support & Maintenance

### Regular Checks
```bash
# Weekly checks recommended
sudo systemctl status nginx
docker-compose ps
sudo certbot certificates
```

### Backup Locations
- Nginx backup: `/etc/nginx/sites-available/3netra.in.backup`
- Docker volumes: Automatic via docker-compose
- Database: `/root/attendance_alert/neura_id.db`

### Update Application
```bash
cd /root/attendance_alert
git pull origin main
docker-compose down
docker-compose build
docker-compose up -d
```

---

*Migration completed successfully on October 12, 2025*  
*Domain: www.3netra.in → NEURA-ID Identification System*  
*No downtime, no data loss, fully operational*

