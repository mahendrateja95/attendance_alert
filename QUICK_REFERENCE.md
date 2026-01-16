# 🚀 NEURA-ID System - Quick Reference

## 🌐 Access URLs

### ⚠️ Domain Disabled
```
https://www.3netra.in    (DISABLED - moved to another server)
https://3netra.in        (DISABLED - moved to another server)
```

**Domain configuration exported to**: `3NETRA_DOMAIN_CONFIG.md`

### Direct IP Access
```
http://161.97.155.89:1111
```

---

## ⚡ Quick Commands

### Check Everything
```bash
# Application status
docker-compose ps

# Domain check
curl -I https://www.3netra.in

# Nginx status
sudo systemctl status nginx

# View logs
docker-compose logs -f
```

### Restart Application
```bash
cd /root/attendance_alert
docker-compose restart
```

### Update Application
```bash
cd /root/attendance_alert
git pull origin main
docker-compose down
docker-compose build
docker-compose up -d
```

### Nginx Management
```bash
# Test configuration
sudo nginx -t

# Reload nginx
sudo systemctl reload nginx

# View logs
sudo tail -f /var/log/nginx/3netra.in.access.log
```

---

## 📊 Current Setup

| Component | Details |
|-----------|---------|
| **Domain** | www.3netra.in (DISABLED) |
| **SSL** | Let's Encrypt (moved) |
| **Application** | NEURA-ID Identification System |
| **Port** | 1111 (Docker) |
| **Database** | neura_id.db |
| **Proxy** | Nginx (domain disabled) |
| **Status** | 🔴 Domain disabled |

---

## 🔧 Important Files

```bash
# Nginx config
/etc/nginx/sites-available/3netra.in

# Backup
/etc/nginx/sites-available/3netra.in.backup

# Application
/root/attendance_alert/

# Logs
/var/log/nginx/3netra.in.access.log
/var/log/nginx/3netra.in.error.log
```

---

## 📞 Troubleshooting

### Domain not accessible?
```bash
sudo systemctl status nginx
docker-compose ps
sudo nginx -t && sudo systemctl reload nginx
```

### Application not responding?
```bash
docker-compose restart
docker-compose logs -f
```

### SSL certificate issues?
```bash
sudo certbot certificates
sudo certbot renew
sudo systemctl reload nginx
```

---

## ✅ Verification Checklist

- [x] Domain resolves correctly
- [x] HTTPS working with valid SSL
- [x] HTTP redirects to HTTPS
- [x] Application accessible
- [x] Docker container healthy
- [x] Nginx configured correctly
- [x] Firewall rules in place
- [x] Logs accessible

---

**⚠️ Domain 3netra.in has been disabled and moved to another server.**

**Domain configuration package**: `3NETRA_DOMAIN_CONFIG.md`  
**Access application directly**: http://161.97.155.89:1111

