# 🚀 Face Recognition Attendance System - Server Access Information

## Server Details
- **Server IP**: `161.97.155.89`
- **Application Port**: `1111`
- **Reserved Port**: `1122`

## Access URLs

### Main Application
```
http://161.97.155.89:1111
```

### Alternative Access Methods
- From the server itself: `http://localhost:1111`
- From local network: `http://161.97.155.89:1111`
- From anywhere: `http://161.97.155.89:1111` (if firewall allows)

## Quick Status Check

Check if the application is accessible:
```bash
curl http://161.97.155.89:1111
```

Or from another machine:
```bash
curl http://161.97.155.89:1111
```

## Container Status

View container status:
```bash
docker-compose ps
```

View real-time logs:
```bash
docker-compose logs -f attendance-app
```

## Port Information

| Port | Status | Purpose |
|------|--------|---------|
| 1111 | ✅ Active | Main Flask Application |
| 1122 | ✅ Reserved | Available for future use |

## Firewall Configuration

If you cannot access the application from external networks, ensure your firewall allows traffic on ports 1111 and 1122:

### Check Firewall Status
```bash
# For UFW (Ubuntu Firewall)
sudo ufw status

# For iptables
sudo iptables -L -n | grep -E '1111|1122'
```

### Allow Ports (if needed)
```bash
# Using UFW
sudo ufw allow 1111/tcp
sudo ufw allow 1122/tcp

# Using iptables
sudo iptables -A INPUT -p tcp --dport 1111 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 1122 -j ACCEPT
sudo iptables-save
```

### For Cloud Servers
If you're using a cloud provider (AWS, Azure, GCP, DigitalOcean, etc.), also configure:
- **Security Groups** (AWS)
- **Network Security Groups** (Azure)
- **Firewall Rules** (GCP)
- **Networking** (DigitalOcean)

Allow inbound traffic on ports:
- TCP 1111
- TCP 1122

## Testing Accessibility

### From Server (Local Test)
```bash
curl http://localhost:1111
curl http://127.0.0.1:1111
```

### From Remote (External Test)
```bash
curl http://161.97.155.89:1111
```

### Via Web Browser
Open your browser and navigate to:
```
http://161.97.155.89:1111
```

## Container Management

### Start Application
```bash
cd /root/attendance_alert
docker-compose up -d
```

### Stop Application
```bash
docker-compose stop
```

### Restart Application
```bash
docker-compose restart
```

### View Logs
```bash
docker-compose logs -f
```

### Check Health
```bash
docker inspect attendance_alert_app | grep -A 5 Health
```

## Network Binding

The application is configured to bind to:
- `0.0.0.0:1111` - Listens on all network interfaces
- `0.0.0.0:1122` - Reserved port on all interfaces

This means it's accessible via:
- ✅ localhost (127.0.0.1)
- ✅ Server IP (161.97.155.89)
- ✅ Any other network interface on the server

## Production Recommendations

### 1. Use Nginx as Reverse Proxy
Consider setting up Nginx for:
- SSL/TLS support (HTTPS)
- Load balancing
- Better performance
- Security headers

### 2. SSL Certificate
Get a free SSL certificate from Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### 3. Domain Name
Point a domain name to your server IP:
```
A Record: yourdomain.com → 161.97.155.89
```

### 4. Monitoring
Set up monitoring to ensure the application stays healthy:
```bash
# Simple health check script
watch -n 30 'curl -s http://161.97.155.89:1111/health || echo "DOWN"'
```

## Troubleshooting

### Cannot Access Externally?

1. **Check if Docker is listening on the correct interface:**
   ```bash
   netstat -tlnp | grep -E '1111|1122'
   ```

2. **Check firewall:**
   ```bash
   sudo iptables -L -n | grep -E '1111|1122'
   ```

3. **Test from server:**
   ```bash
   curl -v http://localhost:1111
   ```

4. **Test from external network:**
   ```bash
   telnet 161.97.155.89 1111
   ```

### Connection Refused?

- Ensure the container is running:
  ```bash
  docker-compose ps
  ```

- Check container logs:
  ```bash
  docker-compose logs --tail=100 attendance-app
  ```

### Port Already in Use?

Check what's using the ports:
```bash
sudo lsof -i :1111
sudo lsof -i :1122
```

## Support

For issues or questions:
1. Check the logs: `docker-compose logs -f`
2. Verify container status: `docker-compose ps`
3. Check firewall rules
4. Verify network connectivity

---

**Last Updated**: $(date)
**Server IP**: 161.97.155.89
**Status**: 🟢 Running and Healthy

