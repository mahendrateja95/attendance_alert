# Use Python 3.9 slim (Debian trixie)
FROM python:3.9-slim

WORKDIR /app

# System deps for OpenCV (camera-specific packages removed for WebRTC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1 \
    libglx-mesa0 \
    libgthread-2.0-0 \
    build-essential \
    cmake \
    pkg-config \
    curl \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p users attendance_collections static templates

# Environment variables
ENV FLASK_APP=app2.py \
    FLASK_ENV=production \
    PYTHONPATH=/app \
    PORT=5000

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "app2.py"]