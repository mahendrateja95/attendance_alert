# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and camera access
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgtk-3-0 \
    libgl1 \
    libglx-mesa0 \
    libgthread-2.0-0 \
    build-essential \
    cmake \
    pkg-config \
    curl \
    v4l-utils \
    libv4l-0 \
    libv4l2rds0 \
    libv4lconvert0 \
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    && rm -rf /var/lib/apt/lists/*


# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p users attendance_collections static templates

# Set environment variables
ENV FLASK_APP=app2.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "app2.py"]
