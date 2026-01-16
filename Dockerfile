# Face Recognition Attendance System - Docker Image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY static/ ./static/
COPY templates/ ./templates/

# Create necessary directories
RUN mkdir -p users attendance_collections input_files output_files

# Create empty database file (will be mounted as volume)
RUN touch neura_id.db

# Expose ports
EXPOSE 1111
EXPOSE 1122

# Set environment variables
ENV FLASK_ENV=production
ENV PORT=1111
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:1111/health')" || exit 1

# Run the application
CMD ["python", "app.py"]

