# Use Python 3.9 slim (Debian trixie)
FROM python:3.9-slim

WORKDIR /app

# System deps for OpenCV + camera
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
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p users attendance_collections static templates

ENV FLASK_APP=app2.py \
    FLASK_ENV=production \
    PYTHONPATH=/app

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

CMD ["python", "app2.py"]
