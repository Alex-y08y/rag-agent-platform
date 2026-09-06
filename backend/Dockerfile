FROM docker.m.daocloud.io/library/python:3.11-slim

WORKDIR /app

# Install system dependencies (use Tsinghua Debian mirror for reliable China network)
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g; s|security.debian.org|mirrors.tuna.tsinghua.edu.cn|g' \
        /etc/apt/sources.list /etc/apt/sources.list.d/*.sources 2>/dev/null || true; \
    for i in 1 2 3; do apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && break || { echo "apt retry $i"; sleep 5; }; done; \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first (much smaller: ~200MB vs 2GB+ CUDA version)
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (use Tsinghua mirror for China)
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 10 --timeout 120 -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create data directories
RUN mkdir -p /app/data/uploads /app/data/model_cache /app/data/logs /app/data/eval

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
