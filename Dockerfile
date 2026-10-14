# DDoSGuard Production Dockerfile
FROM python:3.11-slim

# System setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application codebase and artifacts
COPY src/ ./src/
COPY api/ ./api/
COPY dashboard/ ./dashboard/
COPY models/ ./models/
COPY artifacts/ ./artifacts/
COPY entrypoint.sh .

# Ensure entrypoint is executable
RUN chmod +x entrypoint.sh

# Expose ports: 8000 (FastAPI), 8501 (Streamlit standard), 7860 (Hugging Face Spaces)
EXPOSE 8000 8501 7860

# Docker Healthcheck on API service
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
CMD ["all"]
