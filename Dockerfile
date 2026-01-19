# Dockerfile
FROM python:3.12-slim

# Prevents Python from writing .pyc files, and keeps logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional but useful): curl for healthchecks/debug, ca-certificates for TLS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*


# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy only what we need
COPY app /app/app

# Uvicorn listens here
EXPOSE 8000

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
