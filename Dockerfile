# SETU FastAPI service image.
FROM python:3.11-slim

# Fail fast and stream logs unbuffered so `docker logs` is useful in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep HF model downloads on a mountable path instead of the layer cache,
    # so re-creating the container does not re-download several GB of weights.
    HF_HOME=/home/setu/.cache/huggingface

# Build toolchain is needed by some transitive wheels; curl backs the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user up front — nothing in this image needs root.
RUN useradd --create-home --uid 1000 setu

WORKDIR /app

# Copy requirements alone first so dependency layers stay cached when only
# application code changes.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Own the tree as the runtime user so caches and downloads are writable.
RUN chown -R setu:setu /app /home/setu
USER setu

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# NOTE: api/main.py is created in a later phase; until then this image builds
# but the container will exit on start.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
