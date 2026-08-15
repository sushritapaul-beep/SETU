# SETU FastAPI service image.
FROM python:3.11-slim

# Fail fast and stream logs unbuffered so `docker logs` is useful in real time.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# curl backs the healthcheck. No build toolchain needed — api/requirements.txt
# has no compiled-from-source wheels since embeddings/reranking/translation
# all moved to Jina/Azure API calls.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user up front — nothing in this image needs root.
RUN useradd --create-home --uid 1000 setu

WORKDIR /app

# Copy requirements alone first so dependency layers stay cached when only
# application code changes. api/requirements.txt, not the root file — that
# one also carries eval/'s dependencies, which never run in this image.
COPY api/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY agents/ agents/
COPY api/ api/
COPY config/ config/
COPY indic/ indic/
COPY observability/ observability/
COPY retrieval/ retrieval/
COPY data/eligibility_rules.json data/eligibility_rules.json

# Own the tree as the runtime user so caches and downloads are writable.
RUN chown -R setu:setu /app /home/setu
USER setu

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
