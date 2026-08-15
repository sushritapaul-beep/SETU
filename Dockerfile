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
# retrieval/vectorstore.py imports the Chunk dataclass from here. Only this
# one file, not the whole package — ingestion/scraper.py pulls in crawl4ai
# and Playwright, which this image has no other reason to install.
COPY ingestion/__init__.py ingestion/chunker.py ingestion/

# Own the tree as the runtime user so caches and downloads are writable.
RUN chown -R setu:setu /app /home/setu
USER setu

# Bake IndicLID's two fastText classifiers into the image, into the same
# cache path hf_hub_download() reads at runtime (~/.cache/huggingface/hub
# under this user). Otherwise the first non-English query downloads them
# over the network inline — real, measured latency on a cold container.
RUN python -c "from huggingface_hub import hf_hub_download as d; \
    d('ai4bharat/IndicLID-FTN', 'model_baseline_roman.bin'); \
    d('ai4bharat/IndicLID-FTR', 'model_baseline_roman.bin')"

# Both models are now guaranteed cached, so skip the network round-trip
# hf_hub_download() would otherwise make to check for a newer revision before
# falling back to the local file — pure latency, same bytes either way.
ENV HF_HUB_OFFLINE=1

EXPOSE 8000

# Render assigns $PORT at runtime (default 10000, not 8000) and routes
# traffic to it regardless of EXPOSE, so the process must bind to whatever
# it's set to. Shell-form CMD/HEALTHCHECK so that expansion actually happens
# — the previous exec-form CMD hardcoded 8000 and would have been
# unreachable on Render, which routes to $PORT, not the container's default.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
