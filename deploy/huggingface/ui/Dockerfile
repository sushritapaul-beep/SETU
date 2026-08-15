# SETU Chainlit UI image — Hugging Face Space variant.
#
# HF Docker Spaces always build from a Dockerfile at the repo root, using
# the whole repo as context, with no dockerContext/dockerfilePath override
# (unlike Render's Blueprint). The canonical ui/Dockerfile builds from the
# main repo's root instead, to reach .chainlit/, public/, and chainlit.md,
# which live there for Render's sake. This variant expects those three
# copied alongside it instead — see deploy/huggingface/README.md for the
# exact assembly commands. Everything else is identical to ui/Dockerfile.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 setu

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY chainlit_app.py .
COPY .chainlit/ .chainlit/
COPY public/ public/
COPY chainlit.md .

RUN chown -R setu:setu /app
USER setu

# HF Spaces doesn't inject $PORT; app_port: 8000 in README.md routes to this
# Dockerfile's own default instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/" || exit 1

CMD chainlit run chainlit_app.py --host 0.0.0.0 --port ${PORT:-8000} -h
