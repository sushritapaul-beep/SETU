# SETU Chainlit UI image — thin HTTP client to the SETU API, not the agent.
#
# Build from the REPO ROOT (not ui/), since .chainlit/, public/, and
# chainlit.md all live there:
#   docker build -f ui/Dockerfile -t setu-ui .
# On Render: set the Dockerfile path to "ui/Dockerfile" and leave the root
# directory as the repo root.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 setu

WORKDIR /app

COPY ui/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Only what the UI actually needs: the app itself, Chainlit's own config, and
# the branding assets — not agents/, retrieval/, indic/, or eval/.
COPY ui/chainlit_app.py ui/
COPY .chainlit/ .chainlit/
COPY public/ public/
COPY chainlit.md .

RUN chown -R setu:setu /app
USER setu

# Render assigns $PORT at runtime; there is no fixed port to declare here.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/" || exit 1

CMD chainlit run ui/chainlit_app.py --host 0.0.0.0 --port ${PORT:-8000} -h
