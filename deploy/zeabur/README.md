# Zeabur deploy configs — the zero-cost recommendation

Chosen over Cloud Run specifically because it's the one option researched
that is **architecturally incapable of charging you**, not merely unlikely
to. Confirmed via [Zeabur's own docs](https://zeabur.com/docs/en-US/pricing/free-plan):
"No credit card required — just sign up and start deploying." With no
payment instrument ever on file, there is nothing for Zeabur to charge —
exceeding the free plan's $5/month usage credit can only throttle or
suspend the service, never generate a bill. Cloud Run's card-on-file model
can't offer that guarantee even with a budget-triggered auto-disable
Cloud Function, since Google's own docs admit a timing gap between usage
and that safeguard firing.

**Trade-off, stated plainly**: Free Plan services auto-sleep after
inactivity (cold start on the next request) and carry no SLA — acceptable
for a personal/demo deployment, not for anything needing guaranteed
uptime. 1 vCPU / 2GB per service, confirmed sufficient for `setu-api`'s
measured ~1.4GB peak (`docs/DEPLOYMENT_HANDOFF.md` §4).

## Port

Zeabur injects `$PORT` (default 8080) into the container, same mechanism
as Render/Cloud Run — confirmed via Zeabur's docs. Both existing
Dockerfiles already read `${PORT:-8000}`, so nothing needs to change here.

## `setu-api` — connect the repo directly

Zeabur auto-detects a root-level `Dockerfile` and builds with it — same as
every other target in this repo. In the Zeabur dashboard: **New Service →
GitHub → select this repo**, root directory left at `.`, no config needed
beyond the environment variables below.

**Environment variables** (same values as `render.yaml`'s `setu-api`):
`LLM_PROVIDER=groq`, `GROQ_MODEL=qwen/qwen3.6-27b`, `QDRANT_COLLECTION=setu-schemes`,
`LANGFUSE_HOST=https://cloud.langfuse.com`, `LOG_LEVEL=INFO`,
`GROQ_API_KEY`, `JINA_API_KEY`, `AZURE_TRANSLATOR_KEY`,
`AZURE_TRANSLATOR_REGION`, `QDRANT_URL`, `QDRANT_API_KEY`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `API_KEY` (generate via
`python -c "import secrets; print(secrets.token_urlsafe(32))"`).
`ALLOWED_ORIGINS` left blank for now — `setu-ui`'s URL isn't known until
the next step, same two-step reason as `docs/DEPLOY.md` §3.

## `setu-ui` — separate repo, same reason as the HF Spaces config

`ui/Dockerfile` builds from the main repo's root to reach `.chainlit/`,
`public/`, and `chainlit.md`. Zeabur's monorepo support
(`ZBPACK_DOCKERFILE_PATH`) lets you point at a different *file*, but
whether it also lets you point the *build context* at a different
directory than the service's configured root isn't confirmed in Zeabur's
own docs — untested here, and not worth risking on an unverified
assumption given everything else in this file is about eliminating risk,
not adding it. Reusing the same proven pattern as
`deploy/huggingface/README.md` instead: a small, separate repo containing
the self-contained `deploy/ui-standalone/Dockerfile` — already built and
run successfully earlier in this deployment effort — at its root.

```sh
rm -rf /tmp/setu-ui-zeabur && mkdir -p /tmp/setu-ui-zeabur
cp ui/chainlit_app.py ui/requirements.txt /tmp/setu-ui-zeabur/
cp -r .chainlit public /tmp/setu-ui-zeabur/
cp chainlit.md /tmp/setu-ui-zeabur/
cp deploy/ui-standalone/Dockerfile /tmp/setu-ui-zeabur/
cd /tmp/setu-ui-zeabur
git init -b main && git add -A && git commit -m "setu-ui for Zeabur"
```

Push this to a new, small GitHub repo (e.g. `setu-ui-deploy`), then in
Zeabur: **New Service → GitHub → select that repo**. Zeabur auto-detects
the root `Dockerfile`.

**Environment variables**: `SETU_API_URL` (the `setu-api` service's public
URL from the step above), `SETU_API_KEY` (must exactly match `setu-api`'s
`API_KEY`).

## Tighten CORS

Once `setu-ui`'s URL is known, set `setu-api`'s `ALLOWED_ORIGINS` to it and
redeploy. Same caveat as `docs/DEPLOY.md` §3 — not strictly required, since
the UI calls the API server-side and CORS never governs that, only worth
doing if a browser is ever expected to call the API directly.

## Verify

```sh
curl https://<setu-api-zeabur-url>/health
```

Open `setu-ui`'s URL and run a real query end to end.
