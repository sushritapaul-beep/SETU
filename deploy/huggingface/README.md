# Hugging Face Spaces deployment configs

Free-tier alternative to Render: HF Spaces CPU Basic gives 16GB RAM / 2 vCPU
per Space, free, no card required — vs. Render free's 512MB / 0.1 vCPU, which
`docs/DEPLOYMENT_HANDOFF.md` §4/§5 measured as too small to run `setu-api`
without OOM-killing on the IndicLID model load. See that doc for the full
comparison; this directory holds the config each Space needs.

## Why two separate directories, not two `render.yaml`-style entries

Render's Blueprint lets one repo define two services with different
`dockerfilePath`/`dockerContext` each. HF Docker Spaces don't have that
option — confirmed against the
[official Spaces config reference](https://huggingface.co/docs/hub/en/spaces-config-reference):
there is no `dockerfile`/context key, only `app_port`. Every Docker Space
builds from a `Dockerfile` at the root of *that Space's own git repo*, using
the whole repo as build context.

`setu-api`'s root `Dockerfile` already satisfies that as-is. `ui/Dockerfile`
doesn't — it deliberately builds from the *repo* root (not `ui/`) to reach
`.chainlit/`, `public/`, and `chainlit.md`, which live there for Render's
sake. Rather than moving those three (which would break the documented local
dev commands in `README.md`/`docs/DEMO.md` and `ui/chainlit_app.py`'s own
docstring, all of which run `chainlit run ui/chainlit_app.py` from the repo
root), `ui/` here holds a self-contained variant that expects them
copied alongside it instead. Nothing in the working tree changes.

## `api/` — push the repo as-is

The existing root `Dockerfile` needs no changes. Only the Space needs a
`README.md` with frontmatter at its repo root, which the canonical
`README.md` shouldn't carry (it's GitHub-facing, not Space config).

```sh
rm -rf /tmp/setu-api-space && cp -r . /tmp/setu-api-space
rm -rf /tmp/setu-api-space/.git /tmp/setu-api-space/venv /tmp/setu-api-space/ui
cp deploy/huggingface/api/README.md /tmp/setu-api-space/README.md
cd /tmp/setu-api-space
git init -b main && git add -A && git commit -m "Deploy setu-api"
git remote add space https://huggingface.co/spaces/<your-username>/setu-api
git push space main
```

Then set `GROQ_API_KEY`, `JINA_API_KEY`, `AZURE_TRANSLATOR_KEY`,
`AZURE_TRANSLATOR_REGION`, `QDRANT_URL`, `QDRANT_API_KEY`, `LANGFUSE_*`,
`API_KEY`, `ALLOWED_ORIGINS` as **Space secrets** (Settings → Variables and
secrets) — same list as `render.yaml`'s `setu-api` env vars.

## `ui/` — assemble from the three Render-root assets plus `ui/`

```sh
rm -rf /tmp/setu-ui-space && mkdir -p /tmp/setu-ui-space
cp ui/chainlit_app.py ui/requirements.txt /tmp/setu-ui-space/
cp -r .chainlit public /tmp/setu-ui-space/
cp chainlit.md /tmp/setu-ui-space/
cp deploy/huggingface/ui/Dockerfile deploy/huggingface/ui/README.md /tmp/setu-ui-space/
cd /tmp/setu-ui-space
git init -b main && git add -A && git commit -m "Deploy setu-ui"
git remote add space https://huggingface.co/spaces/<your-username>/setu-ui
git push space main
```

Then set `SETU_API_URL` (the `setu-api` Space's public URL) and
`SETU_API_KEY` (matching that Space's `API_KEY`) as secrets.

## Port

Both Dockerfiles already default to `${PORT:-8000}`. HF Spaces doesn't
inject a `$PORT` env var the way Render does, so the container falls
through to 8000 — matching `app_port: 8000` declared in both `README.md`
files below. No Dockerfile change needed.

## Not used here: `preload_from_hub`

HF Spaces has a native build-time model preload key
(`preload_from_hub` in the config reference) that overlaps with the
`Dockerfile`'s own `RUN hf_hub_download(...)` bake step (see
`docs/DEPLOYMENT_HANDOFF.md`). Skipped in favor of the Dockerfile step
because it's host-agnostic — it works identically on Render, HF, or any
other Docker host, where `preload_from_hub` only works on HF.
