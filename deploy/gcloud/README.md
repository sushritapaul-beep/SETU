# Google Cloud Run deploy configs

Genuinely free within Cloud Run's Always Free grant — verified directly
against [Cloud Run pricing](https://cloud.google.com/run/pricing) and
[Google Cloud's Free Tier docs](https://docs.cloud.google.com/free/docs/free-cloud-features),
not a blog aggregator (an earlier pass at researching Hugging Face Spaces
got burned exactly that way):

- **2 million requests, 360,000 GiB-seconds, 180,000 vCPU-seconds, 1GB
  egress from North America — per month, no end date.**
- The grant is tied to **Tier 1 region pricing** — `us-central1` (used
  below) is confirmed Tier 1.
- **A billing account (card on file) is required** even though usage should
  stay $0 at low traffic: "A Google Cloud billing account is required to
  access the Google Cloud Free Tier" (Google's own wording).

Chosen over Azure Container Apps — same numeric free grant — specifically
for the request-timeout ceiling: Cloud Run defaults to 300s and is
configurable up to 3600s for free, while Azure's free Consumption ingress
is a **fixed 240s with no free way to extend it** (Premium ingress needs a
separately-billed workload profile). Both figures confirmed against each
provider's own current docs.

## 0. One-time setup

```sh
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com secretmanager.googleapis.com

for s in setu-groq-api-key setu-jina-api-key setu-azure-translator-key \
         setu-qdrant-api-key setu-langfuse-secret-key setu-api-key; do
  gcloud secrets create "$s" --replication-policy=automatic
done

printf '%s' "$GROQ_API_KEY"        | gcloud secrets versions add setu-groq-api-key --data-file=-
printf '%s' "$JINA_API_KEY"        | gcloud secrets versions add setu-jina-api-key --data-file=-
printf '%s' "$AZURE_TRANSLATOR_KEY" | gcloud secrets versions add setu-azure-translator-key --data-file=-
printf '%s' "$QDRANT_API_KEY"      | gcloud secrets versions add setu-qdrant-api-key --data-file=-
printf '%s' "$LANGFUSE_SECRET_KEY" | gcloud secrets versions add setu-langfuse-secret-key --data-file=-
printf '%s' "$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" \
  | gcloud secrets versions add setu-api-key --data-file=-
```

`setu-api-key` is shared: it becomes both `setu-api`'s `API_KEY` and
`setu-ui`'s `SETU_API_KEY`, which must match exactly — same reason as
`docs/DEPLOY.md`'s Render setup. Values that aren't real secrets
(`AZURE_TRANSLATOR_REGION`, `QDRANT_URL`, `LANGFUSE_PUBLIC_KEY` — a public
key by name, `ALLOWED_ORIGINS`) go as plain env vars in step 1 instead of
Secret Manager, matching how `render.yaml` only marks genuine credentials
`sync: false` as secrets.

## 1. Deploy `setu-api`

```sh
gcloud run deploy setu-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi --cpu 1 \
  --timeout 300 \
  --port 8000 \
  --max-instances 3 \
  --set-env-vars LLM_PROVIDER=groq,GROQ_MODEL=qwen/qwen3.6-27b,QDRANT_COLLECTION=setu-schemes,LANGFUSE_HOST=https://cloud.langfuse.com,LOG_LEVEL=INFO,AZURE_TRANSLATOR_REGION=<your-region>,QDRANT_URL=<your-qdrant-url>,LANGFUSE_PUBLIC_KEY=<your-langfuse-public-key>,ALLOWED_ORIGINS=http://localhost:8001 \
  --set-secrets GROQ_API_KEY=setu-groq-api-key:latest,JINA_API_KEY=setu-jina-api-key:latest,AZURE_TRANSLATOR_KEY=setu-azure-translator-key:latest,QDRANT_API_KEY=setu-qdrant-api-key:latest,LANGFUSE_SECRET_KEY=setu-langfuse-secret-key:latest,API_KEY=setu-api-key:latest
```

`--source .` builds from the repo root, where the root `Dockerfile`
already lives — no assembly step, same image this whole
deployment-readiness effort already validated with a real `docker build` +
`docker run` (`docs/DEPLOYMENT_HANDOFF.md`). `--memory 2Gi --cpu 1` matches
what that same testing measured `setu-api` actually needs (~1.4GB peak, per
§4) — Cloud Run only bills while an instance is actively serving a request,
so the headroom above the measured floor costs nothing extra at low
traffic.

`ALLOWED_ORIGINS` is a placeholder — `setu-ui`'s real URL isn't known until
step 3, same two-step reason as Render (`docs/DEPLOY.md` §3).

Cloud Run prints the service URL when this finishes:
`https://setu-api-<hash>-uc.a.run.app`.

## 2. Deploy `setu-ui`

`ui/Dockerfile` builds from the repo root (it needs `.chainlit/`,
`public/`, `chainlit.md` from there), but `gcloud run deploy --source`
requires the Dockerfile at the root of whatever directory `--source`
points at, with no override flag — confirmed against the
[gcloud reference](https://docs.cloud.google.com/sdk/gcloud/reference/run/deploy)
and
[Cloud Run's source-deploy docs](https://docs.cloud.google.com/run/docs/deploying-source-code).
Identical constraint to Hugging Face Spaces, so this reuses the same
self-contained Dockerfile at `deploy/ui-standalone/Dockerfile` (see
`deploy/huggingface/README.md` for why that variant exists) instead of a
second copy.

```sh
rm -rf /tmp/setu-ui-src && mkdir -p /tmp/setu-ui-src
cp ui/chainlit_app.py ui/requirements.txt /tmp/setu-ui-src/
cp -r .chainlit public /tmp/setu-ui-src/
cp chainlit.md /tmp/setu-ui-src/
cp deploy/ui-standalone/Dockerfile /tmp/setu-ui-src/

gcloud run deploy setu-ui \
  --source /tmp/setu-ui-src \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi --cpu 1 \
  --timeout 300 \
  --port 8000 \
  --max-instances 3 \
  --set-env-vars SETU_API_URL=<setu-api-url-from-step-1> \
  --set-secrets SETU_API_KEY=setu-api-key:latest
```

`512Mi` because the UI never loads a model — a thin `httpx` client only,
per `config/settings.py`'s own comments on why embeddings/reranking/
translation all moved to API calls. Not empirically re-measured under this
exact memory cap the way `setu-api` was; `docs/DEPLOYMENT_HANDOFF.md` §4
flags the identical gap for Render — worth a real check before trusting it
blind on either platform.

## 3. Tighten CORS

```sh
gcloud run services update setu-api --region us-central1 \
  --update-env-vars ALLOWED_ORIGINS=<setu-ui-url-from-step-2>
```

Same caveat as `docs/DEPLOY.md` §3: not strictly required, since the UI's
calls to the API happen server-side and CORS never governs those — only
worth doing if anything else is ever expected to call the API directly
from a browser.

## 4. Verify

```sh
curl https://setu-api-<hash>-uc.a.run.app/health
```

Open `setu-ui`'s URL and run a real query end to end.

## Known Cloud Run-specific behavior

- **Scales to zero, not to sleep.** No 15-minute idle window like Render,
  and no equivalent to the unconfirmed community report `docs/DEPLOY.md`
  flags about POST requests not waking a sleeping Render free service —
  a Cloud Run cold start is a new container starting, not an existing one
  waking up, so there's no "wrong HTTP method" failure mode to work around.
- **Cold start is fast now.** The pre-warm + baked-model commits already in
  this repo mean a cold container loads both IndicLID models from the
  image's own baked-in cache during startup (~12s, measured via a real
  `--network none` build test) instead of downloading them on the first
  request. That fix was written after `docs/DEPLOY.md` was, so Render's
  guide still describes the slower, pre-fix behavior — worth updating
  there too when this file's contents get folded in.
- **`--timeout 300` is the actual request ceiling** for this deploy,
  configurable up to 3600s for free if a query ever needs longer — no
  separately-billed tier gates that, unlike Azure Container Apps' Premium
  ingress.
- **The filesystem is still ephemeral** — same as Render.
  `data/feedback.sqlite3` and Chainlit's `.files/` don't persist across a
  new revision or a scale-to-zero-then-back-up cycle.
- **Billing account is required**, even though usage should stay within
  the Always Free grant at low traffic — stated as a hard requirement on
  Google's own free-tier docs, not a soft nudge.
