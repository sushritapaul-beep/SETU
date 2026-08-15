# Deploying SETU to Render

Two Docker web services, both free tier: `setu-api` (FastAPI, `Dockerfile`)
and `setu-ui` (Chainlit, `ui/Dockerfile`). `render.yaml` at the repo root
defines both as a Render Blueprint.

## 1. Apply the blueprint

Render dashboard -> **New** -> **Blueprint** -> point at this repo. Render
reads `render.yaml` and creates both services. You'll be prompted for every
`sync: false` variable — the secrets neither service should have a default
for:

**setu-api** needs: `GROQ_API_KEY`, `JINA_API_KEY`, `AZURE_TRANSLATOR_KEY`,
`AZURE_TRANSLATOR_REGION` (blank if the Translator resource is "Global"),
`QDRANT_URL`, `QDRANT_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`.
All free-tier signups — see `.env.example` for the exact signup links.

Also set `API_KEY` to a generated secret (`python -c "import secrets;
print(secrets.token_urlsafe(32))"`) — the service is about to be public, and
`/query`/`/query/stream` spend metered LLM quota per call.

Leave `ALLOWED_ORIGINS` blank for the first deploy; come back to it in step 3.

**setu-ui** needs `SETU_API_URL`, and **`SETU_API_KEY` set to the exact same
value as `setu-api`'s `API_KEY`** — the UI sends this as `X-API-Key` on every
call. Leave `SETU_API_URL` blank for the first deploy; `setu-api`'s URL isn't
known until it's deployed at least once. The service will start and show a
"cannot reach the SETU API" message until you set it in step 2.

> Earlier drafts of this guide told you to set `API_KEY` without also
> covering `SETU_API_KEY` — following that alone would have had the UI's own
> requests 401 against its own API, since nothing sent the header. Caught
> before deploy by re-reading `ui/chainlit_app.py` against this guide's own
> instructions, not after.

## 2. Wire the UI to the API

Once `setu-api` finishes deploying, copy its URL from the Render dashboard
(`https://setu-api-xxxx.onrender.com` — Render appends a random suffix if the
plain name is taken). Set that as `setu-ui`'s `SETU_API_URL` env var and
redeploy `setu-ui`.

**Why not automatic:** Render blueprint's `fromService` only exposes a
service's *private*-network address (`host`/`port`/`hostport`), and **free
web services can send private-network requests but cannot receive them** —
so private networking between two free services doesn't work here regardless.
The public `RENDER_EXTERNAL_URL` a service gets isn't something blueprint
syntax can inject into a *different* service, so this one hop has to be
manual.

The limitation is specifically on the *receiving* side. If `setu-api` alone
were ever moved to a paid plan (Starter+) while `setu-ui` stayed free,
private networking between them would work — a free service can still send.
Not worth it for a demo (private networking's only benefit here is skipping
a public-internet hop; both services already work fine over public URLs),
but worth knowing if `setu-api` ever needs a reason to go paid anyway.

## 3. (Recommended) Tighten CORS

Set `setu-api`'s `ALLOWED_ORIGINS` to `setu-ui`'s real URL and redeploy.
Not strictly required — the UI's calls to the API happen server-side
(`httpx`, from Chainlit's Python backend), which CORS never governs, only a
browser's own cross-origin JS calls are — but worth doing if anything else
is ever expected to call the API directly from a browser.

## 4. Verify

```bash
curl https://setu-api-xxxx.onrender.com/health
# {"status":"ok","qdrant":true,...}
```

Open `setu-ui`'s URL and run a real query end to end.

## Known Render-specific behavior

- **Both fixed Dockerfiles now bind to `$PORT`, not a hardcoded port.**
  Render's default `PORT` is `10000`, not `8000` — a service that ignores it
  and listens on a fixed port deploys "successfully" and is then completely
  unreachable, since Render routes external traffic to `$PORT` regardless of
  what the container actually listens on. (`ui/Dockerfile` already handled
  this correctly; the root `Dockerfile` didn't — fixed to match.)
- **Free services spin down after 15 minutes idle**, and take roughly
  30–60s to spin back up on the next request. Since `setu-api` is only ever
  hit indirectly (via `setu-ui` forwarding a query), whenever `setu-ui` has
  been idle long enough to sleep, `setu-api` has almost certainly been
  idle at least as long too — this isn't a rare edge case, it's the expected
  shape of every "first visitor in a while" request: two cold starts back
  to back before the first answer streams. The UI's own request timeout
  (300s) comfortably survives this; it's a real, visible wait for whoever's
  watching, not a failure.
- **A cold `setu-api` adds its own latency on top of Render's spin-up.**
  `indic/lang_detect.py` downloads IndicLID's fastText models from
  HuggingFace Hub lazily, on the first query that needs them — not at
  container startup — and the download cache lives on the same ephemeral
  filesystem (below), so it's gone after every spin-down. The very first
  query after a cold start pays for that download again, on top of
  whatever Render's own spin-up took.
- **Unconfirmed, but mitigated regardless: some Render users have reported
  a sleeping free service not waking on a POST request, only on GET or a
  new WebSocket connection** ([community thread](https://community.render.com/t/web-service-not-waking-up-on-post-request-only-get-works/38159),
  not Render's own docs — treated as a real risk, not a settled fact). Since
  `/query/stream` is a POST, `ui/chainlit_app.py` fires a best-effort `GET
  /health` immediately before it on every query, so the wake trigger — if
  `setu-api` is asleep — is always the method Render's docs do confirm
  works. Free (a few hundred ms) when the API is already warm.
- **The per-client-IP rate limiter would have rate-limited the whole
  deployment, not each visitor.** `setu-ui` calls `setu-api` server-side —
  every visitor's query reaches the API from the same connection, so
  `request.client.host`-based limiting couldn't tell visitors apart, and the
  documented 20-requests/60s ceiling would have applied to *all* of them
  combined. Fixed: `setu-ui` now sends `X-Client-Id` (its own per-browser
  `session_id`), and `enforce_rate_limit` prefers it over the source IP when
  present.
- **The filesystem is ephemeral** on the free tier (no persistent disk
  attached). `data/feedback.sqlite3` (Chainlit's thumbs-up/down store) and
  `.files/` (Chainlit's own local storage for message elements like the
  Sources card) are both recreated empty on every restart, redeploy, or
  spin-down/spin-up cycle. Feedback submitted between restarts is lost —
  fine for a demo, not for anything that needs to retain it; a paid plan
  with a persistent disk mounted at `data/` would fix this if it matters
  later.
- **750 free instance-hours/month, shared per Render workspace** across
  every free service in it. Two services spinning down after 15 minutes of
  low-traffic use stay well under that; it would only bind if something
  else in the same workspace runs near-continuously too.
