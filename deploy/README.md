# Deploy

SETU runs on **Oracle Cloud Always Free** — the only free tier priced by
allocation (2 OCPU/12GB) rather than by active or awake compute time, which
is what continuous uptime needs.

See [`oracle/`](oracle/) for the full setup: provisioning the instance,
Cloudflare Tunnel for a public URL, secrets, and the idle-reclamation
mitigation. `docker-compose.yml` at the repo root is the actual deploy
artifact — both existing Dockerfiles, unchanged, plus `cloudflared`.

Component-by-component serving reasoning: [`docs/ARCHITECTURE.md`](../docs/ARCHITECTURE.md) §7.
