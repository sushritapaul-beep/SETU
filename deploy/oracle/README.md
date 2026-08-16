# Oracle Cloud Always Free deploy

Oracle's Always Free A1 shape is priced in OCPU-hours/GB-hours regardless of
whether the instance is idle or serving a request, so staying on
continuously is the *expected* usage pattern within the free grant, not an
edge case that burns through a credit — most other free-tier hosts meter
*active or awake compute time* instead, where 720 hours/month of continuous
wakefulness blows through the free allowance. That makes this the target for
a service that needs to be reachable at any time without a cold-start delay.

**docker-compose.yml** (repo root) is the actual deploy artifact for this
target — both existing Dockerfiles, unchanged, plus `cloudflared` for the
public URL. Nothing below modifies application code.

## 1. Provision the instance

Console → **Compute → Instances → Create instance**.

- **Shape**: `VM.Standard.A1.Flex`, set to **2 OCPU / 12 GB** — the full
  Always Free allocation for this shape (confirmed against
  [Oracle's own Always Free Resources docs](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm):
  "1,500 OCPU hours and 9,000 GB hours per month... equivalent to 2 OCPUs
  and 12 GB of memory"). Don't split this into two smaller instances unless
  you specifically want to isolate `api` and `ui` onto separate hosts —
  running both on one 12GB instance is simpler, and `setu-api`'s measured
  ~1.4GB peak RAM leaves ample room.
- **Image**: Ubuntu (latest LTS), **ARM64/Ampere** build — not the x86 image.
- **Networking**: default VCN, public IP assigned automatically (Always Free
  includes one public IP per instance).
- **If instance creation fails with an "out of host capacity" error**, this
  is a known, documented Oracle behavior for Always Free A1 shapes in
  high-demand regions — retry in a different availability domain, or wait
  and retry, per Oracle's own troubleshooting guidance for this exact error.
  Not a sign anything is misconfigured.

## 2. Install Docker

SSH in, then run Docker's own current official install (verified against
[docs.docker.com/engine/install/ubuntu](https://docs.docker.com/engine/install/ubuntu/)
directly — explicitly confirmed to support `arm64` architecture):

```sh
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
# log out and back in for the group change to apply
```

`docker-compose-plugin` gives the `docker compose` (v2) command this repo's
`docker-compose.yml` and this guide both use — not the deprecated standalone
`docker-compose` v1 binary.

## 3. Get the code and configure secrets

```sh
git clone https://github.com/sushritapaul-beep/SETU.git
cd SETU
cp .env.example .env   # if present; otherwise create .env directly
```

Set in `.env`: `GROQ_API_KEY`, `JINA_API_KEY`, `AZURE_TRANSLATOR_KEY`,
`AZURE_TRANSLATOR_REGION`, `QDRANT_URL`, `QDRANT_API_KEY`, `API_KEY`
(generate via `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`),
and `CLOUDFLARE_TUNNEL_TOKEN` (from step 4). `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` are optional — the app runs untraced without them.

## 4. Set up Cloudflare Tunnel

This gets a public HTTPS URL **without opening any firewall port or
configuring TLS certificates** — `cloudflared` makes an outbound connection
to Cloudflare's edge; nothing needs to accept inbound traffic on the
instance at all.

1. [Cloudflare Zero Trust dashboard](https://one.dash.cloudflare.com/) →
   **Networks → Tunnels → Create a tunnel** → **Cloudflared** connector.
2. Name it (e.g. `setu`), copy the **tunnel token** shown — this is what
   goes in `.env` as `CLOUDFLARE_TUNNEL_TOKEN`.
3. **Public Hostname** tab → add a hostname (a free `*.trycloudflare.com`
   works for testing; a real domain if you have one) → Service type `HTTP`,
   URL `ui:8000` — the Chainlit service's name and port **on the Docker
   Compose network**, not a host port, since `docker-compose.yml` publishes
   nothing to the host.

`docker-compose.yml`'s `cloudflared` service uses the official
`cloudflare/cloudflared:latest` image — confirmed via Docker Hub's own tag
listing to support `linux/arm64`, matching this instance's architecture.

## 5. Deploy

```sh
docker compose up -d --build
docker compose ps        # both api and ui should report "healthy"
docker compose logs -f   # watch startup; Ctrl-C to stop watching, containers keep running
```

The images build directly on the instance — no external registry involved.
Expect the `api` image build to include the ARM64 IndicLID model-bake step;
every dependency in `api/requirements.txt` has a confirmed prebuilt
`aarch64` wheel, so the `linux/arm64` build needs no compilation.

## 6. Keep it reachable — the idle-reclamation mitigation

Oracle reclaims Always Free instances if, over a rolling 7-day window, CPU,
network, **and** memory utilization (A1 shapes) all stay under 20%. An
intermittently-demoed project is a plausible match for "looks idle" — set up
a scheduled ping against a real endpoint (an uptime monitor like
UptimeRobot, or a cron job) hitting `https://<your-tunnel-hostname>/health`
every few minutes.

**Oracle is the one target among the five in `deploy/` where a keep-alive
monitor is the correct tool, not a counterproductive one — the opposite of
Zeabur.** Zeabur bills for awake-time, so a keep-alive there burns through
the free credit in days. Oracle doesn't meter uptime at all within its
OCPU-hour/GB-hour caps — the only risk is being reclaimed for looking *too
idle*, so keeping it awake costs nothing extra and directly addresses that
risk.

One gap worth flagging: Oracle's reclamation threshold is a rolling
utilization *percentage*, and Oracle's own documentation doesn't state
whether a lightweight `GET /health` ping alone generates enough measured
CPU/network/memory activity to clear 20%, versus needing something heavier.
**Check the actual utilization graphs in the Oracle console (Instance →
Metrics) after the first week** and confirm they read above the threshold —
don't assume the ping alone is sufficient without looking.

## 7. Verify

```sh
curl https://<your-tunnel-hostname>/  # UI should respond
```

Open the tunnel URL in a browser and run one real multilingual query
end-to-end.

## Restart behavior

`docker-compose.yml` sets `restart: unless-stopped` on every service — if
the instance itself reboots (a maintenance restart, a manual reboot), Docker
starts automatically on boot (default on Ubuntu after the install above) and
all three containers come back up without manual intervention.
