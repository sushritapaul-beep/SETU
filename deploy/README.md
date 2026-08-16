# Deploy targets

Five ways to run `setu-api` + `setu-ui` in the cloud, all built from the same
two unmodified Dockerfiles (root `Dockerfile`, `ui/Dockerfile`). None is
strictly "the" answer — pick by which constraint matters most:

| Target | Pick this when | Cost model |
|---|---|---|
| [`oracle/`](oracle/) | You need genuine 24/7 uptime, including via a keep-alive monitor | Perpetual free VM allocation (2 OCPU/12GB), not metered by uptime |
| [`zeabur/`](zeabur/) | Occasional cold starts (a few seconds after idle) are fine | No card ever on file — architecturally cannot bill you |
| [`gcloud/`](gcloud/) | You want the strongest platform technically and don't mind a card on file | Metered per-request; genuinely free within a generous monthly grant |
| [`huggingface/`](huggingface/) | You already pay for HF PRO for other reasons | $9/mo subscription (Docker Spaces aren't free on HF) |
| Render (`../render.yaml`, `../docs/DEPLOY.md`) | The original deploy target | Free tier, but `setu-api`'s RAM need exceeds it — see `docs/DEPLOYMENT_HANDOFF.md` §4 |

[`ui-standalone/`](ui-standalone/) is not a deploy target itself — it's a
shared, self-contained variant of `ui/Dockerfile` that the targets whose
build step requires the Dockerfile at the root of its own build context
(HF Spaces, Cloud Run's `--source` flow) both reuse instead of duplicating.

Full comparison, falsification pass, and the reasoning behind the Oracle
recommendation specifically: see the deployment audit referenced in this
repo's own history (search recent commits/PRs for "deployment audit" if the
published copy isn't at hand).
