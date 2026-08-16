# SETU — Architecture

A walkthrough of every component, written for someone reading the repo to judge
whether the engineering is sound. It explains **why** each choice was made, and
is honest about what does not work.

---

## 1. The problem this shape solves

Three constraints drive the whole design.

1. **The corpus is English; the users are not.** Scheme pages on myScheme are
   published in English. Users ask in Hindi, Tamil, and romanised Hinglish.
2. **Wrong answers here cause real harm.** A hallucinated citation is
   embarrassing. A hallucinated *"yes, you qualify for a ₹10 lakh subsidy"*
   sends someone to a government office with paperwork they cannot use.
3. **Government sources are messy.** Portal HTML mixes real content with
   navigation, hidden modals, and embedded SVG path data.

Constraint 1 gives translation at the edges. Constraint 2 gives a deterministic
eligibility engine that the LLM is forbidden to override. Constraint 3 gives an
ingestion filter with a quality floor.

---

## 2. Data flow

```
                          INGESTION (offline)
  myScheme pages ──Crawl4AI──▶ raw markdown
  local PDFs     ──Docling───▶ markdown  (pymupdf4llm fallback)
                       │
                       ▼
        normalize → strip boilerplate → parent/child chunk
                       │
                       ▼
           Jina embeddings-v3 (1024-d) ──▶ Qdrant Cloud `setu-schemes`
                                           dense (cosine) + sparse (BM25/IDF)

                          QUERY (online)
  user query
     │
     ├─ detect_language        IndicLID FTN/FTR → hin_Deva | tam_Taml | hin_Latn | en  (local)
     ├─ translate_query_to_en  Azure AI Translator  (API call, no-op if English)
     ├─ classify_intent        eligibility | procedure | list | benefit_amount
     ├─ extract_profile        natural language → {turnover_lakhs, state, ...}
     ├─ retrieve_hybrid    ◀───────────────┐    dense + BM25, fused by RRF in Qdrant Cloud
     ├─ rerank                             │    Jina reranker-v2 (API call)
     ├─ check_eligibility                  │    pure Python, no LLM
     ├─ generate_answer                    │    Groq (qwen3.6-27b), grounded, cites chunk ids
     ├─ self_critique ─────────────────────┘    faithfulness < 0.7 and retries < 2
     └─ translate_answer_from_en                Azure AI Translator, back to the user's language
```

Everything above the `retrieve_hybrid` fan-in used to run as ~6GB of locally-loaded
models (BGE-M3, bge-reranker-v2-m3, IndicTrans2). It's all API calls now — see
§5 and §9 for why, and what changed as a result.

---

## 3. Ingestion

### `ingestion/scraper.py`
Crawl4AI drives a real headless browser. This is not optional: myScheme is a
client-rendered SPA, and a plain HTTP GET returns an empty application shell.
Concurrency is capped at 3 — these are public government servers, not a CDN.
`CacheMode.BYPASS` is set because scheme rules change and a stale cache would
silently poison the index.

### `ingestion/parser.py`
Docling first, because it reconstructs tables, and eligibility criteria are
usually tabular. It falls back to pymupdf4llm on an exception **or on empty
output** — a blank parse is a failure wearing a disguise.

### `ingestion/chunker.py`
Two things happen here that matter.

**Boilerplate removal, in two mechanisms** — because one cannot catch both
problems:

| Problem | Shape | Mechanism |
|---|---|---|
| Site footer (`Quick Links`, `About Us`, DigiLocker) | mostly **letters** | truncate document at footer heading |
| Embedded SVG path data (`.028 .996 .112 …`) | mostly **digits** | reject below 55% letters |
| Header UI (`Ok`, `CancelSign In`, bare nav bullets) | mostly **letters** | exact whole-line match |

A character-ratio test alone would never catch the footer. A phrase list alone
would never catch the SVG. This removed **28 of 73 chunks (38%)** from the
corpus.

**Two details worth defending:**

- The letter-ratio test uses `str.isalpha()`, which is `True` for Devanagari and
  Tamil. An ASCII-based check would have silently deleted every Hindi and Tamil
  chunk the moment the Indic layer landed.
- UI lines match on the **whole stripped line**, never as substrings. So
  `"You may cancel your application within 30 days"` survives while a bare
  `"Cancel"` does not.

**Parent/child chunking.** Parents (~2048 tokens) are split first, then children
(~512 tokens) *within* each parent — so `parent_id` is a real containment link,
not a guess. Children are what get embedded and searched (precision); the parent
is what an LLM would read (context).

**Chunk ids are `sha256(source_url|kind|index|text)[:32]`, not random.**
Re-ingesting an unchanged document reproduces identical ids, which makes vector
upserts idempotent. Child ids are seeded with the parent id, so the same sentence
under two parents cannot collide.

---

## 4. Retrieval

### Why hybrid
Dense embeddings blur exact tokens — scheme acronyms, `Udyam`, `85%`. BM25 nails
those and misses paraphrase. Running both and fusing covers each other's failure
mode.

### Why fusion happens inside Qdrant
One collection holds two named vectors: `dense` (1024-d cosine) and `bm25`
(sparse, `Modifier.IDF`). A single `query_points` call runs both branches as
prefetches and merges them with `FusionQuery(fusion=Fusion.RRF)`.

**RRF merges on *rank*, not score.** The two branches never need comparable
scales — which is precisely where hand-rolled hybrid search goes wrong.

`Modifier.IDF` is load-bearing: without it, sparse search degenerates from BM25
into raw term counting.

### Point ids
Qdrant accepts only unsigned ints or UUIDs. A 32-char SHA-256 prefix is exactly
a UUID's worth of bits, so `uuid.UUID(hex=chunk_id)` converts losslessly and
deterministically. The original hex id is kept in the payload for citations.

### Reranking
Bi-encoders embed query and document separately — fast, approximate. A
cross-encoder reads the pair together — far too slow for a corpus, ideal for the
top ~20. Jina's `jina-reranker-v2-base-multilingual` runs this as an API call
rather than a locally-loaded ~2GB model. Measured pre-migration, on the older
45-chunk corpus and the local `bge-reranker-v2-m3`:

| Query | Top rerank score |
|---|---|
| "What documents are required for ZED certification?" (in corpus) | **0.977** |
| "Who is eligible for PMEGP?" (**not** in corpus) | **0.008** |

That two-orders-of-magnitude gap is what makes `rerank_score` a usable relevance
gate — the agent can say *"I don't have that scheme"* instead of confabulating.
The gap itself is a property of cross-encoder reranking generally, not this
specific model; the exact numbers haven't been re-measured against Jina's
reranker or the current 234-chunk corpus. `scripts/smoke_retrieval.py`
reproduces this check on demand.

---

## 5. The Indic layer

### `indic/lang_detect.py`
IndicLID ships **two** fastText classifiers, and which to use is decided from the
input's Unicode blocks — a romanised Hindi sentence contains no Devanagari at
all.

| Input | Model | Result |
|---|---|---|
| `ZED प्रमाणन के लिए…` | FTN (native) | `hin` / `Deva` 1.0000 |
| `இந்த திட்டத்திற்கு…` | FTN (native) | `tam` / `Taml` 1.0000 |
| `ZED certification ke liye…` | FTR (roman) | `hin` / `Latn` 0.9112 |

Two non-obvious details:

- IndicLID labels Tamil `tam_Tamil`; the standard FLORES-200 code is
  `tam_Taml`. Normalised in `SCRIPT_ALIASES`. Unhandled, this fails silently.
- fasttext 0.9.2's `predict()` calls `np.array(..., copy=False)`, **removed in
  NumPy 2**. We call the C++ binding `model.f.predict()` directly.

Language ID stays local deliberately — the model is small and fast, and
routing a sub-10ms call through a network round trip would only add latency
for a component that was never the RAM problem (see §9).

### `indic/translator.py`
Azure AI Translator, in both directions, via a plain HTTP call — not a
locally-loaded model. This replaced a local IndicTrans2 (200M distilled, both
directions) that cost ~2GB of RAM and decoded quadratically on CPU; Azure's
free F0 tier (2M characters/month) removed both the memory footprint and the
translation latency. Retrieval and generation still always happen in
English — translating at the edges is far cheaper and more accurate than
maintaining one index per language, independent of which backend translates.

**Citations are held out of translation.** A raw chunk-id marker sent to any
MT engine risks transliteration or mangling — `[57fd5bb0]` becoming
`[57எஃப்டி5பிபி0]`, destroying the citation. `_segment()` splits text into
translatable prose and verbatim citation markers, translates only the prose,
and reassembles byte-for-byte. This protection is backend-agnostic by design;
it survived the IndicTrans2 → Azure migration unchanged. `to_romanized()`
(below) reuses the same split/reassemble logic via a shared
`_segment_and_convert()` helper, so citation markers survive romanisation too.

**Hinglish answers come back in Latin script, not Devanagari.** A user who
types `ZED certification ke liye kya documents chahiye` wrote in Latin script;
getting a Devanagari answer back would be unreadable to them even though the
translation itself is correct. `IndicTranslator.to_romanized()` calls Azure's
separate `/transliterate` endpoint (`hi`/`ta` Deva/Taml → Latn — verified
against Microsoft's language-support table and live-tested against a real
key) after `from_english()` produces the native-script answer.
`translate_answer_from_en` triggers it whenever `state["script"] == "Latn"` —
**not** whenever `state["lang"] == "code_mixed"`, which was the first attempt
and was wrong: IndicLID reports `lang="hin", script="Latn"` (a single
confident language) for many genuinely-romanised queries, and only labels
input `code_mixed` when two languages both clear its confidence threshold.
Gating on `code_mixed` alone silently skipped the single-language case, caught
by testing against the real graph rather than trusting the mocked unit tests.

---

## 6. The agent

`agents/graph.py` is a LangGraph state machine over `SetuState`. Nodes return
only the keys they changed, so LangGraph merges updates and checkpoints each
step independently.

### The self-critique loop
`generate_answer → self_critique → (retry | finalize)`. On retry, the critic
supplies a rewritten query and the graph loops back to `retrieve_hybrid`.

**The termination logic lives in exactly one place.** `self_critique` sets
`final_answer` only when finalizing; `route_after_critique` just reads whether it
is set. Duplicating the threshold in the router is how these loops end up
disagreeing with themselves and spinning forever. A test proves a critic
hard-coded to always demand a retry still terminates at `retries == 2`.

### `agents/tools.py` — the deterministic part
`EligibilityChecker` evaluates rules from `data/eligibility_rules.json` with
plain Python comparisons. **No LLM is involved, by design.**

The critical rule: **missing information can never produce "eligible."** If a
rule's field is absent from the profile it is reported in `missing_fields` and
the verdict is `False`. Treating unknowns as passes would reintroduce exactly the
harm this component exists to prevent.

Observed on a real Gujarat profile:

```
samsmeata  eligible=True   fails=0  missing=0
mszcs      eligible=False  fails=0  missing=1   ← zero failures, still not eligible
msme-stes  eligible=False  fails=1  missing=2
```

The `mszcs` row is the point: no rule failed, but Udyam registration was never
stated, so the system asks rather than assumes.

The rules are transcribed from each scheme's `### Eligibility` section, and each
rule carries the portal's own wording in `description`.

### `agents/prompts.py`
Prompts are frozen dataclasses with a `version`. Every run records
`prompt_versions()` in its trace metadata, so a score regression can be traced
back to the exact wording that caused it.

---

## 7. Serving

`api/main.py` exposes `/query`, `/query/stream` (SSE), and `/health`. The graph
is compiled once at startup — rebuilding per request would discard the
checkpointer and reload models.

`/health` returns **`degraded`** rather than failing when a dependency is
unusable, so a load balancer can distinguish "up but can't answer" from "down".

`/query` and `/query/stream` require an `X-API-Key` header when `API_KEY` is
set (constant-time compared via `secrets.compare_digest`), and are rate
limited per client IP — a fixed-window in-process limiter, since a public
endpoint proxying metered LLM/API quota needs its own ceiling or one caller
can exhaust the daily budget for everyone. Both are opt-in, unset by default
for local development.

The Chainlit UI (`ui/chainlit_app.py`) is a thin HTTP client over
`/query/stream`, not an importer of the agent — it never loads a model, so it
can restart independently of the API and deploys as its own, much smaller,
container (`ui/Dockerfile`, `ui/requirements.txt`).

Both containers deploy the same way regardless of target: the root
`Dockerfile` and `ui/Dockerfile`, unmodified, built directly from this repo.
Five targets are documented under [`deploy/`](../deploy/) plus
[`render.yaml`](../render.yaml)/[`docs/DEPLOY.md`](DEPLOY.md) for Render —
each fits a different constraint, not a strict ranking:

| Target | Fits when |
|---|---|
| [`deploy/oracle/`](../deploy/oracle/) + [`docker-compose.yml`](../docker-compose.yml) | Continuous 24/7 uptime matters — the only target whose free allocation isn't metered by awake-time, so it's the one place a keep-alive monitor is the *correct* tool rather than counterproductive |
| [`deploy/zeabur/`](../deploy/zeabur/) | Occasional cold starts (a few seconds after idle) are acceptable — architecturally cannot bill you, since no payment method is ever attached to the account |
| [`deploy/gcloud/`](../deploy/gcloud/) | The strongest platform technically (configurable request timeout to 3600s, real SLA-backed infra) at the cost of a card on file |
| [`deploy/huggingface/`](../deploy/huggingface/) | Already paying for HF PRO for other reasons |
| Render (`render.yaml`, `docs/DEPLOY.md`) | The original target; caught two real bugs worth knowing about regardless of where you deploy — the API `Dockerfile` hardcoded port 8000 instead of binding Render's `$PORT`, and `retrieval/vectorstore.py`'s `from ingestion.chunker import Chunk` was never actually reachable in the API image because `Dockerfile` never copied `ingestion/` |

`setu-api`'s IndicLID models are pre-warmed during FastAPI's `lifespan`
startup hook and baked into the image at build time (`Dockerfile`), rather
than downloaded lazily on the first request — the first real query on a
cold container used to pay that cost inline; now it lands during startup,
before `/health` reports ready, on every target above alike.

---

## 8. Evaluation

`eval/gold_set.jsonl` — 25 queries (15 Hindi, 5 Tamil, 5 Hinglish). Every
`expected_answer` carries `source_chunk_ids` pointing at real chunks in the
corpus; none were invented.

Both evaluators score **`answer_en`**, not the translated output. Scoring the
translation would measure the translation API, not the RAG pipeline.

- `eval/ragas_eval.py` — faithfulness, answer relevancy, context precision,
  context recall, judged by Groq at `temperature=0` so runs are comparable.
  `--metrics` scores a subset, since the judge model's free-tier rate limit
  makes all four expensive to run together.
- `eval/deepeval_tests.py` — a 10-query CI gate at ≥ 0.7.

---

## 9. Known limitations

Stated plainly, because a reviewer will find them anyway.

| Limitation | Detail |
|---|---|
| **Corpus is 44 schemes** | 234 chunks. Anything else returns an honest "not in my sources". Grown from an original 5-scheme/45-chunk corpus via a broader official-source crawl; still a fraction of the hundreds of live MSME/welfare schemes. |
| **Source errors propagate** | The Design Scheme page reads `₹ 40,000,00/-` (likely ₹40,00,000). SETU reproduces it verbatim. Grounding works; the source is wrong. Cited, not corrected. |
| **Romanised Hindi vs Urdu** | Nearly indistinguishable transliterated; IndicLID sometimes prefers `urd_Latn`. Unaffected by the translation-backend migration — this is a language-*detection* limitation, upstream of translation. |
| **Post-migration faithfulness regression (0.933 → 0.857) is unresolved** | Re-measured against the current Jina/Azure/`qwen3.6-27b` stack; below the 0.90 target. Two evidenced causes (see `README.md` Evaluation section): Jina's reranker surfacing a tangentially-related chunk from a different scheme in at least one top-5, and `qwen3.6-27b` triggering the deterministic eligibility block — accurate rule-engine output, but invisible to a metric that checks claims against retrieved passages — more often than the prior model did. Neither is fixed yet. |
| **RAGAS needs a shim** | ragas 0.4.3 imports `langchain_community.chat_models.vertexai`, removed in langchain-community 0.4.x. A stub module is registered in `eval/ragas_eval.py`. Real workaround, documented as one. |
| **Token budgets are approximate** | 4 chars/token. Devanagari and Tamil tokenize less efficiently, so real Indic chunks exceed 512 tokens. |
| **This machine's chronic RAM pressure drove the cloud migration** | The pipeline originally held ~6GB locally (BGE-M3, the reranker, IndicTrans2) on a 16GB dev machine already committing far more than that between the OS, an IDE, and Docker — the direct cause of repeated `MemoryError`s and silent fallback degradation during development. Embeddings, reranking, and translation now run as Jina/Azure API calls specifically to remove that ceiling; IndicLID and the deterministic eligibility engine stay local because they were never the problem. |
| **Feedback persistence is a minimal stub, not a real data layer** | Chainlit's thumbs-up/down UI requires implementing its `BaseDataLayer` (18 methods) to appear at all. Only `upsert_feedback`/`delete_feedback` do real work, writing to a local SQLite file (`data/feedback.sqlite3`) — there's no thread-history/resume feature backing the other 16, which are deliberate no-ops. Fine for a single-instance deployment; would need a real backing store to survive across container restarts or multiple instances. |

---

## 10. Testing

189 tests total, in two tiers:

- **184 offline** (the default `pytest tests/` run) — no network, no LLM, no
  Qdrant server, no real API calls. External dependencies (Jina, Azure,
  Groq, Qdrant) are stubbed at the same `_request`-method boundary each
  client class calls internally, so a test never depends on model output —
  it depends on the code around the call.
- **5 integration** (`pytest -m integration`, excluded by default) — real
  Qdrant (in-memory), real embeddings and reranking calls, real end-to-end
  ingest→index→retrieve round trips. Added specifically because the fully
  offline suite had passed while a stale index and a provenance-dropping bug
  both shipped — mocks that are wrong in the same way as the code they mock
  can't catch what only shows up at the real seams.

Notable cases: the retry loop terminating under an adversarial critic; missing
profile fields never yielding "eligible"; Indic joiners (U+200C/U+200D) surviving
normalisation; citations surviving translation; blank `.env` secrets reading as
unset; the live-indexed Qdrant collection's point count matching `chunks.jsonl`
on disk.
