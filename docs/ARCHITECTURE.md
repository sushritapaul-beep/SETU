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
              BGE-M3 (1024-d) ──▶ Qdrant `setu_schemes`
                                  dense (cosine) + sparse (BM25/IDF)

                          QUERY (online)
  user query
     │
     ├─ detect_language        IndicLID FTN/FTR → hin_Deva | tam_Taml | hin_Latn | en
     ├─ translate_query_to_en  IndicTrans2 200M  (no-op if English)
     ├─ classify_intent        eligibility | procedure | list | benefit_amount
     ├─ extract_profile        natural language → {turnover_lakhs, state, ...}
     ├─ retrieve_hybrid    ◀───────────────┐    dense + BM25, fused by RRF in Qdrant
     ├─ rerank                             │    bge-reranker-v2-m3 cross-encoder
     ├─ check_eligibility                  │    pure Python, no LLM
     ├─ generate_answer                    │    grounded, cites chunk ids
     ├─ self_critique ─────────────────────┘    faithfulness < 0.7 and retries < 2
     └─ translate_answer_from_en                back to the user's language
```

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
top ~20. Measured on this corpus:

| Query | Top rerank score |
|---|---|
| "What documents are required for ZED certification?" (in corpus) | **0.977** |
| "Who is eligible for PMEGP?" (**not** in corpus) | **0.008** |

That two-orders-of-magnitude gap makes `rerank_score` a usable relevance gate:
the agent can say *"I don't have that scheme"* instead of confabulating.

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

- IndicLID labels Tamil `tam_Tamil`; FLORES-200 and IndicTrans2 expect
  `tam_Taml`. Normalised in `SCRIPT_ALIASES`. Unhandled, this fails silently.
- fasttext 0.9.2's `predict()` calls `np.array(..., copy=False)`, **removed in
  NumPy 2**. We call the C++ binding `model.f.predict()` directly.

### `indic/translator.py`
IndicTrans2 distilled 200M, in both directions. Retrieval and generation always
happen in English — translating at the edges is far cheaper and more accurate
than maintaining one index per language.

**Citations are held out of translation.** Without this, IndicTrans2
transliterates them: `[57fd5bb0]` became `[57எஃப்டி5பிபி0]`, destroying the
citation. `_segment()` splits text into translatable prose and verbatim citation
markers, translates only the prose, and reassembles byte-for-byte.

**Known constraint:** `use_cache=False` is mandatory. IndicTrans2's remote
modelling code reads `past_key_values[0][0].shape` — the legacy tuple cache —
while modern transformers passes an `EncoderDecoderCache` whose first entry is
`None`. Disabling the cache avoids that path but makes decoding quadratic.
Query translation is ~1s; long answers are much slower.

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

The Gradio UI is a thin HTTP client, not an importer of the agent. The heavy
models live in one process; the UI can be restarted without reloading 6 GB.

---

## 8. Evaluation

`eval/gold_set.jsonl` — 25 queries (15 Hindi, 5 Tamil, 5 Hinglish). Every
`expected_answer` carries `source_chunk_ids` pointing at real chunks in the
corpus; none were invented.

Both evaluators score **`answer_en`**, not the translated output. Scoring the
translation would measure IndicTrans2, not the RAG pipeline.

- `eval/ragas_eval.py` — faithfulness, answer relevancy, context precision,
  context recall, judged by Groq Llama-3.3 at `temperature=0` so runs are
  comparable.
- `eval/deepeval_tests.py` — a 10-query CI gate at ≥ 0.7.

---

## 9. Known limitations

Stated plainly, because a reviewer will find them anyway.

| Limitation | Detail |
|---|---|
| **Corpus is 5 schemes** | 45 chunks. Anything else returns an honest "not in my sources". |
| **Source errors propagate** | The Design Scheme page reads `₹ 40,000,00/-` (likely ₹40,00,000). SETU reproduces it verbatim. Grounding works; the source is wrong. Cited, not corrected. |
| **Hinglish answers return in Devanagari** | IndicTrans2 has no romanised output mode. A Hinglish speaker who cannot read Devanagari is poorly served. Fixing this needs a transliteration step (IndicXlit). |
| **Romanised Hindi vs Urdu** | Nearly indistinguishable transliterated; IndicLID sometimes prefers `urd_Latn`. |
| **Quadratic decoding** | `use_cache=False` is forced by IndicTrans2's cache assumption. Long answer translation is slow on CPU. |
| **Memory** | The full pipeline holds ~6 GB. On a 16 GB machine already committing 22 GB, Windows raises `os error 1455` and features fail silently into their fallbacks. Run the API and UI in separate processes. |
| **RAGAS needs a shim** | ragas 0.4.3 imports `langchain_community.chat_models.vertexai`, removed in langchain-community 0.4.x. A stub module is registered in `eval/ragas_eval.py`. Real workaround, documented as one. |
| **Token budgets are approximate** | 4 chars/token. Devanagari and Tamil tokenize less efficiently, so real Indic chunks exceed 512 tokens. |
| **No auth, no rate limiting** | The API is a demo, not a deployment. |

---

## 10. Testing

93 offline tests. No network, no LLM, no Qdrant server, no model downloads —
external dependencies are injected through module-level getters and replaced with
stubs.

Notable cases: the retry loop terminating under an adversarial critic; missing
profile fields never yielding "eligible"; Indic joiners (U+200C/U+200D) surviving
normalisation; citations surviving translation; blank `.env` secrets reading as
unset.
