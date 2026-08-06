---
title: SETU
emoji: 🇮🇳
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: ui/app.py
pinned: false
short_description: Multilingual cited RAG over Indian government MSME schemes
tags:
  - rag
  - multilingual
  - indic
  - hindi
  - tamil
  - msme
  - qdrant
---

# SETU — Scheme Eligibility & Translation Understanding for Bharat

Ask about Indian government MSME schemes in **English, Hindi, Tamil or
Hinglish**. Answers are grounded in official myScheme pages and cited.

> **This Space runs a reduced demo.** It answers from a bundled snapshot using
> hybrid retrieval, cross-encoder reranking and cited generation. The full
> version — IndicTrans2 translation, the deterministic eligibility engine and
> the LangGraph self-critique loop — runs locally. See the
> [GitHub repo](https://github.com/) for the complete system.

## What is included here

| Component | In this Space | Local only |
|---|---|---|
| Hybrid retrieval (dense + BM25 + RRF) | ✅ | |
| Cross-encoder reranking | ✅ | |
| Cited answer generation | ✅ | |
| IndicTrans2 translation | | ✅ |
| Deterministic eligibility engine | | ✅ |
| LangGraph self-critique loop | | ✅ |
| Langfuse tracing | | ✅ |

Translation is excluded because IndicTrans2 adds ~2GB of weights and decodes
quadratically on CPU (its remote modelling code forces `use_cache=False`). The
demo relies on the LLM's own multilingual ability instead.

## Required Space secret

Set this under **Settings → Variables and secrets**:

| Name | Value |
|---|---|
| `GROQ_API_KEY` | A free key from <https://console.groq.com> |

Optional: `SETU_STANDALONE=1` to force demo mode. A Space sets `SPACE_ID`
automatically, so the app already detects it.

## Files this Space needs

```
ui/app.py                 # entrypoint (app_file)
ui/standalone.py          # snapshot retrieval pipeline
retrieval/                # embeddings + reranker
agents/nodes.py           # LLM factory
config/settings.py
data/snapshot/            # chunks.jsonl + vectors.npy  ← generated, must be committed
requirements.txt
```

Generate the snapshot before pushing:

```bash
python scripts/ingest.py            # build the corpus
python scripts/export_snapshot.py   # write data/snapshot/
```

`data/snapshot/` is small (45 chunks × 1024 floats ≈ 200 KB) so it can be
committed directly with no need for Git LFS.

## Credits

Built with [AI4Bharat](https://ai4bharat.iitm.ac.in/) IndicLID and IndicTrans2,
[BAAI](https://huggingface.co/BAAI) BGE-M3 and bge-reranker-v2-m3,
[Qdrant](https://qdrant.tech/), [LangGraph](https://langchain-ai.github.io/langgraph/)
and [Groq](https://groq.com/). Scheme data from
[myScheme](https://www.myscheme.gov.in/), a Government of India platform.
