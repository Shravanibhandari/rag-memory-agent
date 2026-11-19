# Advanced RAG Memory Agent

* **Repository**: \[https://github.com/Shravanibhandari/rag-memory-agent]

Hybrid (dense + BM25, RRF) parent-document retrieval, Cohere ReRank context
compression (20 → 5), Mem0 long-term memory, FastAPI + latency telemetry.

> \*\*Note on OpenAI → Groq:\*\* the brief mentions OpenAI. This project uses \*\*Groq\*\*
> (`llama-3.3-70b-versatile`) for chat generation and Mem0 fact extraction, because
> Groq has no embedding models. Dense embeddings use \*\*Cohere\*\* `embed-english-v3.0`;
> Mem0 uses a local HuggingFace embedder (`multi-qa-MiniLM-L6-cos-v1`).

## Architecture

```
query ─► Mem0 profile ─► hybrid (dense + BM25 → RRF → parents, top 20)
      ─► Cohere ReRank (top 5, min score 0.3) ─► Groq LLM (profile in system prompt)
      ─► save facts to Mem0 ─► response + telemetry
```

## Installation

1. `python -m venv .venv \&\& source .venv/bin/activate` (Windows: `.venv\\Scripts\\activate`)
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill `GROQ\_API\_KEY`, `COHERE\_API\_KEY`
4. Put `.txt` files in `data/docs/` (sample docs included; add more, 30+ parents recommended)
5. Ingest: `python -m app.ingestion`
6. Run: `uvicorn app.main:app --reload`
7. Tests: `pytest` (no real API calls)

Note: `--reload` can lock local Qdrant; if you see a lock error, run without `--reload`.

## Endpoints

* `POST /api/chat` — `{query, user\_id, session\_id}` → `{answer, sources, telemetry}`
* `GET /api/users/{user\_id}/memories` — stored semantic facts

Telemetry: `retrieval\_latency\_ms`, `rerank\_latency\_ms`, `generation\_latency\_ms`, `total\_latency\_ms`.
Logs show candidate reduction, e.g. `hybrid: 20 -> rerank: 5`.

## Verification

1. **Latency**: POST `/api/chat`, check the 4 telemetry values are positive.
2. **Memory**: as `usr\_99`, send "Hello! I work in Python and prefer brief, code-only responses."
→ GET `/api/users/usr\_99/memories` → new `session\_id` → "Write a quick merge sort routine."
3. **Tests**: `pytest`

Import `postman/rag-agent.postman\_collection.json` into Postman/Bruno.

## Git history

`bash scripts/make\_commits.sh` creates a staged 10-commit history (or commit as you go).

