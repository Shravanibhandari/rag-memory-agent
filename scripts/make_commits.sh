#!/usr/bin/env bash
# Creates a staged git history (10 commits). Run once from the repo root.
set -e
git init -q 2>/dev/null || true
c() { git add $1 && git commit -q -m "$2" && echo "committed: $2"; }
c ".gitignore requirements.txt app/__init__.py app/retrieval/__init__.py tests/__init__.py" "chore: project scaffold"
c "app/config.py .env.example" "feat: add settings and .env.example (Groq + Cohere)"
c "app/ingestion.py data/docs" "feat: parent/child ingestion pipeline with sample docs"
c "app/retrieval/rrf.py tests/test_rrf.py tests/conftest.py" "feat: Reciprocal Rank Fusion + tests"
c "app/retrieval/hybrid.py" "feat: hybrid dense+BM25 retriever with parent mapping"
c "app/retrieval/rerank.py tests/test_rerank.py" "feat: Cohere ReRank compressor with score threshold + tests"
c "app/memory.py app/llm.py" "feat: Mem0 semantic memory and Groq LLM"
c "app/agent.py app/schemas.py" "feat: agent orchestration with latency telemetry"
c "app/main.py tests/test_api.py" "feat: FastAPI endpoints and API tests"
c "README.md postman" "docs: README and Postman collection"
