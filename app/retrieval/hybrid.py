"""Hybrid (dense + BM25) retrieval fused by RRF, with child->parent mapping."""
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import settings
from app.retrieval.rrf import rrf

log = logging.getLogger("rag.hybrid")


class HybridParentRetriever(BaseRetriever):
    """dense top-N + sparse top-N -> RRF -> unique parents -> top k parents."""

    dense_search: Any   # Callable[[str, int], List[Document]]
    sparse_search: Any  # Callable[[str, int], List[Document]]
    parents: Dict[str, Any]
    k: int = 20
    fetch_k: int = 40

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        dense = self.dense_search(query, self.fetch_k)
        sparse = self.sparse_search(query, self.fetch_k)
        fused = rrf([dense, sparse])

        out, seen = [], set()
        for child in fused:
            pid = child.metadata.get("parent_id")
            if pid in seen or pid not in self.parents:
                continue
            seen.add(pid)
            p = self.parents[pid]
            out.append(Document(
                page_content=p["text"],
                metadata={**p.get("metadata", {}), "parent_id": pid}))
            if len(out) >= self.k:
                break
        log.info("hybrid: dense=%d sparse=%d fused=%d -> %d parents",
                 len(dense), len(sparse), len(fused), len(out))
        return out


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def build_hybrid_retriever() -> HybridParentRetriever:
    from rank_bm25 import BM25Okapi
    from langchain_cohere import CohereEmbeddings
    from langchain_qdrant import QdrantVectorStore

    parents = json.loads(Path(settings.parents_file).read_text(encoding="utf-8"))
    raw_children = json.loads(Path(settings.children_file).read_text(encoding="utf-8"))
    children = [Document(page_content=c["text"], metadata=c["metadata"]) for c in raw_children]
    bm25 = BM25Okapi([_tokenize(c.page_content) for c in children])

    store = QdrantVectorStore.from_existing_collection(
        embedding=CohereEmbeddings(
            model=settings.embed_model, cohere_api_key=settings.cohere_api_key),
        path=settings.qdrant_path, collection_name=settings.collection)

    def dense_search(q: str, n: int) -> List[Document]:
        return store.similarity_search(q, k=n)

    def sparse_search(q: str, n: int) -> List[Document]:
        scores = bm25.get_scores(_tokenize(q))
        top = sorted(range(len(children)), key=lambda i: scores[i], reverse=True)[:n]
        return [children[i] for i in top if scores[i] > 0]

    return HybridParentRetriever(
        dense_search=dense_search, sparse_search=sparse_search, parents=parents,
        k=settings.hybrid_top_k, fetch_k=settings.fetch_k)
