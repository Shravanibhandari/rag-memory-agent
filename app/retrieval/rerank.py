"""Cohere ReRank wrapped with a minimum-relevance filter."""
import logging
from typing import Any, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import BaseDocumentCompressor, Document

from app.config import settings

log = logging.getLogger("rag.rerank")


class ThresholdReranker(BaseDocumentCompressor):
    """Runs a rerank compressor (CohereRerank, top_n=5) then drops chunks
    whose metadata['relevance_score'] is below min_score."""

    reranker: Any
    min_score: float = 0.3
    top_n: int = 5

    def compress_documents(
        self, documents: Sequence[Document], query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        ranked = list(self.reranker.compress_documents(documents, query))
        kept = [d for d in ranked
                if d.metadata.get("relevance_score", 0.0) >= self.min_score]
        kept = kept[: self.top_n]
        log.info("rerank: %d candidates -> %d ranked -> %d kept (min_score=%.2f)",
                 len(documents), len(ranked), len(kept), self.min_score)
        return kept


def build_compressor() -> ThresholdReranker:
    from langchain_cohere import CohereRerank
    cohere = CohereRerank(
        model=settings.rerank_model, top_n=settings.rerank_top_n,
        cohere_api_key=settings.cohere_api_key)
    return ThresholdReranker(
        reranker=cohere, min_score=settings.min_relevance, top_n=settings.rerank_top_n)


def build_compression_retriever(hybrid_retriever):
    """Brief-style wiring (ContextualCompressionRetriever)."""
    from langchain.retrievers import ContextualCompressionRetriever
    return ContextualCompressionRetriever(
        base_compressor=build_compressor(), base_retriever=hybrid_retriever)
