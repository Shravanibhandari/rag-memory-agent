from langchain_core.documents import Document
from app.retrieval.rerank import ThresholdReranker


class FakeCohere:
    def __init__(self, scores):
        self.scores = scores

    def compress_documents(self, documents, query, callbacks=None):
        return [Document(page_content=doc.page_content,
                         metadata={**doc.metadata, "relevance_score": s})
                for doc, s in zip(documents, self.scores)]


def docs(n):
    return [Document(page_content=f"doc{i}", metadata={}) for i in range(n)]


def test_drops_chunks_below_threshold():
    rr = ThresholdReranker(reranker=FakeCohere([0.9, 0.7, 0.35, 0.2, 0.05]),
                           min_score=0.3, top_n=5)
    out = rr.compress_documents(docs(5), "q")
    assert [d.page_content for d in out] == ["doc0", "doc1", "doc2"]
    assert all(d.metadata["relevance_score"] >= 0.3 for d in out)


def test_caps_at_top_n():
    rr = ThresholdReranker(reranker=FakeCohere([0.9] * 20), min_score=0.3, top_n=5)
    assert len(rr.compress_documents(docs(20), "q")) == 5


def test_all_below_threshold_returns_empty():
    rr = ThresholdReranker(reranker=FakeCohere([0.1, 0.2]), min_score=0.3, top_n=5)
    assert rr.compress_documents(docs(2), "q") == []
