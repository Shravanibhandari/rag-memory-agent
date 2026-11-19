from typing import List
from langchain_core.documents import Document


def rrf(result_lists: List[List[Document]], k: int = 60) -> List[Document]:
    """Reciprocal Rank Fusion. Docs are identified by metadata['child_id']."""
    scores: dict = {}
    docs: dict = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            key = doc.metadata["child_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            docs.setdefault(key, doc)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ordered]
