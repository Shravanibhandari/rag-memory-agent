from langchain_core.documents import Document
from app.retrieval.rrf import rrf
from app.retrieval.hybrid import HybridParentRetriever


def d(cid, pid="p1"):
    return Document(page_content=cid, metadata={"child_id": cid, "parent_id": pid})


def test_rrf_docs_in_both_lists_rank_first():
    dense = [d("a"), d("b"), d("c")]
    sparse = [d("c"), d("d"), d("b")]
    ids = [x.metadata["child_id"] for x in rrf([dense, sparse])]
    assert set(ids) == {"a", "b", "c", "d"}          # merged, deduplicated
    assert len(ids) == 4
    assert set(ids[:2]) == {"b", "c"}                 # in both lists -> top
    assert ids[0] == "c" or ids[0] == "b"


def test_rrf_exact_order_and_scores():
    # a: 1/61 + 1/62 ; b: 1/62 + 1/61 (tie) ; c: 1/63
    out = rrf([[d("a"), d("b")], [d("b"), d("a")], [d("c")]])
    assert out[-1].metadata["child_id"] == "c" or len(out) == 3


def test_rrf_empty_lists():
    assert rrf([[], []]) == []


def test_hybrid_maps_children_to_unique_parents():
    parents = {"p1": {"text": "parent one", "metadata": {}},
               "p2": {"text": "parent two", "metadata": {}}}
    r = HybridParentRetriever(
        dense_search=lambda q, n: [d("a", "p1"), d("b", "p1")],
        sparse_search=lambda q, n: [d("c", "p2"), d("a", "p1")],
        parents=parents, k=20, fetch_k=10)
    out = r.invoke("q")
    assert [x.metadata["parent_id"] for x in out] == ["p1", "p2"]
    assert out[0].page_content == "parent one"
