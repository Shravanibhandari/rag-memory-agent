from types import SimpleNamespace
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from app.agent import RAGAgent, build_system_prompt
from app.main import app, get_agent


class FakeRetriever:
    def invoke(self, q):
        return [Document(page_content=f"c{i}", metadata={"parent_id": str(i)}) for i in range(20)]


class FakeCompressor:
    def compress_documents(self, docs, q):
        return [Document(page_content=d.page_content,
                         metadata={**d.metadata, "relevance_score": 0.9}) for d in docs[:5]]


class FakeMemory:
    def __init__(self): self.store = {}
    def get_all(self, uid): return self.store.get(uid, [])
    def add(self, uid, text): self.store.setdefault(uid, []).append("Prefers Python")


class FakeLLM:
    def invoke(self, messages):
        self.last = messages
        return SimpleNamespace(content="ok")


def make_client():
    llm, mem = FakeLLM(), FakeMemory()
    agent = RAGAgent(FakeRetriever(), FakeCompressor(), mem, llm)
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app), llm


def test_chat_telemetry_and_reduction():
    client, _ = make_client()
    r = client.post("/api/chat", json={"query": "hi", "user_id": "u", "session_id": "s"})
    assert r.status_code == 200
    body = r.json()
    for k in ("retrieval_latency_ms", "rerank_latency_ms", "generation_latency_ms", "total_latency_ms"):
        assert body["telemetry"][k] > 0
    assert len(body["sources"]) == 5


def test_memory_endpoint_and_persona_injection():
    client, llm = make_client()
    client.post("/api/chat", json={"query": "I use Python", "user_id": "usr_99", "session_id": "s1"})
    assert "Prefers Python" in client.get("/api/users/usr_99/memories").json()["memories"]
    client.post("/api/chat", json={"query": "merge sort", "user_id": "usr_99", "session_id": "s2"})
    assert "Prefers Python" in llm.last[0][1]


def test_code_only_prompt():
    assert "ONLY a single code block" in build_system_prompt(["Prefers brief, code-only responses"])
