"""Agent orchestration: memory -> hybrid retrieve -> rerank -> generate -> save memory."""
import logging
import time
from typing import Any, Dict, List

log = logging.getLogger("rag.agent")

BASE_PROMPT = (
    "You are a helpful assistant. Answer using the provided context when relevant. "
    "If the context lacks the answer, say so briefly."
)


def _ms(start: float) -> float:
    return max(round((time.perf_counter() - start) * 1000, 3), 0.001)


def build_system_prompt(memories: List[str]) -> str:
    if not memories:
        return BASE_PROMPT
    profile = "\n".join(f"- {m}" for m in memories)
    prompt = f"{BASE_PROMPT}\n\nUser profile (from long-term memory):\n{profile}\n\nFollow these preferences strictly."
    low = profile.lower()
    if "code-only" in low or "code only" in low:
        prompt += ("\nIf the profile says code-only, output ONLY a single code block "
                   "in the user's preferred language, with NO prose, NO explanation, NO comments outside the block.")
    elif "brief" in low or "concise" in low:
        prompt += "\nKeep answers brief."
    return prompt


class RAGAgent:
    def __init__(self, retriever: Any, compressor: Any, memory: Any, llm: Any,
                 history_turns: int = 6):
        self.retriever, self.compressor = retriever, compressor
        self.memory, self.llm = memory, llm
        self.history_turns = history_turns
        self.sessions: Dict[str, List[tuple]] = {}

    def chat(self, query: str, user_id: str, session_id: str) -> dict:
        t_all = time.perf_counter()
        memories = self.memory.get_all(user_id)

        t = time.perf_counter()
        candidates = self.retriever.invoke(query)
        retrieval_ms = _ms(t)

        t = time.perf_counter()
        docs = list(self.compressor.compress_documents(candidates, query))
        rerank_ms = _ms(t)
        log.info("hybrid: %d -> rerank: %d", len(candidates), len(docs))

        context = "\n\n---\n\n".join(d.page_content for d in docs) or "(no relevant context found)"
        history = self.sessions.setdefault(session_id, [])
        messages = [("system", build_system_prompt(memories))]
        messages += history[-self.history_turns * 2:]
        messages.append(("user", f"Context:\n{context}\n\nQuestion: {query}"))

        t = time.perf_counter()
        answer = self.llm.invoke(messages).content
        generation_ms = _ms(t)

        history += [("user", query), ("assistant", answer)]
        try:
            self.memory.add(user_id, query)
        except Exception as e:  # memory failure must not break chat
            log.warning("memory save failed: %s", e)

        return {
            "answer": answer,
            "sources": [{"parent_id": d.metadata.get("parent_id"),
                         "source": d.metadata.get("source"),
                         "relevance_score": d.metadata.get("relevance_score")} for d in docs],
            "telemetry": {
                "retrieval_latency_ms": retrieval_ms,
                "rerank_latency_ms": rerank_ms,
                "generation_latency_ms": generation_ms,
                "total_latency_ms": _ms(t_all),
            },
        }


def build_agent() -> RAGAgent:
    from app.config import settings
    from app.llm import build_llm
    from app.memory import MemoryStore
    from app.retrieval.hybrid import build_hybrid_retriever
    from app.retrieval.rerank import build_compressor
    return RAGAgent(build_hybrid_retriever(), build_compressor(), MemoryStore(),
                    build_llm(), settings.history_turns)
