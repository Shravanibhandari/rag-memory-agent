"""Long-term semantic memory using Mem0 (Groq LLM + local HF embeddings)."""
import os
from typing import Any, List

from app.config import settings


def _texts(res: Any) -> List[str]:
    items = res.get("results", []) if isinstance(res, dict) else (res or [])
    return [i["memory"] for i in items if isinstance(i, dict) and "memory" in i]


class MemoryStore:
    def __init__(self, mem: Any = None):
        self._mem = mem

    @property
    def mem(self):
        if self._mem is None:
            from mem0 import Memory
            os.environ.setdefault("GROQ_API_KEY", settings.groq_api_key)
            self._mem = Memory.from_config({
                "llm": {"provider": "groq", "config": {
                    "model": settings.groq_model, "api_key": settings.groq_api_key}},
                "embedder": {"provider": "huggingface", "config": {
                    "model": settings.mem0_embed_model}},
                "vector_store": {"provider": "qdrant", "config": {
                    "collection_name": "user_memories",
                    "path": settings.mem0_qdrant_path,
                    "embedding_model_dims": settings.mem0_embed_dims}},
            })
        return self._mem

    def get_all(self, user_id: str) -> List[str]:
        return _texts(self.mem.get_all(user_id=user_id))

    def add(self, user_id: str, text: str) -> None:
        """Extract + persist facts from a user message."""
        self.mem.add([{"role": "user", "content": text}], user_id=user_id)
