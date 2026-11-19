import logging

from fastapi import Depends, FastAPI

from app.agent import RAGAgent, build_agent
from app.schemas import ChatRequest, ChatResponse, MemoriesResponse

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Advanced RAG Memory Agent")

_agent = None


def get_agent() -> RAGAgent:
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, agent: RAGAgent = Depends(get_agent)):
    return agent.chat(req.query, req.user_id, req.session_id)


@app.get("/api/users/{user_id}/memories", response_model=MemoriesResponse)
def memories(user_id: str, agent: RAGAgent = Depends(get_agent)):
    return {"user_id": user_id, "memories": agent.memory.get_all(user_id)}
