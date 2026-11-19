from typing import List
from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    user_id: str
    session_id: str


class Telemetry(BaseModel):
    retrieval_latency_ms: float
    rerank_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    telemetry: Telemetry


class MemoriesResponse(BaseModel):
    user_id: str
    memories: List[str]
