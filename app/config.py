from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Empty defaults so pytest can import the app without real keys
    groq_api_key: str = ""
    cohere_api_key: str = ""

    groq_model: str = "llama-3.3-70b-versatile"
    embed_model: str = "embed-english-v3.0"
    rerank_model: str = "rerank-english-v3.0"  # verify against Cohere docs
    mem0_embed_model: str = "multi-qa-MiniLM-L6-cos-v1"
    mem0_embed_dims: int = 384

    # Local embedded Qdrant. Two DIFFERENT folders (local mode locks the folder)
    qdrant_path: str = "./data/qdrant"
    mem0_qdrant_path: str = "./data/qdrant_mem0"
    parents_file: str = "./data/parents.json"
    children_file: str = "./data/children.json"
    collection: str = "child_chunks"

    parent_size: int = 1000
    child_size: int = 200
    hybrid_top_k: int = 20      # candidates sent to reranker
    fetch_k: int = 40           # child chunks fetched from each index
    rerank_top_n: int = 5       # docs sent to the LLM
    min_relevance: float = 0.3  # drop chunks scoring below this
    history_turns: int = 6


settings = Settings()
