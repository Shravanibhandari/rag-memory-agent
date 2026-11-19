from app.config import settings


def build_llm():
    from langchain_groq import ChatGroq
    return ChatGroq(
        model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.1)
