"""Ingest .txt docs: parents (~1000 chars) -> JSON docstore,
children (~200 chars) -> local Qdrant (dense) + JSON (for BM25)."""
import json
import uuid
from pathlib import Path

from app.config import settings


def ingest(docs_dir: str = "data/docs") -> None:
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_cohere import CohereEmbeddings
    from langchain_qdrant import QdrantVectorStore

    raw_docs = DirectoryLoader(
        docs_dir, glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    ).load()

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_size, chunk_overlap=100)
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_size, chunk_overlap=20)

    parents, children = {}, []
    for parent in parent_splitter.split_documents(raw_docs):
        pid = str(uuid.uuid4())
        parents[pid] = {"text": parent.page_content, "metadata": parent.metadata}
        for child in child_splitter.split_documents([parent]):
            child.metadata.update({"parent_id": pid, "child_id": str(uuid.uuid4())})
            children.append(child)

    Path(settings.parents_file).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.parents_file).write_text(json.dumps(parents), encoding="utf-8")
    Path(settings.children_file).write_text(
        json.dumps([{"text": c.page_content, "metadata": c.metadata} for c in children]),
        encoding="utf-8")

    embeddings = CohereEmbeddings(
        model=settings.embed_model, cohere_api_key=settings.cohere_api_key)
    QdrantVectorStore.from_documents(
        children, embeddings,
        path=settings.qdrant_path, collection_name=settings.collection,
        force_recreate=True,
    )
    print(f"Ingested {len(parents)} parents and {len(children)} children")


if __name__ == "__main__":
    ingest()
