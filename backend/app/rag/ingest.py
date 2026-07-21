from pathlib import Path
from backend.app.rag.loader import load_documents
from backend.app.rag.splitter import split_documents
from backend.app.rag.embeddings import embed_texts
from backend.app.rag.vector_store import VectorStore


def ingest_documents (data_dir: Path) -> None:
    """
    Load, split, embed, and store documents.
    """
    
    documents = load_documents(data_dir)
    chunks = split_documents(documents)
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts)
    
    vector_store = VectorStore()
    
    vector_store.add_documents(
        documents=chunks,
        embeddings=embeddings
    )