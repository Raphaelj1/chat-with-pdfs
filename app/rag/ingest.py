from pathlib import Path
from app.rag.loader import load_documents
from app.rag.splitter import split_documents
from app.rag.embeddings import embed_texts
from app.rag.vector_store import VectorStore


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