from backend.app.rag.embeddings import embed_text
from backend.app.rag.vector_store import VectorStore
from backend.app.models.document import Document


def retrieve(
    query: str,
    vector_store: VectorStore,
    limit: int = 5,
) -> list[Document]:
    """
    Retrieve the most relevant document chunks for a query.
    """

    query_embedding = embed_text(query)

    return vector_store.search(
        query_embedding=query_embedding,
        limit=limit,
    )