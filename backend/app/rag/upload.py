from app.models.document import UploadedDocument
from app.rag.collection_manager import CollectionManager
from app.rag.embeddings import embed_texts
from app.rag.loader import load_pdf_stream
from app.rag.splitter import split_documents
from app.rag.vector_store import VectorStore


def upload_documents(
    session_id: str,
    uploaded_documents: list[UploadedDocument],
) -> None:
    """
    Process and index uploaded PDFs for a session.
    """

    manager = CollectionManager()

    collection = manager.get_or_create_collection(session_id)

    vector_store = VectorStore(collection)

    for uploaded_document in uploaded_documents:

        documents = load_pdf_stream(
            file=uploaded_document.file,
            filename=uploaded_document.filename,
        )

        chunks = split_documents(documents)
        
        texts = [chunk.text for chunk in chunks]

        embeddings = embed_texts(texts)

        vector_store.add_documents(
            chunks=chunks,
            embeddings=embeddings,
        )