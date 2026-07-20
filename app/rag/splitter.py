from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from app.models.document import Document


def split_document(document: Document) -> list[Document]:
    """
    Split a single document into overlapping chunks.
    """

    chunks: list[Document] = []

    text = document.text

    start = 0
    chunk_number = 1

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk_text = text[start:end]

        metadata = document.metadata.copy()
        metadata["chunk"] = chunk_number

        chunks.append(
            Document(
                text=chunk_text,
                metadata=metadata,
            )
        )

        start += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_number += 1

    return chunks


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split multiple documents into chunks.
    """

    chunks: list[Document] = []

    for document in documents:
        chunks.extend(split_document(document))

    return chunks