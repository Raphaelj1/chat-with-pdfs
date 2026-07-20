import hashlib
from app.models.document import Document


def generate_chunk_id(document: Document) -> str:
    """
    Generate a deterministic ID for a document chunk.
    """

    content = (
        f"{document.metadata['source']}"
        f"{document.metadata['page']}"
        f"{document.metadata['chunk']}"
        f"{document.text}"
    )

    return hashlib.sha256(content.encode("utf-8")).hexdigest()