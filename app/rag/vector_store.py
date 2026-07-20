import uuid
from chromadb import PersistentClient
from app.config import DB_DIR
from app.models.document import Document


class VectorStore:
    def __init__(self):
        self.client = PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_or_create_collection(name="documents")
    
    
    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]]
    ) -> None:
        ids = [str(uuid.uuid4()) for _ in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )