from chromadb import PersistentClient
from app.config import DB_DIR
from app.models.document import Document
from app.utils.ids import generate_chunk_id


class VectorStore:
    def __init__(self):
        self.client = PersistentClient(path=str(DB_DIR))
        self.collection = self.client.get_or_create_collection(name="documents")
    
    
    def add_documents(
        self,
        documents: list[Document],
        embeddings: list[list[float]]
    ) -> None:
        ids = [generate_chunk_id(doc) for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
    
    def search(
        self,
        query_embedding: list[float],
        limit: int = 5
    ) -> list[Document]:
        """
        Search for the most similar documents.
        """
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=limit
        )
        
        documents: list[Document] = []
        
        retrieved_documents = results["documents"][0]
        retrieved_metadatas = results["metadatas"][0]
        
        for text, metadata in zip(
            retrieved_documents,
            retrieved_metadatas
        ):
            documents.append(
                Document(
                    text=text,
                    metadata=metadata
                )
            )
            
        return documents