# import chromadb
# from chromadb.api.models.Collection import Collection
from chromadb import PersistentClient, Collection

from app.config import DB_DIR


class CollectionManager:
    def __init__(self):
        self.client = PersistentClient(path=DB_DIR)

    
    def _collection_name(self, session_id: str) -> str:
        return f"session_{session_id}"

    
    def get_or_create_collection(
        self,
        session_id: str,
    ) -> Collection:
        return self.client.get_or_create_collection(
            name=self._collection_name(session_id)
        )


    def get_collection(
        self,
        session_id: str,
    ) -> Collection:
        return self.client.get_collection(
            name=self._collection_name(session_id)
        )


    def delete_collection(
        self,
        session_id: str,
    ) -> None:
        self.client.delete_collection(
            name=self._collection_name(session_id)
        )


    def list_collections(self):
        return self.client.list_collections()