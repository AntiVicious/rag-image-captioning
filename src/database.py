"""
ChromaDB operations: a thin, typed wrapper around the persistent collection
used to store CLIP embeddings and their captions.
"""
from typing import Dict, List, Sequence

from .config import Config


class DatabaseManager:
    """Owns a ChromaDB collection for a given Config."""

    def __init__(self, config: Config):
        self.config = config
        self.client = None
        self.collection = None

    def initialize(self):
        """Create/open the persistent ChromaDB collection."""
        import chromadb

        self.client = chromadb.PersistentClient(path=self.config.chroma_db_dir)
        existing = [c.name for c in self.client.list_collections()]
        if self.config.collection_name in existing:
            self.collection = self.client.get_collection(self.config.collection_name)
        else:
            self.collection = self.client.create_collection(self.config.collection_name)
        return self.collection

    def _get_collection(self):
        if self.collection is None:
            self.initialize()
        return self.collection

    def add_embeddings(self, embeddings: Sequence, documents: List[str], ids: List[str]) -> None:
        """Add a batch of (embedding, caption, id) rows to the collection."""
        collection = self._get_collection()
        collection.add(embeddings=[list(e) for e in embeddings], documents=documents, ids=ids)

    def query_similar(self, query_embedding: Sequence, top_k: int = 5) -> Dict:
        """Return the top_k nearest captions to a query embedding."""
        collection = self._get_collection()
        return collection.query(query_embeddings=[list(query_embedding)], n_results=top_k)

    def get_stats(self) -> Dict:
        """Return basic statistics about the collection."""
        collection = self._get_collection()
        return {"total_embeddings": collection.count(), "collection_name": self.config.collection_name}
