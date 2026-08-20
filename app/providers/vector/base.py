"""Abstract Base Interface for Vector Store Providers."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional
from app.chunking.models import Chunk
from app.providers.base import BaseProvider


class VectorStoreProvider(BaseProvider):
    """Abstract interface for vector database providers."""

    @abstractmethod
    def upsert(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        collection_name: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        """Insert or update chunks with dense embedding vectors.

        Args:
            chunks: List of text chunk objects.
            embeddings: Corresponding list of float vectors.
            collection_name: Optional target collection name.

        Returns:
            int: Number of records successfully upserted.
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        collection_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """Query vector index for nearest neighbors.

        Args:
            query_vector: Query dense vector.
            top_k: Maximum number of neighbors to return.
            collection_name: Optional collection name.
            filters: Optional metadata payload filters.

        Returns:
            List[Any]: Retrieved match objects or ScoredPoint instances.
        """
        pass

    @abstractmethod
    def count(self, collection_name: Optional[str] = None) -> int:
        """Return total number of points in the collection."""
        pass
