"""Qdrant vector store abstraction for local and embedded collection management."""

import logging
import os
import uuid
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from app.chunking.models import Chunk
from app.config import get_settings

logger = logging.getLogger(__name__)

# Global registry of initialized local clients to prevent concurrent file locking
_CLIENT_CACHE: Dict[str, QdrantClient] = {}


def chunk_id_to_uuid(chunk_id: str) -> str:
    """Generate a deterministic UUID string from a chunk_id for Qdrant point ID compliance."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def get_shared_qdrant_client(location: Optional[str] = None, in_memory: bool = False) -> QdrantClient:
    """Return a shared singleton QdrantClient instance per path/URL to avoid file-lock conflicts."""
    if in_memory:
        return QdrantClient(":memory:")
    
    settings = get_settings()

    # If configured for Qdrant Cloud, connect to the remote cluster
    if settings.VECTOR_PROVIDER == "qdrant_cloud" and settings.VECTOR_DB_URL:
        cloud_url = settings.VECTOR_DB_URL.strip()
        if cloud_url not in _CLIENT_CACHE:
            logger.info("Initializing Qdrant Cloud client at %s", cloud_url)
            _CLIENT_CACHE[cloud_url] = QdrantClient(
                url=cloud_url,
                api_key=settings.VECTOR_DB_API_KEY,
            )
        return _CLIENT_CACHE[cloud_url]

    path = location or settings.QDRANT_PATH
    norm_path = os.path.normcase(os.path.abspath(path))

    if norm_path not in _CLIENT_CACHE:
        os.makedirs(norm_path, exist_ok=True)
        logger.info("Initializing persistent local Qdrant client at %s", norm_path)
        _CLIENT_CACHE[norm_path] = QdrantClient(path=norm_path)

    return _CLIENT_CACHE[norm_path]


class QdrantVectorStore:
    """Qdrant vector store handling collection lifecycle, batch upserts, and vector queries."""

    def __init__(
        self,
        location: Optional[str] = None,
        in_memory: bool = False,
        client: Optional[QdrantClient] = None,
    ):
        settings = get_settings()
        self.in_memory = in_memory
        self.path = None if in_memory else (location or settings.QDRANT_PATH)

        if client is not None:
            self.client = client
        else:
            self.client = get_shared_qdrant_client(location=self.path, in_memory=in_memory)

    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists in the vector store."""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception as e:
            logger.warning("Error checking collection existence for '%s': %s", collection_name, e)
            return False

    def create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> bool:
        """Create a collection if it does not already exist."""
        if not self.collection_exists(collection_name):
            logger.info(
                "Creating Qdrant collection '%s' [dim=%d, distance=%s]",
                collection_name,
                vector_size,
                distance,
            )
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=distance),
            )
            return True
        return False

    def recreate_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Recreate collection, dropping existing points if already present."""
        logger.info("Recreating Qdrant collection '%s' [dim=%d, distance=%s]", collection_name, vector_size, distance)
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

    def upsert_chunks(
        self,
        collection_name: str,
        chunks: List[Chunk],
        embeddings: Optional[List[List[float]]] = None,
        vectors: Optional[List[List[float]]] = None,
        batch_size: Optional[int] = None,
    ) -> int:
        """Upsert chunks with embeddings into Qdrant collection using deterministic point UUIDs."""
        if not chunks:
            return 0

        vec_list = embeddings if embeddings is not None else vectors
        if vec_list is None:
            raise ValueError("Must provide either 'embeddings' or 'vectors' list to upsert_chunks.")

        settings = get_settings()
        bs = batch_size or settings.QDRANT_UPSERT_BATCH_SIZE
        total_upserted = 0

        for i in range(0, len(chunks), bs):
            batch_chunks = chunks[i : i + bs]
            batch_embeddings = vec_list[i : i + bs]

            points = []
            for chunk, vector in zip(batch_chunks, batch_embeddings):
                point_id = chunk_id_to_uuid(chunk.chunk_id)
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "text": chunk.text,
                    "chunk_type": chunk.chunk_type,
                    "chunk_index": chunk.chunk_index,
                    "start_position": chunk.start_position,
                    "end_position": chunk.end_position,
                    "token_count": chunk.token_count,
                    "language": chunk.metadata.get("language"),
                    "source": chunk.metadata.get("source"),
                    "query_id": chunk.metadata.get("query_id"),
                    "passage_index": chunk.metadata.get("passage_index"),
                    "is_selected": chunk.metadata.get("is_selected", 0),
                    "metadata": chunk.metadata,
                }
                points.append(PointStruct(id=point_id, vector=vector, payload=payload))

            self.client.upsert(collection_name=collection_name, points=points)
            total_upserted += len(points)

        return total_upserted

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        query_filter: Optional[models.Filter] = None,
    ) -> List[Any]:
        """Perform vector similarity search using query_points API."""
        if hasattr(self.client, "query_points"):
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )
            return response.points
        else:
            return self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            )

    def count_points(self, collection_name: str) -> int:
        """Count total vectors in a collection."""
        try:
            return self.client.count(collection_name=collection_name).count
        except Exception:
            return 0

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Retrieve collection metrics, status, and vector counts."""
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "collection_name": collection_name,
                "exists": True,
                "status": info.status.name if hasattr(info.status, "name") else str(info.status),
                "points_count": info.points_count or 0,
                "vectors_count": getattr(info, "vectors_count", info.points_count),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", 0),
                "config": {
                    "vector_size": getattr(info.config.params.vectors, "size", None),
                    "distance": getattr(info.config.params.vectors, "distance", None),
                },
            }
        except Exception as e:
            logger.error("Error retrieving stats for collection '%s': %s", collection_name, e)
            return {"collection_name": collection_name, "exists": False, "error": str(e)}


@lru_cache()
def get_qdrant_store() -> QdrantVectorStore:
    """Return singleton cached QdrantVectorStore instance."""
    return QdrantVectorStore()
